#!/usr/bin/env python3
"""The rules of a conversion, each tested on the graphs of real organisms against Ensembl's own tables.

A conversion that starts and ends at one release has nothing to do with the id history: what it must return is
written in Ensembl's tables of that release. So the first part of this module asks such conversions for a spread
of releases (the first and the last, both sides of every change of assembly, every seventh in between) and
compares each answer with the cut tables themselves, read by ``tests._real_tables.Ensembl``:

* a gene converted into a database gives the identifiers that the database has for the gene at that release, as
  the database wrote them at that release, and a gene without one is named itself, flagged;
* an identifier read as an id of a named database stands for the genes that this database attaches it to; the
  best answer is the gene that most databases attach the identifier to, and genes equal in that are a tie;
* an identifier read without a database stands for the genes of every database, by the same rule.

The second part holds the rules that need a particular case (the hyper-connectivity threshold, the strict engine
and the flagged fall-back of the API, the suffix of an accession, a dataset whose database is named, and, for
human, the copies of a gene outside the reference sequence); each case is taken from the tables or from the truth
file (``tests/data/real/<organism>.truth.jsonl``), never from what idtrack answered.
"""

from __future__ import annotations

import collections
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from idtrack._db import DB
from idtrack._the_graph import TheGraph
from idtrack._track import InactiveAtReleaseError, UnknownVersionError
from tests._real_tables import ORGANISMS, SNAPSHOT, RealTables, read_truth, tables_path, truth_path, verdict

SYNONYM = DB.synonym_id_nodes_prefix
AVAILABLE = [organism for organism in ORGANISMS if tables_path(organism).is_file()]


@lru_cache(maxsize=None)
def spread_of_releases(organism: str) -> tuple[int, ...]:
    """Return the releases at which the same-release conversions are asked."""
    indices = RealTables.load(organism).indices
    releases_of = {int(a): set(payload["releases"]) for a, payload in indices.items()}
    serving = {}
    for release in sorted(set().union(*releases_of.values())):
        if release <= SNAPSHOT:
            serving[release] = next(a for a in DB.assemblies_by_priority(organism) if release in releases_of.get(a, ()))
    releases = sorted(serving)
    edges = {
        r
        for r in releases
        if serving.get(r + 1, serving[r]) != serving[r] or serving.get(r - 1, serving[r]) != serving[r]
    }
    return tuple(sorted({releases[0], releases[-1], *releases[::7]} | edges))


SPREAD = [
    pytest.param(organism, release, id=f"{organism}:{release}")
    for organism in AVAILABLE
    for release in spread_of_releases(organism)
]


class Release:
    """What Ensembl's tables say at one release of the graph, on the assembly that serves it."""

    def __init__(self, world, release: int):
        """Read the genes and the selected cross references of the release."""
        self.release = release
        self.assembly = world.backbone[release]
        self.genes = world.genes_at(release)
        self.outside = world.ensembl.non_reference_genes(self.assembly, release)
        self.ids_of_gene: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        self.genes_of_id: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        self.databases_of: dict[str, dict[str, set[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        self.attached: dict[str, set[tuple[str, str, str]]] = collections.defaultdict(set)
        for gene, name, value, kind, form in world.ensembl.links(self.assembly, release):
            if kind == "id" and world.ensembl.selected(form, name, self.assembly, release):
                self.ids_of_gene[(gene, name)].add(value)
                self.genes_of_id[(value, name)].add(gene)
                self.databases_of[value.lower()][gene].add(name)
                self.attached[value.lower()].add((gene, form, name))
        self.databases = sorted({name for _gene, name in self.ids_of_gene})

    def nearest_form_has_every_database(self, value: str) -> bool:
        """Tell whether, for each gene, the databases that attach an identifier to it do so on one form.

        An identifier can be attached to a gene itself and, by another database, to one of its transcripts or
        translations (old releases list a protein accession on the gene and on its translation). ``False`` when
        some gene has a database only on a form further from the gene than another database's.
        """
        order = {"gene": 0, "transcript": 1, "translation": 2}
        for gene in {owner for owner, _form, _database in self.attached[value.lower()]}:
            mine = [(order[form], database) for owner, form, database in self.attached[value.lower()] if owner == gene]
            nearest = min(depth for depth, _database in mine)
            if {database for depth, database in mine if depth == nearest} != {database for _depth, database in mine}:
                return False
        return True

    def node(self, gene: str) -> str:
        """Return a gene id with its version of the release."""
        return f"{gene}.{self.genes[gene]}"

    def answer(self, owners: set[str], value: str) -> tuple[set[str], set[str], set[str]]:
        """Return the genes an identifier stands for, the best of them, and the copies reported apart.

        Copies outside the reference sequence leave an answer that holds a gene of the reference sequence; among
        the genes left, the best are those that most databases attach the identifier to.
        """
        reference = owners - self.outside
        kept = reference or owners
        support = {gene: len(self.databases_of[value.lower()][gene]) for gene in kept}
        best = {self.node(gene) for gene, count in support.items() if count == max(support.values())}
        apart = {self.node(gene) for gene in owners - reference} if reference else set()
        return {self.node(gene) for gene in kept}, best, apart


def at(world, release: int) -> Release:
    """Return the view of a release, kept with the world of the session."""
    views = world.__dict__.setdefault("release_views", {})
    if release not in views:
        views[release] = Release(world, release)
    return views[release]


def sample(items: list, count: int) -> list:
    """Return ``count`` items spread evenly over a sorted list."""
    return items[:: max(1, len(items) // count)][:count]


# --------------------------------------------------------------------------- one release, read from the tables


@pytest.mark.parametrize("organism, release", SPREAD)
def test_gene_gets_the_ids_a_database_has_for_it_at_the_target_release(real_world, organism, release):
    """The final conversion returns Ensembl's cross references of that release, spelt as written there.

    Also at releases of an earlier assembly, whose databases are those of that assembly. A database that writes
    one identifier of a gene in two letter cases at one release is answered with one of the two spellings.
    """
    world = real_world(organism)
    view = at(world, release)
    api = world.api
    own = ("ensembl_gene", "base_ensembl_gene")
    finals = [d for d in api.valid_final_databases(release) if not d.startswith(SYNONYM) and d not in own]
    # every database of the assembly may be asked for; one that has nothing at this release answers for no gene
    assert set(view.databases) <= set(finals)
    elsewhere = sorted(set(finals) - set(view.databases))[:1]

    wrong = []
    asked = 0
    for database in view.databases + elsewhere:
        having = sorted(gene for gene, name in view.ids_of_gene if name == database)
        lacking = sorted(set(view.genes) - set(having))
        for gene in sample(having, 6) + lacking[:1]:
            expected = view.ids_of_gene.get((gene, database), set())
            node = view.node(gene)
            result = api.convert_identifier(
                node, from_release=release, to_release=release, final_database=database, strategy="all"
            )
            asked += 1
            got = result["target_id"]
            if expected:
                right = (
                    set(got) <= expected
                    and {i.lower() for i in got} == {i.lower() for i in expected}
                    and len({i.lower() for i in got}) == len(got)
                    and (result["final_database"], result["no_target"]) == (database, False)
                    and all(gene_node == node for gene_node, _target in result["last_node"])
                )
            else:
                right = (got, result["final_database"], result["no_target"]) == ([node], "ensembl_gene", True)
            if not right:
                wrong.append((database, node, sorted(expected), got, result["final_database"], result["no_target"]))
    assert not wrong, wrong[:10]
    assert asked >= 20


@pytest.mark.parametrize("organism, release", SPREAD)
def test_id_of_a_named_database_stands_for_the_genes_of_that_database(real_world, organism, release):
    """A named database says which genes an identifier means; every database's support decides between them.

    ``strategy="all"`` returns the genes that the named database attaches the identifier to at the release;
    ``strategy="best"`` the one that most databases attach it to, and several only when nothing tells them apart
    (a tie). Copies outside the reference sequence are reported apart.
    """
    world = real_world(organism)
    view = at(world, release)
    api = world.api
    wrong = []
    asked = ties = 0
    for database in [d for d in api.list_external_databases() if not d.startswith(SYNONYM)]:
        ids = sorted(value for value, name in view.genes_of_id if name == database)
        shared = [value for value in ids if len(view.genes_of_id[(value, database)]) > 1]
        for value in sample(ids, 5) + shared[:4]:
            everyone, best, apart = view.answer(view.genes_of_id[(value, database)], value)
            arguments = dict(from_release=release, to_release=release, from_database=database)
            all_of_them = api.convert_identifier(value, strategy="all", **arguments)
            the_best = api.convert_identifier(value, strategy="best", **arguments)
            asked += 1
            ties += len(best) > 1
            right = (
                set(all_of_them["target_id"]) == everyone
                and set(all_of_them["alternative_locus_genes"]) == apart
                and set(the_best["target_id"]) == best
                and the_best["tie"] is (len(best) > 1)
                and the_best["target_id"] == sorted(the_best["target_id"])
                and the_best["from_databases"] == [database]
                and (the_best["from_release"], the_best["from_release_fallback"]) == (release, False)
                and the_best["graph_id"].lower() == value.lower()
            )
            if not right:
                wrong.append(
                    (database, value, sorted(best), the_best["target_id"], the_best["tie"], all_of_them["target_id"])
                )
    assert not wrong, wrong[:10]
    assert asked >= 20


@pytest.mark.parametrize("organism, release", SPREAD)
def test_id_without_a_named_database_stands_for_the_genes_of_every_database(real_world, organism, release):
    """Without a database an identifier stands for every gene that some database attaches it to at the release.

    The best answer is the gene with the most databases behind it; a symbol that two databases give to two genes
    with equal support is a tie, flagged, in ascending order of id. Asked for the identifiers that the tables
    write exactly so and in one letter case only (how another spelling is matched is tested with the truth file).
    """
    world = real_world(organism)
    view = at(world, release)
    api = world.api
    wrong = []
    asked = 0
    for database in view.databases:
        ids = sorted(value for value, name in view.genes_of_id if name == database)
        shared = [value for value in ids if len(view.databases_of[value.lower()]) > 1]
        for value in sample(ids, 5) + shared[:4]:
            match, here, databases = world.ensembl.meaning(value, release)
            if match != "exact" or not here:
                continue
            everyone, best, apart = view.answer(set(view.databases_of[value.lower()]), value)
            all_of_them = api.convert_identifier(value, from_release=release, to_release=release, strategy="all")
            the_best = api.convert_identifier(value, from_release=release, to_release=release, strategy="best")
            asked += 1
            right = (
                set(all_of_them["target_id"]) == everyone
                and set(all_of_them["alternative_locus_genes"]) == apart
                and the_best["target_id"] == sorted(best)
                and the_best["tie"] is (len(best) > 1)
                and (the_best["match"], the_best["graph_id"]) == ("exact", value)
                and sorted(the_best["from_databases"]) == sorted(databases or ())
            )
            if not right:
                wrong.append((value, sorted(best), the_best["target_id"], the_best["tie"], all_of_them["target_id"]))
    assert not wrong, wrong[:10]
    assert asked >= 20


@pytest.mark.parametrize("organism, release", SPREAD)
def test_every_gene_id_of_a_release_is_itself_at_that_release(real_world, organism, release):
    """A gene id, with its version or without, converted to a release at which it exists, is that gene.

    Without ``from_release`` an id that exists at the target release is read there, also when the target is an
    old release and the id has later versions. Every gene of the release is asked, in both spellings.
    """
    world = real_world(organism)
    api = world.api
    genes = world.genes_at(release)
    assert len(genes) >= 20
    wrong = []
    for gene, version in genes.items():
        node = f"{gene}.{version}"
        for identifier in (node, gene):
            free = api.convert_identifier(identifier, to_release=release)
            given = api.convert_identifier(identifier, from_release=release, to_release=release, strategy="all")
            for result, inferred in ((free, True), (given, False)):
                right = (
                    result["target_id"] == [node]
                    and result["last_node"] == [[node, node]]
                    and (result["graph_id"], result["match"]) == (identifier, "exact")
                    and (result["from_release"], result["from_release_inferred"]) == (release, inferred)
                    and not (result["tie"] or result["ambiguous"] or result["from_release_fallback"])
                    and result["final_database"] == "ensembl_gene"
                )
                if not right:
                    wrong.append((identifier, inferred, result["target_id"], result["from_release"]))
    assert not wrong, wrong[:10]


@pytest.mark.parametrize("organism", AVAILABLE)
def test_gene_id_that_exists_at_the_target_release_is_its_own_answer(real_world, organism):
    """Whatever lies between two releases, an id that Ensembl has at both is answered with itself.

    Every gene of the tables is asked from its first release to its last, back, and from a release in between
    (also over the changes of assembly): the best answer is the id with its version of the target release, alone
    and not as a tie; with ``strategy="all"`` it is the first of the genes returned.
    """
    world = real_world(organism)
    api = world.api
    releases_of: dict[str, list[int]] = collections.defaultdict(list)
    for release in sorted(world.backbone):
        for gene in world.genes_at(release):
            releases_of[gene].append(release)
    wrong = []
    asked = versions_changed = 0
    for gene, releases in sorted(releases_of.items()):
        first, last, middle = releases[0], releases[-1], releases[len(releases) // 2]
        for source, target in {(first, last), (last, first), (middle, last), (first, middle)}:
            if source == target:
                continue
            node = f"{gene}.{world.genes_at(target)[gene]}"
            versions_changed += world.genes_at(source)[gene] != world.genes_at(target)[gene]
            best = api.convert_identifier(gene, from_release=source, to_release=target)
            everyone = api.convert_identifier(gene, from_release=source, to_release=target, strategy="all")
            asked += 1
            there = world.genes_at(target)
            if not (
                best["target_id"] == [node]
                and best["tie"] is False
                and everyone["target_id"][:1] == [node]
                and best["from_release"] == source
                and node not in best["alternative_locus_genes"]
                # whatever else the history reaches is a gene of the target release
                and all(
                    there.get(other.rpartition(".")[0]) == int(other.rpartition(".")[2])
                    for other in everyone["target_id"]
                )
            ):
                wrong.append((gene, source, target, best["target_id"], everyone["target_id"][:3]))
    assert not wrong, wrong[:10]
    assert asked >= 500 and versions_changed >= 100


def one_session_fates(world) -> list[tuple[int, str, dict[str, float], int]]:
    """Return the ids that one mapping session replaced by several ids, or merged from several, read from the tables.

    Each item: the release the id is read at, the id with its version, the ids it is to become at the
    neighbouring release with Ensembl's mapping score of each, and that release. Taken are the sessions between two consecutive
    releases in which the id is not mapped onto itself, exists on its side only, and every counterpart exists on
    the other side with the version of the event; forward (an id that was split) and backward (an id that was
    merged from several).
    """
    events = world.ensembl.events(world.backbone[SNAPSHOT], SNAPSHOT)
    events = events[events["new_release"] == events["old_release"] + 1]
    events = events[events["old_stable_id"].notna() & events["new_stable_id"].notna()]
    found = []
    for (old_release, new_release), session in events.groupby(["old_release", "new_release"]):
        old_release, new_release = int(old_release), int(new_release)
        if old_release not in world.backbone or new_release not in world.backbone:
            continue
        sides = (
            ("old", "new", old_release, new_release),  # forward: what an id of the old release became
            ("new", "old", new_release, old_release),  # backward: what an id of the new release was made from
        )
        for this, other, here, there in sides:
            genes_here, genes_there = world.genes_at(here), world.genes_at(there)
            for identifier, rows in session.groupby(f"{this}_stable_id"):
                version = int(rows[f"{this}_version"].iloc[0])
                partners = {
                    f"{row[f'{other}_stable_id']}.{int(row[f'{other}_version'])}": float(row["score"])
                    for _, row in rows.iterrows()
                }
                if (
                    len(partners) < 2
                    or identifier in set(rows[f"{other}_stable_id"])
                    or genes_here.get(identifier) != version
                    or identifier in genes_there
                    or rows[f"{this}_version"].nunique() != 1
                    or any(genes_there.get(p.rpartition(".")[0]) != int(p.rpartition(".")[2]) for p in partners)
                    or any(p.rpartition(".")[0] in genes_here for p in partners)
                ):
                    continue
                found.append((here, f"{identifier}.{version}", partners, there))
    return found


@pytest.mark.parametrize("organism", AVAILABLE)
def test_split_or_merged_id_goes_to_every_counterpart_and_best_to_the_highest_score(real_world, organism):
    """An id that one session split into several ids converts to all of them; the best is Ensembl's best mapping.

    Read from Ensembl's events: the counterparts of the id in the session and the mapping score of each. With
    ``strategy="all"`` the answer is every counterpart; with ``"best"`` the one with the highest score (a tie when
    two share it), and the explanation reports that score. Backward in time the same holds for an id that was
    made from several. Counterparts on patches and haplotypes are reported apart when a counterpart of the
    reference sequence exists (the genes of the human MHC haplotypes).
    """
    world = real_world(organism)
    api = world.api
    cases = one_session_fates(world)
    forward = [case for case in cases if case[3] > case[0]]
    backward = [case for case in cases if case[3] < case[0]]
    assert len(forward) >= 5 and len(backward) >= 5
    wrong = []
    decided_by_score = 0
    for here, identifier, counterparts, there in cases:
        everyone = api.convert_identifier(identifier, from_release=here, to_release=there, strategy="all", explain=True)
        best = api.convert_identifier(identifier, from_release=here, to_release=there, strategy="best")
        # counterparts outside the reference sequence leave an answer that holds a gene of the reference sequence
        outside = world.ensembl.non_reference_genes(world.backbone[there], there)
        copies = {p for p in counterparts if p.rpartition(".")[0] in outside}
        apart = copies if len(copies) < len(counterparts) else set()
        partners = {p: score for p, score in counterparts.items() if p not in apart}
        top = max(partners.values())
        highest = sorted(p for p, score in partners.items() if score == top)
        decided_by_score += len(highest) == 1 and len(set(partners.values())) > 1
        scores = {path["ensembl_gene"]: path["score"] for path in everyone["the_path"]}
        if not (
            set(everyone["target_id"]) == set(partners)
            and set(everyone["alternative_locus_genes"]) == apart
            and all(abs(scores[p] - partners[p]) < 1e-9 for p in partners)
            and everyone["target_id"][0] in highest
            and set(best["target_id"]) <= set(highest)
            and (len(highest) > 1 or (best["target_id"] == highest and best["tie"] is False))
            and all(path["external_jump"] == 0 for path in everyone["the_path"])
        ):
            wrong.append((here, there, identifier, counterparts, best["target_id"], everyone["target_id"], scores))
    assert not wrong, wrong[:5]
    assert decided_by_score >= 5


@pytest.mark.parametrize("organism, release", SPREAD)
def test_transcripts_and_translations_stand_for_their_gene_of_the_release(real_world, organism, release):
    """A transcript or translation id of a release, with its version or without, converts to its gene there."""
    world = real_world(organism)
    api = world.api
    genes = world.genes_at(release)
    products = world.ensembl.products(world.backbone[release], release)
    wrong = []
    asked = 0
    for form in ("transcript", "translation"):
        for identifier in sample(sorted(products[form]), 25):
            version, gene = products[form][identifier]
            node = f"{gene}.{genes[gene]}"
            for typed in (f"{identifier}.{version}", identifier):
                result = api.convert_identifier(typed, from_release=release, to_release=release, strategy="all")
                asked += 1
                if not (
                    result["target_id"] == [node]
                    and result["graph_id"] == f"{identifier}.{version}"
                    and result["from_release"] == release
                    and not result["no_corresponding"]
                ):
                    wrong.append((typed, node, result["target_id"], result["graph_id"]))
    assert not wrong, wrong[:10]
    assert asked >= 40


def ids_attached_on_several_forms(world) -> list[tuple[int, str, list[str]]]:
    """Return ``(release, identifier, databases)`` of the identifiers that reach a gene through several forms.

    Read from the tables at the releases of the spread: an identifier that one database attaches to a gene and
    another to a transcript or translation of the same gene, written in one way and matched exactly.
    """
    found = []
    for release in spread_of_releases(world.organism):
        view = at(world, release)
        for value in sorted({written for written, _database in view.genes_of_id}):
            match, here, databases = world.ensembl.meaning(value, release)
            if not view.nearest_form_has_every_database(value) and match == "exact" and here:
                found.append((release, value, sorted(databases or ())))
    return found


@pytest.mark.parametrize("organism", ["sus_scrofa"] if "sus_scrofa" in AVAILABLE else [])
def test_tables_hold_ids_that_reach_a_gene_through_several_forms(real_world, organism):
    """The case of the next test exists in the tables (pig, the releases of the first assembly)."""
    assert len(ids_attached_on_several_forms(real_world(organism))) >= 2


@pytest.mark.parametrize("organism", ["sus_scrofa"] if "sus_scrofa" in AVAILABLE else [])
def test_databases_of_an_id_attached_on_several_forms_are_all_reported(real_world, organism):
    """``from_databases`` names every database through which the identifier belongs to its gene at the release.

    An identifier that one database attaches to a gene and another to its translation (pig release 63: ``O19071``
    is a UniProtKB Gene Name id of ``ENSSSCG00000005006`` and the Uniprot/SWISSPROT accession of its translation)
    is read through both, and both count as the support of the gene.
    """
    world = real_world(organism)
    wrong = []
    for release, value, databases in ids_attached_on_several_forms(world):
        result = world.api.convert_identifier(value, from_release=release, to_release=release)
        engine = world.api.track.convert(value, from_release=release, to_release=release)
        view = at(world, release)
        support = {
            view.node(gene): len({database for owner, _form, database in view.attached[value.lower()] if owner == gene})
            for gene in {owner for owner, _form, _database in view.attached[value.lower()]}
        }
        if sorted(result["from_databases"]) != databases or any(
            entry["entry_support"] != support[entry["entry_gene"]] for entry in engine.values()
        ):
            wrong.append((release, value, result["from_databases"], databases, support))
    assert not wrong, wrong[:10]


# --------------------------------------------------------------------------- a named database (decisions 33, 36)


def symbols_of(world) -> tuple[str, dict[str, tuple[set[str], set[str]]]]:
    """Return the symbol database of the snapshot and, per symbol, its best genes named and not named.

    Only symbols that the tables write in one spelling and that are matched exactly are taken.
    """
    view = at(world, SNAPSHOT)
    database = next(d for d in DB.symbol_databases[world.organism] if d in view.databases)
    found = {}
    for value, name in sorted(view.genes_of_id):
        if name != database or world.ensembl.meaning(value, SNAPSHOT)[0] != "exact":
            continue
        named = view.answer(view.genes_of_id[(value, database)], value)[1]
        unnamed = view.answer(set(view.databases_of[value.lower()]), value)[1]
        found[value] = (named, unnamed)
    return database, found


@pytest.mark.parametrize("organism", AVAILABLE)
def test_naming_the_database_removes_the_ties_it_should_and_makes_none(real_world, organism):
    """Every symbol of the snapshot, read with its database named and without, against the tables.

    The tables must hold the three cases the rule is about: a symbol that another database gives to another gene
    with equal support (a tie without the database, one gene with it), a symbol whose gene in the symbol database
    is not the gene most databases give it to (two different single answers), and a symbol that the symbol
    database itself gives to several genes.
    """
    world = real_world(organism)
    database, symbols = symbols_of(world)
    cases = collections.Counter()
    for symbol, (named, unnamed) in symbols.items():
        with_database = world.api.convert_identifier(symbol, to_release=SNAPSHOT, from_database=database)
        without = world.api.convert_identifier(symbol, to_release=SNAPSHOT)
        assert (with_database["target_id"], with_database["tie"]) == (sorted(named), len(named) > 1), symbol
        assert (without["target_id"], without["tie"]) == (sorted(unnamed), len(unnamed) > 1), symbol
        assert with_database["from_databases"] == [database]
        cases["tie without the database, one gene with it"] += len(unnamed) > 1 and len(named) == 1
        cases["another gene with the database than without"] += (
            len(unnamed) == 1 and len(named) == 1 and named != unnamed
        )
        cases["several genes in the symbol database"] += len(view_owners(world, symbol, database)) > 1
    assert len(symbols) > 100
    assert cases["tie without the database, one gene with it"] >= 2, dict(cases)
    assert cases["another gene with the database than without"] >= 2, dict(cases)
    # the symbol database of pig gives no symbol to two genes at this release; HGNC does
    assert (cases["several genes in the symbol database"] >= 2) is (organism == "homo_sapiens"), dict(cases)


def view_owners(world, symbol: str, database: str) -> set[str]:
    """Return the genes that a database attaches a symbol to at the snapshot release."""
    return at(world, SNAPSHOT).genes_of_id[(symbol, database)]


@pytest.mark.parametrize("organism", AVAILABLE)
def test_harmonizer_reads_a_dataset_in_the_database_it_is_told(real_world, organism, tmp_path):
    """``HarmonizeFeatures(source_databases=...)``: the symbols of a dataset are read as ids of that database.

    A dataset of symbols of the snapshot release, among them every symbol on which naming the database matters,
    converted to the same symbol database. Each feature's gene is compared with the tables: with the database
    named, the gene that the database gives the symbol to (the best supported, should it give it to several);
    without, the best supported of all genes that carry the string and have a symbol in the target database, and
    ``ambiguous`` when two are equal.
    """
    import anndata as ad

    from idtrack import HarmonizeFeatures

    world = real_world(organism)
    view = at(world, SNAPSHOT)
    database, symbols = symbols_of(world)
    differing = [symbol for symbol, (named, unnamed) in symbols.items() if named != unnamed]
    names = sorted(set(differing) | set(sample(sorted(symbols), 40)))
    assert len(differing) >= 4
    path = tmp_path / "study.h5ad"
    ad.AnnData(X=np.ones((2, len(names)), dtype=np.float32), var=pd.DataFrame(index=names)).write_h5ad(path)

    def harmonized(folder, **more):
        harmonizer = HarmonizeFeatures(
            "real", {"study": str(path)}, str(tmp_path / folder), world.repository, SNAPSHOT,
            final_database=database, api=world.api, verbose_level=0, **more,
        )  # fmt: skip
        table = harmonizer.feature_table("study")
        return {name: (row["status"], row["ensembl_gene"]) for name, row in table.iterrows()}

    def expected(owners, symbol):
        # one result has one final database: the genes that have a symbol in it are the candidates
        with_symbol = {gene for gene in owners if view.ids_of_gene.get((gene, database))}
        best = view.answer(with_symbol, symbol)[1]
        return ("converted", next(iter(best))) if len(best) == 1 else ("ambiguous", None)

    without = harmonized("unnamed")
    named = harmonized("named", source_databases={"study": database})
    statuses = collections.Counter()
    for symbol in names:
        pairs = (
            (named[symbol], expected(view.genes_of_id[(symbol, database)], symbol)),
            (without[symbol], expected(set(view.databases_of[symbol.lower()]), symbol)),
        )
        for (status, gene), (status_expected, gene_expected) in pairs:
            assert status == status_expected, (symbol, status, status_expected)
            assert gene == gene_expected or (gene_expected is None and pd.isna(gene)), (symbol, gene, gene_expected)
        statuses[(pairs[0][0][0], pairs[1][0][0])] += 1
        if pairs[0][1] != pairs[1][1]:
            statuses["changed by naming the database"] += 1
    assert statuses["changed by naming the database"] >= 2, dict(statuses)


# --------------------------------------------------------------------------- cases of the truth file


def lines(shows: str | None = None, kind: str | None = None):
    """Return the parameters of the truth lines that show a rule (``shows``) or are of a kind."""
    return [
        pytest.param(organism, query, id=f"{organism}:{query['source']}")
        for organism in AVAILABLE
        if truth_path(organism).is_file()
        for query in read_truth(organism)
        if (shows is None or query.get("shows") == shows) and (kind is None or query["kind"] == kind)
    ]


def ask(api, query, **more):
    """Ask a query of the truth file."""
    return api.convert_identifier(
        query["id"], from_release=query["from_release"], to_release=query["to_release"],
        final_database=query["final_database"], strategy=query["strategy"], **more,
    )  # fmt: skip


@pytest.mark.parametrize("organism, query", lines(shows="threshold"))
def test_id_shared_by_more_than_five_genes_at_once_is_no_bridge(real_world, organism, query, monkeypatch):
    """A retired id that Ensembl gives no successor is not bridged through a family name to another gene.

    With the threshold of the graph (more than 5 genes at once) the answer is Ensembl's: nothing. The same graph
    read with a threshold of 20 follows the family name and returns genes that Ensembl's history does not give,
    which is what the threshold is for.
    """
    world = real_world(organism)
    assert TheGraph.genes_at_once_threshold == 5
    assert verdict(query, ask(world.api, query)) is None

    monkeypatch.setattr(TheGraph, "genes_at_once_threshold", 20)
    import idtrack

    lenient = idtrack.API(local_repository=world.repository)
    lenient.build_graph(organism, SNAPSHOT)
    assert verdict(query, ask(lenient, query)) is not None


@pytest.mark.parametrize("organism, query", lines(shows="clone"))
def test_clone_based_name_is_no_bridge_between_two_genes(real_world, organism, query, monkeypatch):
    """A retired id is not carried to another gene by a clone-based name that both once had.

    Clone-based names (``AC012345.1``) name a stretch of sequence, not a gene. With them excluded from bridging
    the answer is Ensembl's; the same graph asked with these names allowed as bridges returns genes that
    Ensembl's history does not give.
    """
    from idtrack._track import Track

    world = real_world(organism)
    assert any(name.startswith("Clone_based") for name in Track.non_bridging_databases)
    assert verdict(query, ask(world.api, query)) is None

    monkeypatch.setattr(Track, "non_bridging_databases", frozenset())
    assert verdict(query, ask(world.api, query)) is not None


@pytest.mark.parametrize("organism, query", lines(shows="fallback"))
def test_engine_is_strict_about_the_release_and_the_api_falls_back_with_a_flag(real_world, organism, query):
    """An identifier that the graph does not hold at ``from_release``: the engine raises, the API says what it did."""
    world = real_world(organism)
    track = world.api.track

    with pytest.raises(InactiveAtReleaseError) as error:
        track.convert(query["id"], from_release=query["from_release"], to_release=query["to_release"])
    assert str(query["from_release"]) in str(error.value)

    result = ask(world.api, query)
    read_at = result["from_release"]
    assert verdict(query, result) is None
    assert (result["from_release_fallback"], result["from_release_given"]) == (True, query["from_release"])
    assert read_at != query["from_release"] and result["from_release_inferred"] is True
    # the fall-back is the reading without a release: the same node, the same release, the same answer
    free = world.api.convert_identifier(query["id"], to_release=query["to_release"], strategy=query["strategy"])
    assert (free["from_release"], free["graph_id"], free["target_id"]) == (
        read_at,
        result["graph_id"],
        result["target_id"],
    )
    assert free["from_release_fallback"] is False


@pytest.mark.parametrize("organism, query", lines(kind="version_unknown"))
def test_engine_raises_for_a_version_that_never_existed(real_world, organism, query):
    """The engine names the versions that exist; the API reports the same facts as data (tested with the truth)."""
    world = real_world(organism)
    with pytest.raises(UnknownVersionError) as error:
        world.api.track.convert(query["id"], to_release=query["to_release"])
    assert list(error.value.versions) == world.api.convert_identifier(query["id"])["known_versions"]
    assert error.value.versions


@pytest.mark.parametrize("organism", AVAILABLE)
def test_version_suffix_is_cut_off_an_accession_and_never_off_a_symbol(real_world, organism):
    """``NM_000546.9`` is the accession ``NM_000546``; a symbol followed by ``.<digits>`` is no identifier."""
    world = real_world(organism)
    view = at(world, SNAPSHOT)
    written = world.ensembl.written(None)
    accessions = [d for d in view.databases if d.startswith(DB.accession_database_prefixes)]
    symbol_databases = [d for d in DB.symbol_databases[organism] if d in view.databases]
    assert accessions and symbol_databases

    asked = 0
    for database in accessions:
        ids = sorted(value for value, name in view.genes_of_id if name == database)
        for value in sample(ids, 8):
            if "." in value or world.ensembl.meaning(value, SNAPSHOT)[0] != "exact":
                continue
            if any(name in symbol_databases for names in view.databases_of[value.lower()].values() for name in names):
                continue
            everyone, _best, _apart = view.answer(set(view.databases_of[value.lower()]), value)
            result = world.api.convert_identifier(f"{value}.7", to_release=SNAPSHOT, strategy="all")
            assert (result["match"], result["graph_id"]) == ("version", value), value
            assert set(result["target_id"]) == everyone
            asked += 1
    assert asked >= 8

    asked = 0
    for database in symbol_databases:
        for value in sample(sorted(v for v, name in view.genes_of_id if name == database), 60):
            typed = f"{value}.7"
            if any((kind, typed.lower().replace("_", "-").replace(".", "-")) in written for kind in ("id", "synonym")):
                continue  # really an identifier in another spelling
            result = world.api.convert_identifier(typed, to_release=SNAPSHOT)
            assert (result["no_corresponding"], result["target_id"], result["match"]) == (True, [], None), typed
            asked += 1
    assert asked >= 20


@pytest.mark.parametrize("organism", AVAILABLE)
def test_spelling_returned_is_that_of_the_target_release(real_world, organism):
    """A database that changed the letter case of an identifier: it is returned as written at the target release.

    The cases are read from the tables: a gene whose identifier in a database is written in one way at a release
    and in another letter case at the last release at which the database has it for the gene.
    """
    world = real_world(organism)
    history: dict[tuple[str, str, str], dict[int, set[str]]] = collections.defaultdict(dict)
    for release in sorted(world.backbone):
        view = at(world, release)
        for (gene, database), values in view.ids_of_gene.items():
            for value in values:
                history[(gene, database, value.lower())].setdefault(release, set()).add(value)
    asked = 0
    for (gene, database, _lower), spellings in sorted(history.items()):
        latest = spellings[max(spellings)]
        earlier = [r for r, written in sorted(spellings.items()) if len(written) == 1 and not written & latest]
        if len(latest) != 1 or not earlier or asked >= 25:
            continue
        release = earlier[-1]
        node = f"{gene}.{world.genes_at(release)[gene]}"
        result = world.api.convert_identifier(
            node, from_release=release, to_release=release, final_database=database, strategy="all"
        )
        (then,), (now,) = spellings[release], latest
        assert then in result["target_id"] and now not in result["target_id"], (gene, database, release, then, now)
        asked += 1
    assert asked >= 3


# --------------------------------------------------------------------------- a name before a synonym


def names_that_are_also_synonyms(world) -> list[tuple[int, str, set[str], bool]]:
    """Return the strings that are a name of one gene and a synonym of another at one release, from the tables.

    Each item: the release, the string exactly as a database writes it there as an identifier, the genes that
    carry it as an identifier there, and whether this is the one way in which the identifier is written at the
    last release at which a database has it (``False`` also when two databases write it in two ways there).
    Taken are the strings whose genes as a synonym are not the genes of the name, so that the reading matters.
    """
    found = world.__dict__.get("names_and_synonyms")
    if found is not None:
        return found
    views = {release: at(world, release) for release in sorted(world.backbone)}
    last_spellings: dict[str, set[str]] = {}
    for release in sorted(views):
        now: dict[str, set[str]] = collections.defaultdict(set)
        for value, _database in views[release].genes_of_id:
            now[value.lower()].add(value)
        last_spellings.update(now)
    found = []
    for release, view in views.items():
        synonym_genes: dict[str, set[str]] = collections.defaultdict(set)
        for gene, name, value, kind, form in world.ensembl.links(view.assembly, release):
            if kind == "synonym" and world.ensembl.selected(form, SYNONYM + name, view.assembly, release):
                synonym_genes[value].add(gene)
        for value in sorted({written for written, _database in view.genes_of_id}):
            owners = set(view.databases_of[value.lower()])
            if value in synonym_genes and synonym_genes[value] != owners:
                found.append((release, value, owners, last_spellings[value.lower()] == {value}))
    world.__dict__["names_and_synonyms"] = found
    return found


def read_as_names(world, cases) -> list[tuple]:
    """Ask each string at its release; return the cases that were not answered with the genes of the name."""
    wrong = []
    for release, value, owners, _same in cases:
        expected = at(world, release).answer(owners, value)[0]
        result = world.api.convert_identifier(value, from_release=release, to_release=release, strategy="all")
        if set(result["target_id"]) != expected or str(result["match"]).startswith("synonym"):
            wrong.append((release, value, sorted(expected), result["target_id"], result["match"]))
    return wrong


@pytest.mark.parametrize("organism", AVAILABLE)
def test_name_of_one_gene_wins_over_the_same_string_as_a_synonym_of_another(real_world, organism):
    """A string that is a name of one gene and a synonym of another at the release asked about is the name.

    Each step of the matching is tried on names and then on synonyms. The cases are read from the tables: a
    database writes the string as an identifier of a gene at the release, another gene has exactly that string as
    a synonym at the same release, and the identifier is still written so at its last release.
    """
    world = real_world(organism)
    cases = [case for case in names_that_are_also_synonyms(world) if case[3]]
    assert len({value for _release, value, _owners, _same in cases}) >= 5
    assert not read_as_names(world, cases)
    # the tables also hold the case of the next test: such a name written in another letter case today
    assert any(not case[3] for case in names_that_are_also_synonyms(world))


@pytest.mark.parametrize("organism", AVAILABLE)
def test_name_written_otherwise_today_still_wins_over_a_synonym_at_its_release(real_world, organism):
    """The same rule for a name that is written otherwise, or in two ways, at its last release.

    At the release asked about a database writes the string exactly as typed for one gene, and another gene has
    it as a synonym; at its last release the identifier is written in another letter case, or two databases
    write it in two letter cases (the graph then names it by one of them). The step "exact" is judged against
    how the databases wrote the name at the release asked about, so the string is the name there (pig: ``PRR2``
    at release 82 is NCBI's name of one gene and a synonym of another, and the node is ``prr2`` after UniProt's
    spelling of today).
    """
    world = real_world(organism)
    cases = [case for case in names_that_are_also_synonyms(world) if not case[3]]
    assert cases
    assert not read_as_names(world, cases)


# --------------------------------------------------------------------------- copies outside the reference sequence

HUMAN = [organism for organism in AVAILABLE if organism == "homo_sapiens"]


@pytest.mark.parametrize("organism", HUMAN)
def test_copies_outside_the_reference_sequence_are_reported_apart(real_world, organism):
    """A symbol of a gene of the reference sequence and of its copies on patches, haplotypes and LRG records.

    Which genes lie outside the reference sequence is Ensembl's ``non_ref`` mark of their sequence region, read
    from the tables of the target release (the last release, and the last release of GRCh37). Every symbol of the
    symbol database that a copy carries is asked: the genes of the reference sequence are the answer, in both
    strategies, and the copies are listed in ``alternative_locus_genes``; a symbol that only copies carry is
    answered with them, and nothing is listed apart.
    """
    world = real_world(organism)
    api = world.api
    database = "HGNC Symbol"
    last_of_earlier = max(r for r, assembly in world.backbone.items() if assembly != world.backbone[SNAPSHOT])
    cases = collections.Counter()
    for release in (SNAPSHOT, last_of_earlier):
        view = at(world, release)
        assert view.outside, release
        for (value, name), owners in sorted(view.genes_of_id.items()):
            copies = owners & view.outside
            if name != database or not copies or world.ensembl.meaning(value, release)[0] != "exact":
                continue
            everyone, best, apart = view.answer(owners, value)
            for strategy, expected in (("all", everyone), ("best", best)):
                result = api.convert_identifier(
                    value, from_release=release, to_release=release, from_database=database, strategy=strategy
                )
                assert set(result["target_id"]) == expected, (release, value, strategy)
                assert result["alternative_locus_genes"] == sorted(apart), (release, value, strategy)
                assert not set(result["alternative_locus_genes"]) & set(result["target_id"])
            # the ids of a copy in the target database are no answer for the gene: the symbol comes back once
            named = api.convert_identifier(
                value, from_release=release, to_release=release, from_database=database, final_database=database
            )
            assert named["target_id"] == [value] and named["alternative_locus_genes"] == sorted(apart)
            if owners - copies:
                cases[(release, "reference gene and a copy on a patch or haplotype")] += any(
                    not gene.startswith("LRG_") for gene in copies
                )
                cases[(release, "reference gene and an LRG record")] += any(gene.startswith("LRG_") for gene in copies)
            else:
                cases[(release, "copies only")] += 1
    assert cases[(SNAPSHOT, "reference gene and a copy on a patch or haplotype")] >= 3, dict(cases)
    assert cases[(SNAPSHOT, "reference gene and an LRG record")] >= 3, dict(cases)
    assert cases[(SNAPSHOT, "copies only")] >= 1, dict(cases)
    assert cases[(last_of_earlier, "reference gene and a copy on a patch or haplotype")] >= 2, dict(cases)


@pytest.mark.parametrize("organism", HUMAN)
def test_copy_asked_by_its_own_id_is_the_answer(real_world, organism):
    """A gene on a patch, a haplotype or an LRG record, asked by its id, is itself, and nothing is listed apart."""
    world = real_world(organism)
    view = at(world, SNAPSHOT)
    asked = collections.Counter()
    for gene in sorted(view.outside):
        node = view.node(gene)
        for identifier in (node, gene, gene.lower()):
            result = world.api.convert_identifier(identifier, to_release=SNAPSHOT, strategy="all")
            assert (result["target_id"], result["alternative_locus_genes"]) == ([node], []), identifier
        asked["LRG record" if gene.startswith("LRG_") else "gene on a patch or haplotype"] += 1
    assert asked["LRG record"] >= 3 and asked["gene on a patch or haplotype"] >= 5


@pytest.mark.parametrize("organism", HUMAN)
def test_id_that_lay_outside_the_reference_sequence_at_the_target_release_is_still_its_own_answer(real_world, organism):
    """An id that Ensembl has on a patch of the earlier assembly and on a chromosome today, converted back.

    The mark of a copy counts at the target release, and the queried id's own version there is never taken out as
    a copy: it is the best answer, the first of all answers, and not listed apart, whatever else its history
    reaches at that release.
    """
    world = real_world(organism)
    api = world.api
    today = at(world, SNAPSHOT)
    asked = 0
    for release in sorted(world.backbone):
        if world.backbone[release] == world.backbone[SNAPSHOT]:
            continue
        genes = world.genes_at(release)
        beside = world.ensembl.non_reference_genes(world.backbone[release], release)
        for gene in sorted(beside & set(today.genes) - today.outside):
            if gene.startswith("LRG_"):
                continue
            node = f"{gene}.{genes[gene]}"
            best = api.convert_identifier(gene, from_release=SNAPSHOT, to_release=release)
            everyone = api.convert_identifier(gene, from_release=SNAPSHOT, to_release=release, strategy="all")
            assert (best["target_id"], best["tie"]) == ([node], False), (gene, release)
            assert everyone["target_id"][0] == node and node not in everyone["alternative_locus_genes"]
            asked += 1
    assert asked >= 5


@pytest.mark.parametrize("organism", HUMAN)
def test_frozen_parallel_assembly_supplies_nothing_at_a_release_the_primary_assembly_serves(real_world, organism):
    """GRCh37 is still listed at the releases of GRCh38, frozen; what it says there is no part of an answer.

    Read from the tables of both assemblies at releases of GRCh38: a gene that has the same id and version on both,
    and an identifier that a database gives it on GRCh37 only. Converted into that database at that release, the
    gene gets what GRCh38 says and never the identifier of the frozen assembly.
    """
    world = real_world(organism)
    api = world.api
    primary = world.backbone[SNAPSHOT]
    others = sorted({int(a) for a in world.tables.indices} - {primary})
    asked = 0
    for release in [r for r in spread_of_releases(organism) if world.backbone[r] == primary]:
        view = at(world, release)
        for other in others:
            if release not in world.ensembl.releases(other):
                continue
            there = world.ensembl.genes(other, release)
            taken = 0
            for gene, name, value, kind, form in world.ensembl.links(other, release):
                if kind != "id" or not world.ensembl.selected(form, name, other, release) or name not in view.databases:
                    continue
                if gene not in view.genes or view.genes[gene] != there.get(gene):
                    continue  # another version on the frozen assembly: another node of the graph
                here = {written.lower() for written in view.ids_of_gene.get((gene, name), ())}
                if value.lower() in here or taken >= 25:
                    continue
                result = api.convert_identifier(
                    view.node(gene), from_release=release, to_release=release, final_database=name, strategy="all"
                )
                found = {target.lower() for target in result["target_id"]} if not result["no_target"] else set()
                assert found == here, (release, gene, name, value)
                taken += 1
            asked += taken
    assert asked >= 20
