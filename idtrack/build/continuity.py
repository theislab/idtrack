#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

"""Which of Ensembl's databases a selected database is at each release: the continuity of its links.

Ensembl renames and refiles external databases over the releases: RefSeq's mRNA accessions are ``RefSeq_dna`` up to
release 63 and ``RefSeq_mRNA`` from 64 (and its ncRNA accessions, in the same ``RefSeq_dna`` up to 63, are
``RefSeq_ncRNA`` from 64); the projections of HGNC's symbols onto pig and macaque genes are ``HGNC_curated_gene`` and
``HGNC_automatic_gene`` up to release 60 and ``HGNC`` from 61; NCBI's gene ids are attached to translations up to 62
and to genes from 63. A selection of databases by name (the externals YAML) names one of these names; the identifiers
the same database holds under another name, or on another form, are the same identifiers.

The rule reads the database from Ensembl's own links, a link being an (identifier, gene) pair (the gene of the object
the identifier is attached to: a transcript's or a translation's gene):

1. A selected database is its name on the forms the selection holds it on (every selected form together).
2. Another database of Ensembl continues a selected one where links move between them from one release to the next:
   held by the one at a release and not at the next, by the other at the next and not at the first. The databases
   that may: one none of whose names is selected (a database whose labels are selected and whose accessions are not,
   or the reverse, is never another database), and a selected database's name on a form the selection does not hold
   it on (NCBI's gene ids on translations up to release 62, Rfam's accessions on transcripts up to 62), which can be
   that database only (human HGNC symbols on GRCh37's transcripts are not Vega's gene names, which hold the same
   symbols on genes from release 76). It continues
   the selected database when the links that move are at least :py:data:`CONTINUITY_SHARE` of the links of the
   database that leaves at that boundary (the older side's: a database hands its links over to its successor; at
   rhesus 70 to 72 Ensembl filed its own transcript names under EntrezGene on transcripts, and from 73 under
   ``EntrezGene_trans_name``: those 450 are 3 % of EntrezGene's links, so ``EntrezGene_trans_name`` does not continue
   it); where the selected database leaves, the moved links must also be that share of the candidate's links (a later
   name is, at its first release, the selected database's links: ``WikiGene`` takes over 215 of the 219 gene-form NCBI
   symbols of rhesus 54, which are 1 % of its own). A candidate that ends at the boundary (it holds no link at the next
   release) also continues the selected database where the links that move are that share of the selected
   database's links at the next release: a database that Ensembl stopped hands over what it still had,
   however much of it Ensembl dropped (human ``RefSeq_dna_predicted`` at 63 to 64 moves 31 % of its links, which are
   59 % of ``RefSeq_mRNA_predicted``'s and 67 % of ``RefSeq_ncRNA_predicted``'s links at 64). Its identifiers are
   then the selected database's at every release (a selected name on another form: in the runs of releases next to
   each other that moved; the Ensembl transcript names that Ensembl filed under Rfam and miRBase on rhesus transcripts
   at 70 and 71, a run of their own, do not), each by its form (the characters of the identifier before its first
   digit, then ``#`` where the rest is a number with an optional version, ``*`` for any other rest:
   :py:func:`identifier_form`): those of a form go to the selected database that most of their moved links
   went with at the boundaries that decide where that form is that database's identifier scheme (rule 3), others to
   the main continuation (``RefSeq_dna`` holds ``NM_`` and ``NR_`` accessions: they continue ``RefSeq_mRNA`` and
   ``RefSeq_ncRNA``), those of a form that never moved to the database's main continuation, unless the main
   continuation's identifiers all have another form (rule 3's scheme: Ensembl's transcript names that it filed under
   Rfam on rhesus transcripts at 70 and 71, ``RF00017-201``, are no Rfam accessions).
3. An identifier of a selected database whose form is the form of another selected database's identifiers (at least
   :py:data:`SCHEME_SHARE` of that database's identifiers have it: an identifier scheme such as Havana's
   ``OTTMUSG#``), of another Ensembl database, and whose link that database holds at some release, is that database's
   identifier: MGI gave the Havana ids of mouse genes without a symbol as their symbols at releases 48 to 53.

The result is recorded in the graph (:py:attr:`idtrack.graph.GraphRecord.database_names`: per selected database, the
names of Ensembl it holds, with their releases).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: The least share of a database's links of one identifier form that move to a selected database between two adjacent
#: releases for the database to be read as the selected one. The renames and refilings measured on the
#: five organisms move 0.82 to 1.00 of a database's links (per identifier form 0.94 to 1.00); the most that a database
#: Ensembl keeps beside a selected one moves into it at one boundary is 0.38.
CONTINUITY_SHARE = 0.5

#: Version of the rule of this module; a stored filing (:py:meth:`idtrack.DatabaseManager.database_filing`) of another
#: version is computed again.
FILING_RULES_VERSION = 5

#: The least share of a selected database's identifiers that have one form for that form to be its identifier scheme
#: (rule 3 of :py:mod:`idtrack.build.continuity`).
SCHEME_SHARE = 0.99

_FORM = re.compile(r"^([^0-9]*)(.*)$", re.DOTALL)
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def identifier_form(identifier: str) -> str:
    """Return the form of an identifier: its characters before the first digit, then ``#`` where the rest is a number.

    The number may carry a version (``.<digits>``); any other rest is ``*``: ``NM_000546`` and ``NM_000546.6`` are
    ``NM_#``, ``OTTMUSG00000003802`` is ``OTTMUSG#``, ``2633`` is ``#``, ``SMURF1`` is ``SMURF#``, ``5S_rRNA`` is ``*``.
    """
    head, rest = _FORM.match(str(identifier)).groups()  # type: ignore[union-attr]
    return head + ("#" if _NUMBER.fullmatch(rest) else "*" if rest else "")


def _hash(values: Iterable[str]) -> np.ndarray:
    return pd.util.hash_array(np.asarray(list(values), dtype=object)).astype(np.uint64)


def link_hashes(identifiers: Iterable[str], genes: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the hashes by which links are compared: the identifier as written, and the gene's stable id.

    A database under another name writes the same identifiers (``NM_000546`` under ``RefSeq_dna`` and ``RefSeq_mRNA``);
    a human symbol that Ensembl displayed on a mouse gene (``HGNC Symbol``, ``ABCA1``) is not MGI's ``Abca1``.
    """
    return _hash(str(i).strip() for i in identifiers), _hash(str(g) for g in genes)


def release_links(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the links of one release from ``name``, ``form`` (Ensembl's), ``identifier``, ``gene`` (stable ids).

    Identifiers are compared as written (:py:func:`link_hashes`).

    Returns:
        pandas.DataFrame: ``name``, ``eform`` (the Ensembl form), ``form`` (:py:func:`identifier_form`),
        ``identifier`` and ``gene`` (the hashes).
    """
    rows = rows.dropna(subset=["name", "identifier", "gene"])
    identifiers = rows["identifier"].astype(str).str.strip()
    keep = (identifiers != "").to_numpy()
    rows, identifiers = rows[keep], identifiers[keep]
    hashed, genes = link_hashes(identifiers.tolist(), rows["gene"].tolist())
    frame = pd.DataFrame(
        {
            "name": rows["name"].astype(str).to_numpy(dtype=object),
            "eform": rows["form"].astype(str).to_numpy(dtype=object),
            "form": np.asarray([identifier_form(i) for i in identifiers.tolist()], dtype=object),
            "identifier": hashed,
            "gene": genes,
        }
    )
    return frame.drop_duplicates(["name", "eform", "identifier", "gene"], ignore_index=True)


#: The prefix of a selected database's unit in :py:func:`moved_links` (its name follows).
SELECTED = "selected::"


def unit_of_form(name: str, form: str) -> str:
    """Return the unit of a selected database's name on a form the selection does not hold (``RFAM@transcript``)."""
    return f"{name}@{form}"


def moved_links(older: pd.DataFrame, newer: pd.DataFrame) -> pd.DataFrame:
    """Return the links that move between a candidate and a selected database between two adjacent releases.

    The frames hold a ``unit`` per row: ``selected::<name>`` for the selected databases (every selected form of the
    name together), any other text for a candidate. A link moves from ``A`` to ``B`` when ``A`` holds it at the older
    release and not at the newer, and ``B`` holds it at the newer and not at the older.

    Args:
        older: The links of the older release, with ``unit``, ``form``, ``identifier``, ``gene``.
        newer: The links of the release after it.

    Returns:
        pandas.DataFrame: ``side`` (``"older"``: the candidate leaves for the selected database; ``"newer"``: the
        selected database leaves for the candidate), ``candidate``, ``form`` (of the candidate's identifiers),
        ``selected`` (the name), ``moved`` (the number of links).
    """
    link = ["identifier", "gene"]
    a = older[["unit", "form", *link]].drop_duplicates(["unit", *link])
    b = newer[["unit", "form", *link]].drop_duplicates(["unit", *link])
    both = a.merge(b, on=link, suffixes=("_a", "_b"))
    both = both[both["unit_a"] != both["unit_b"]]
    a_selected = both["unit_a"].str.startswith(SELECTED)
    b_selected = both["unit_b"].str.startswith(SELECTED)
    out: list[tuple[str, str, str, str, int]] = []
    for side, part, candidate, selected, form in (
        ("older", both[~a_selected & b_selected], "unit_a", "unit_b", "form_a"),
        ("newer", both[a_selected & ~b_selected], "unit_b", "unit_a", "form_b"),
    ):
        if not len(part):
            continue
        # the unit that leaves does not hold the link on the other side, the one it goes to did not hold it before
        left, right = ("unit_a", "unit_b")
        gone = part.merge(
            b[["unit", *link]].rename(columns={"unit": left}), on=[*link, left], how="left", indicator="still"
        )
        gone = gone.merge(
            a[["unit", *link]].rename(columns={"unit": right}), on=[*link, right], how="left", indicator="already"
        )
        gone = gone[(gone["still"] == "left_only") & (gone["already"] == "left_only")]
        counts = gone.groupby([candidate, form, selected]).size()
        out.extend((side, x, f, y[len(SELECTED) :], int(n)) for (x, f, y), n in counts.items())
    return pd.DataFrame(out, columns=["side", "candidate", "form", "selected", "moved"])


@dataclasses.dataclass
class Filing:
    """Which of Ensembl's database names each selected database holds, and where.

    Attributes:
        runs (dict[str, tuple[tuple[int, int], ...]]): For a selected name on another form (``name@form``), the runs of
            releases (first, last) that continue the selected database; its other runs do not.
        renamed (dict[str, dict[str, str]]): Per database that continues a selected one (rule 2 of
            :py:mod:`idtrack.build.continuity`), its identifiers' forms and the selected database each is (``"*"``: every
            other form).
        releases (dict[str, tuple[int, ...]]): The releases at which each database name holds links.
        schemes (dict[str, str]): The selected databases whose identifiers have one form, with it (rule 3).
        databases (dict[str, frozenset[str]]): The Ensembl databases of each selected name (rule 3 does not move an
            identifier between the names of one Ensembl database: a label that repeats its accession).
        scheme_links (dict[str, dict[tuple[int, int], tuple[int, int]]]): For each of them, its links (hashes) with the
            first and last release at which it holds each.
        evidence (list[tuple]): Per boundary that decided rule 2 (for the measurement): the candidate, the selected
            database, the older release, the links moved, the links of the database that leaves, whether the
            candidate leaves, and the links of the database that receives them at the next release.
    """

    renamed: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    runs: dict[str, tuple[tuple[int, int], ...]] = dataclasses.field(default_factory=dict)
    releases: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    schemes: dict[str, str] = dataclasses.field(default_factory=dict)
    scheme_links: dict[str, dict[tuple[int, int], tuple[int, int]]] = dataclasses.field(default_factory=dict)
    evidence: list[tuple] = dataclasses.field(default_factory=list)

    def to_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the filing as two tables (to store it): what it decides, and the links of the identifier schemes."""
        rows = [
            ("renamed", unit, form, target, -1)
            for unit, forms in self.renamed.items()
            for form, target in forms.items()
        ]
        rows += [("release", name, "", "", int(r)) for name, found in self.releases.items() for r in found]
        rows += [("scheme", name, form, "", -1) for name, form in self.schemes.items()]
        rows += [("database", name, "", database, -1) for name, found in self.databases.items() for database in found]
        rows += [("evidence", str(e[0]), json.dumps(list(e[1:])), "", -1) for e in self.evidence]
        rows += [("run", unit, str(a), str(b), -1) for unit, found in self.runs.items() for a, b in found]
        decided = pd.DataFrame(rows, columns=["kind", "name", "form", "target", "release"]).astype(
            {"release": np.int64}
        )
        links = [
            (name, int(i), int(g), int(first), int(last))
            for name, held in self.scheme_links.items()
            for (i, g), (first, last) in held.items()
        ]
        table = pd.DataFrame(links, columns=["name", "identifier", "gene", "first", "last"])
        table = table.astype({"identifier": np.uint64, "gene": np.uint64, "first": np.int16, "last": np.int16})
        return decided, table

    @classmethod
    def from_tables(cls, decided: pd.DataFrame, links: pd.DataFrame) -> Filing:
        """Return the filing that :py:meth:`to_tables` stored."""
        filing = cls()
        releases: dict[str, list[int]] = {}
        databases: dict[str, set[str]] = {}
        for kind, name, form, target, release in decided.itertuples(index=False):
            if kind == "renamed":
                filing.renamed.setdefault(str(name), {})[str(form)] = str(target)
            elif kind == "release":
                releases.setdefault(str(name), []).append(int(release))
            elif kind == "scheme":
                filing.schemes[str(name)] = str(form)
            elif kind == "database":
                databases.setdefault(str(name), set()).add(str(target))
            elif kind == "evidence":
                filing.evidence.append((str(name), *json.loads(str(form))))
            elif kind == "run":
                filing.runs[str(name)] = (*filing.runs.get(str(name), ()), (int(form), int(target)))
        filing.releases = {name: tuple(sorted(found)) for name, found in releases.items()}
        filing.databases = {name: frozenset(found) for name, found in databases.items()}
        for name, group in links.groupby("name", sort=False):
            filing.scheme_links[str(name)] = {
                (int(i), int(g)): (int(a), int(b))
                for i, g, a, b in zip(
                    group["identifier"].tolist(),
                    group["gene"].tolist(),
                    group["first"].tolist(),
                    group["last"].tolist(),
                )
            }
        return filing

    def digest(self) -> str:
        """Return a digest of what the filing decides (part of the key of a stored table of selected rows).

        Computed once (a filing does not change once made: every table of selected rows of a build asks for it).
        """
        if self._digest is not None:
            return self._digest
        hashed = hashlib.sha256(json.dumps([self.renamed, self.schemes, self.runs], sort_keys=True).encode("utf-8"))
        for name in sorted(self.scheme_links):
            table = np.asarray(sorted((*key, *span) for key, span in self.scheme_links[name].items()), dtype=np.uint64)
            hashed.update(name.encode("utf-8"))
            hashed.update(table.tobytes())
        self._digest = hashed.hexdigest()[:16]
        return self._digest

    def continued(self, name: str, form: str, eform: str | None = None, release: int | None = None) -> str | None:
        """Return the selected database that a database's identifiers of one form are (rule 2), or ``None``.

        Args:
            name: The database's name.
            form: The form of the identifier (:py:func:`identifier_form`).
            eform: The Ensembl form the row is attached to (a selected name on a form the selection does not hold is
                judged on that form: :py:func:`unit_of_form`).
            release: The release of the row (such a name continues the selected database in the runs that moved).
        """
        unit = unit_of_form(name, eform) if eform is not None else ""
        forms = self.renamed.get(unit) if unit else None
        if forms is not None and release is not None:
            if not any(first <= int(release) <= last for first, last in self.runs.get(unit, ())):
                return None
        forms = forms or self.renamed.get(name)
        if not forms:
            return None
        if form in forms:
            return forms[form]
        main = forms["*"]
        # an identifier of a form that never moved is the main continuation's, unless that one's identifiers have
        # another form (its scheme): ``RF00017-201``, an Ensembl transcript name, is no Rfam accession
        scheme = self.schemes.get(main)
        return main if scheme is None or scheme == form else None

    databases: dict[str, frozenset[str]] = dataclasses.field(default_factory=dict)
    _digest: str | None = dataclasses.field(default=None, repr=False, compare=False)

    def scheme_of(self, name: str, form: str, identifier: int, gene: int, release: int) -> str | None:
        """Return the selected database whose identifier scheme a row has and whose link it is (rule 3), or ``None``.

        Args:
            name: The selected database the row is filed under (after rule 2).
            form: The form of its identifier.
            identifier: The identifier's hash (:py:func:`link_hashes`).
            gene: The gene's hash.
            release: The release of the row.

        Returns:
            str | None: Of the databases of that scheme that hold the link at some release, the one that holds it
            nearest the row's release (then by name); ``None`` when the row's own database has that scheme or none
            holds the link.
        """
        if self.schemes.get(name) == form:
            return None
        mine = self.databases.get(name, frozenset({name}))
        found = []
        for other, scheme in self.schemes.items():
            if other == name or scheme != form or mine & self.databases.get(other, frozenset({other})):
                continue
            span = self.scheme_links.get(other, {}).get((identifier, gene))
            if span is not None:
                first, last = span
                found.append((0 if first <= release <= last else min(abs(release - first), abs(release - last)), other))
        return min(found)[1] if found else None

    def database_names(self, selected: Iterable[str]) -> dict[str, dict[str, tuple[int, ...]]]:
        """Return, per selected database, the names of Ensembl it holds by rule 1 and 2, with their releases."""
        result: dict[str, dict[str, tuple[int, ...]]] = {}
        for name in sorted(set(selected)):
            if name in self.releases:
                result.setdefault(name, {})[name] = self.releases[name]
        for unit, forms in sorted(self.renamed.items()):
            name, _, eform = unit.partition("@")
            shown = f"{name} (on {eform}s)" if eform else name
            held = self.releases.get(unit, ())
            if unit in self.runs:  # the runs of releases that continue the selected database
                held = tuple(r for r in held if any(a <= r <= b for a, b in self.runs[unit]))
            for target in sorted(set(forms.values())):
                result.setdefault(target, {})[shown] = held
        return result


def database_filing(
    links_of: Callable[[int], pd.DataFrame],
    releases: Sequence[int],
    selected: Collection[tuple[str, str]],
    database_of: Mapping[str, Collection[str]],
    share: float = CONTINUITY_SHARE,
    scheme_share: float = SCHEME_SHARE,
) -> Filing:
    """Read from Ensembl's links which database names each selected database holds (rules 2 and 3 of the module).

    Args:
        links_of: The rows ``name``, ``form``, ``identifier``, ``gene`` of every cross reference of a release, every
            database and form (accessions named by the database's name, labels by the name of its labels).
        releases: The releases, ascending (adjacent releases are compared).
        selected: The ``(form, name)`` pairs the selection holds.
        database_of: The Ensembl databases (``db_name``) of every name (a name of labels may be the labels' name of
            several); a name of a database that has a selected name is no candidate (but the selected name on a form
            the selection does not hold is: :py:func:`unit_of_form`).
        share: :py:data:`CONTINUITY_SHARE`.
        scheme_share: :py:data:`SCHEME_SHARE`.

    Returns:
        Filing: The names and where they go.
    """
    units = {(str(f), str(n)) for f, n in selected}
    names = {n for _f, n in units}
    selected_databases = {db for name in names for db in database_of.get(name, (name,))}
    filing = Filing(databases={name: frozenset(database_of.get(name, (name,))) for name in names})
    sizes: dict[tuple[str, int], int] = {}
    seen: dict[str, set[int]] = {}
    moved: dict[tuple[str, str, int, str, int], dict[str, int]] = {}
    selected_ids: dict[str, dict[str, set[int]]] = {}
    spans: dict[str, dict[tuple[int, int], list[int]]] = {name: {} for name in names}
    previous: pd.DataFrame | None = None
    previous_release: int | None = None
    for release in releases:
        current = release_links(links_of(int(release)))
        pairs = list(zip(current["eform"].tolist(), current["name"].tolist()))
        unit_of_pair = {
            (f, n): (
                SELECTED + n
                if (f, n) in units
                else (
                    unit_of_form(n, f)
                    if n in names
                    else n if not any(db in selected_databases for db in database_of.get(n, (n,))) else None
                )
            )
            for f, n in set(pairs)
        }
        unit = [unit_of_pair[pair] for pair in pairs]
        current = current.assign(unit=np.asarray(unit, dtype=object))
        current = current[current["unit"].notna()]
        counted = current.drop_duplicates(["unit", "identifier", "gene"])
        for name, count in counted.groupby("unit").size().items():
            sizes[(str(name), int(release))] = int(count)
            seen.setdefault(str(name), set()).add(int(release))
        mine = counted[counted["unit"].str.startswith(SELECTED)]
        for selected_unit, group in mine.groupby("unit"):
            name = str(selected_unit)[len(SELECTED) :]
            of_form = selected_ids.setdefault(name, {})
            for form, ids in group.groupby("form")["identifier"]:
                of_form.setdefault(str(form), set()).update(ids.tolist())
            held = spans.get(name)
            if held is None:
                continue
            for link in zip(group["identifier"].tolist(), group["gene"].tolist()):
                span = held.get(link)
                if span is None:
                    held[link] = [int(release), int(release)]
                else:
                    span[1] = int(release)
            total = sum(len(v) for v in of_form.values())
            if total and max(len(v) for v in of_form.values()) < scheme_share * total:
                spans.pop(name)  # no identifier scheme: its links are not kept
        if previous is not None and previous_release is not None:
            for side, candidate, form, target, count in moved_links(previous, current).itertuples(index=False):
                if "@" in str(candidate) and str(candidate).partition("@")[0] != str(target):
                    continue  # a selected name on another form can be that database only
                # the share is of the links of the database that leaves: the candidate's at the older release, or the
                # selected database's there (it hands its links over to the candidate)
                leaving = str(candidate) if side == "older" else SELECTED + str(target)
                there = previous_release if side == "older" else int(release)  # the candidate's release
                moved_key = (str(candidate), str(form), previous_release, leaving, there)
                bucket = moved.setdefault(moved_key, {})
                bucket[str(target)] = bucket.get(str(target), 0) + int(count)
        previous, previous_release = current, int(release)

    # rule 2: a candidate continues a selected database when, at some boundary, the links that move between them are at
    # least `share` of the links of the database that leaves there (and, where the selected database leaves, of the
    # candidate's there too: a later name is, at its first release, the selected database's links), or, where the
    # candidate ends there, of the selected database's links at the next release (decision 107); the candidate's
    # identifiers of each form are the selected database that most of that form's links went with at such boundaries,
    # the other forms the main one's (unless it has another identifier scheme)
    order = sorted(int(r) for r in releases)
    position = {r: i for i, r in enumerate(order)}

    def run_of(unit: str, release: int) -> tuple[int, int]:
        """The run of releases next to each other at which a unit holds links, around a release."""
        held = seen.get(unit, set())
        first = last = position[release]
        while first > 0 and order[first - 1] in held:
            first -= 1
        while last + 1 < len(order) and order[last + 1] in held:
            last += 1
        return order[first], order[last]

    per_target: dict[tuple[str, int, str, str, int], int] = {}
    for (candidate, _form, at, base, there), targets in moved.items():
        for target, count in targets.items():
            key = (candidate, at, base, target, there)
            per_target[key] = per_target.get(key, 0) + count
    continued: dict[str, dict[str, int]] = {}
    deciding: set[tuple[str, int, str]] = set()
    runs: dict[str, set[tuple[int, int]]] = {}
    following = dict(zip(order, order[1:]))
    for (candidate, at, base, target, there), count in sorted(per_target.items()):
        size = sizes.get((base, at), 0)
        leaves = base == candidate
        # the links of the database that receives them, at the release after the boundary
        received = sizes.get((SELECTED + target if leaves else candidate, following[at]), 0)
        # decision 107: a candidate that holds no link at the next release (it ends there) hands its links over to the
        # selected database whose links at that release they are at least `share` of, however few of its own move
        ends = leaves and following[at] not in seen.get(candidate, set())
        handed = ends and received > 0 and count >= share * received
        if not handed and (not size or count < share * size or (not leaves and count < share * received)):
            continue
        into = continued.setdefault(candidate, {})
        into[target] = into.get(target, 0) + count
        deciding.add((candidate, at, base))
        runs.setdefault(candidate, set()).add(run_of(candidate, there))
        filing.evidence.append((candidate, target, at, count, size, leaves, received))
    per_form: dict[tuple[str, str], dict[str, int]] = {}
    for (candidate, form, at, base, _there), targets in moved.items():
        if (candidate, at, base) in deciding:
            for target, count in targets.items():
                bucket = per_form.setdefault((candidate, form), {})
                bucket[target] = bucket.get(target, 0) + count
    filing.releases = {unit.removeprefix(SELECTED): tuple(sorted(found)) for unit, found in seen.items()}

    # rule 3: identifier schemes of the selected databases
    for name, held in spans.items():
        identifiers = selected_ids.get(name, {})
        total = sum(len(v) for v in identifiers.values())
        if not total:
            continue
        form, ids = max(sorted(identifiers.items()), key=lambda item: len(item[1]))
        if len(ids) >= scheme_share * total:
            filing.schemes[name] = form
            filing.scheme_links[name] = {key: (span[0], span[1]) for key, span in held.items()}
    # a form of a candidate's identifiers goes to another selected database than the main continuation only where it
    # is that database's identifier scheme (``NR_#`` of RefSeq_dna to RefSeq_ncRNA), never by the chance of a few names
    for candidate, into in sorted(continued.items()):
        main = max(sorted(into), key=lambda t: into[t])
        forms = {"*": main}
        for (c, form), targets in sorted(per_form.items()):
            if c == candidate:
                target = max(sorted(targets), key=lambda t: targets[t])
                if target != main and filing.schemes.get(target) == form:
                    forms[form] = target
        filing.renamed[candidate] = forms
        if "@" in candidate:  # a selected name on another form: each run of releases is judged on its own
            filing.runs[candidate] = tuple(sorted(runs[candidate]))
    return filing


__all__ = [
    "FILING_RULES_VERSION",
    "SELECTED",
    "unit_of_form",
    "CONTINUITY_SHARE",
    "SCHEME_SHARE",
    "Filing",
    "database_filing",
    "identifier_form",
    "link_hashes",
    "moved_links",
    "release_links",
]
