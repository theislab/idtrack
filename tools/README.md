# Tools

Scripts that write files of this repository from Ensembl's tables or from the package itself: the externals YAML files the package ships, the cut Ensembl tables of the tests, and the expectations of the consumer-contract test. They are for the maintainers; they are not part of the package and not in its distributions (the wheel and the source distribution hold `idtrack/` only).

Each is run from the root of the repository with the package importable (the development environment of `SETUP.md`, through `poetry run`), takes its places as arguments (`--help` lists them), and over HTTPS needs `IDTRACK_ENSEMBL_ACCESS=https` where the MySQL ports are blocked. "A local repository" is a folder of idtrack's downloaded tables (`SETUP.md`, "Configure"); one in which the organism's graph was built holds every table the tools read, and what it lacks is downloaded into it. Each tool has a test in `tests/unit/test_tools.py` that runs it on the cut tables of `tests/data/real` and compares its result with the file it writes.

| tool                   | writes                                                                         | from                                                                                                                                                      | time (one process)                                                                                             | check                                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `make_externals.py`    | `idtrack/default_config/<organism>_externals_template.yml` and `_modified.yml` | a local repository with the organism's tables up to the snapshot release and the releases after it that the files list (`--last-release`, 116 as shipped) | from the tables of a built graph: cynomolgus 5 min, pig 12 min, rhesus 11 min, mouse 41 min, human about 2.5 h | `--check`: the sha256 of the written and the shipped files, and the databases that differ                                                                      |
| `cut_coverage.py`      | nothing (prints the genes to add to `tests/data/real/<organism>.genes.txt`)    | a local repository with the organism's complete tables; the shipped modified YAML; the gene list                                                          | about a minute (pig)                                                                                           | exit status 0 when every gap can be closed; run it before `cut_tables.py`                                                                                      |
| `cut_tables.py`        | `tests/data/real/<organism>.tables.json.xz` (or `--out`)                       | a local repository with the organism's complete tables; the gene list                                                                                     | from the tables of a built graph: pig 17 min, human about 1.5 h                                                | it builds the graph again from the written file alone and fails when the two graphs differ or the file exceeds 600 kB; `--check-only` replays the present file |
| `contract_answers.py`  | the answers of the consumer-contract test (a file named on its command line)   | the cut tables, in a checkout of the tag `manuscript-2026-03` (commit `d4c2dda`, the last commit with the API before 0.1.0, for which the consumer's calls that the answers record were written; its docstring says how)                                          | the build of the two cut graphs by `d4c2dda`, and their queries                                                | the answers change only when the cut tables do                                                                                                                 |
| `consumer_contract.py` | `tests/data/real/consumer_contract.json`                                       | the present file or `--answers` (the output of `contract_answers.py`), and the cut graphs this checkout builds                                            | about 5 minutes                                                                                                | `--check`: writes nothing and compares the file it would write with the present one                                                                            |

## When to run them

- **An organism's selection of external databases changes** (`idtrack/build/default_externals.py`), or a newer Ensembl release is to be listed: `make_externals.py` for the organism, the two files copied into `idtrack/default_config/`; then `cut_coverage.py` and `cut_tables.py` for each organism of the cut tables (the cut keeps the cross references of the selected databases), then `consumer_contract.py`.
- **A test needs genes the cut tables lack**: add them to `tests/data/real/<organism>.genes.txt`, run `cut_coverage.py` and `cut_tables.py`, then `contract_answers.py` (in the checkout of the tag `manuscript-2026-03`) and `consumer_contract.py --answers ...`.
- **A change of the build changes the cut graphs** (their content digest): `consumer_contract.py`; the answers stay. `tests/unit/test_consumer_contract.py` fails until then, naming the digest.

## Examples

The externals files of pig, from a local repository in which pig's graph of release 115 was built, written to a scratch folder and compared with the shipped files:

```bash
IDTRACK_ENSEMBL_ACCESS=https poetry run python tools/make_externals.py --organism sus_scrofa \
    --repository idtrack_cache --out /tmp/externals --check
```

The cut tables of pig, checked first for the genes of the partial databases (RFAM, miRBase), then written in place:

```bash
IDTRACK_ENSEMBL_ACCESS=https poetry run python tools/cut_coverage.py --organism sus_scrofa --repository idtrack_cache
IDTRACK_ENSEMBL_ACCESS=https poetry run python tools/cut_tables.py --organism sus_scrofa --repository idtrack_cache
```

The descriptions of the consumer contract after a change of the build:

```bash
poetry run python tools/consumer_contract.py --check
poetry run python tools/consumer_contract.py
```
