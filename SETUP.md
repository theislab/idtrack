# Setting up idtrack

How to install idtrack, configure it, build and check its data, run its tests and its checks, use it, and release it. Every command below is meant to be run as written, from the folder the section starts in, on Linux or macOS with `bash` (on Windows, use the same commands in WSL, or activate a virtual environment with `idtrack-env\Scripts\activate`). Times were measured on a Linux server. The documentation (<https://idtrack.readthedocs.io/>) explains what the package does; this file is about getting it to run.

Contents: [1. Install](#1-install) · [2. Configure](#2-configure) · [3. Data: build and check a graph](#3-data-build-and-check-a-graph) · [4. Use it](#4-use-it) · [5. Develop: tests and checks](#5-develop-tests-and-checks) · [6. Release](#6-release)

## 1. Install

idtrack needs Python 3.12 to 3.15. The commands below name the interpreter as `python3.12` (`python3.12 --version` checks that it is there), since a plain `python3` is often older (an older system or conda Python: `python3 --version`); any of 3.12 to 3.15 works in its place (`python3.14`, ...). Where none is installed, get one from the system's packages (`sudo apt-get install python3.12`, `sudo dnf install python3.12`, `brew install python@3.12`), from <https://www.python.org/downloads/>, or as a conda environment (`conda create -n py312 python=3.12`, then its `bin/python3.12`).

### For use

In a new folder:

```bash
python3.12 -m venv idtrack-env
source idtrack-env/bin/activate
pip install idtrack
idtrack --version
```

`idtrack --version` prints `idtrack 0.1.2`. The same release from GitHub is `pip install "git+https://github.com/theislab/idtrack.git@v0.1.2"` (the release's tag; without `@v0.1.2` pip installs the current `main`, which may be ahead of this guide). To harmonize `.h5ad` datasets (`idtrack.harmonize.HarmonizeFeatures`), add the `harmonize` extra (anndata and scipy, about 150 MB):

```bash
pip install "idtrack[harmonize]"
```

**Python 3.15:** h5py publishes no wheels for Python 3.15 yet (h5py 3.16.0), so pip compiles h5py from source, which needs a C compiler and the HDF5 library with its headers: install them first (Debian or Ubuntu `sudo apt-get install build-essential libhdf5-dev`, Fedora or RHEL `sudo dnf install gcc hdf5-devel`, macOS `brew install hdf5`, or `conda install -c conda-forge hdf5` in a conda environment; if pip does not find HDF5, set `HDF5_DIR` to the folder that holds its `include/` and `lib/`). The install then takes about 6 minutes instead of 2.5. Python 3.12 to 3.14 need nothing of this.

### For development

```bash
git clone https://github.com/theislab/idtrack.git
cd idtrack
python3.12 -m venv ../idtrack-tools
../idtrack-tools/bin/pip install poetry poetry-plugin-export nox nox-poetry
export PATH="$PWD/../idtrack-tools/bin:$PATH"
poetry env use python3.12
make install
poetry run idtrack --version
poetry env info --path
```

`make install` (`poetry install`) installs idtrack in editable mode with the development tools at the versions of `poetry.lock`, into a virtual environment that Poetry manages, of the Python that `poetry env use` names (without it Poetry takes the Python it runs on, or one it finds, and `poetry env info --path` prints nothing when that Python is too old); the last line shows where the environment is. The clone is the current `main` (the code under development); `git checkout v0.1.2` gives the code of this release. The block takes about 4 minutes. Every development command runs in it through `poetry run` (the `make` targets do so: `make` lists them). The tools folder `../idtrack-tools` holds Poetry and nox only; any other installation of them works (with pipx, for example), as long as the export plugin sits beside Poetry and nox-poetry beside nox.

What else the checks need: [pandoc](https://pandoc.org/installing.html) for the documentation (`make docs` stops with "Please check that pandoc is installed" without it; without root, unpack the `linux-amd64` tarball of its [GitHub release](https://github.com/jgm/pandoc/releases) and put its `bin/` on the `PATH`), Node.js for the prettier hook of `nox -s pre-commit` (pre-commit installs one if `node` is not on the path), and, for the sessions of several Python versions, each interpreter on the path as `python3.12` to `python3.15`.

## 2. Configure

idtrack has no configuration file. What it needs:

- **The local repository**: a folder you choose, where idtrack keeps what it downloads from Ensembl (`idtrack_tables/`, per assembly one compressed HDF5 file per table and release; `ensembl_https_index_*.json`, the listing of Ensembl's dumps; `assembly_chain/`, Ensembl's mappings between assemblies) and the graph files it builds (`graph_<organism>_assembly<key>_min<first>_max<snapshot>.npz`). It is the `local_repository` argument of `idtrack.API.build` and the `--repository` of `idtrack build` (default `idtrack_cache` in the working folder; the command makes it). The API does not make it: `API.build` refuses a folder that does not exist. A graph file can be copied anywhere and loaded alone (`idtrack.API.load(<file>)`, `idtrack convert <file> ...`): loading needs no tables, no network and no repository. Several graphs (organisms, snapshot releases) share one repository. A file `<organism>_externals_modified.yml` in the repository replaces the selection of external databases shipped with the package (the documentation's page External databases, and tutorial 6); a graph records the selection it was built with and is refused under another.
- **How Ensembl is reached**: over HTTPS (`https://ftp.ensembl.org/pub/`, its MySQL dumps, and `https://rest.ensembl.org/`) or over Ensembl's public MySQL server (`ensembldb.ensembl.org`, ports 3306, 5306 and 3337). The tables are the same either way. Many institutional networks and clusters block the MySQL ports; then set `IDTRACK_ENSEMBL_ACCESS=https` (the default, `auto`, probes each port once per process with a short timeout and falls back to HTTPS). The same setting from Python, with the timeouts and retries: `idtrack.settings.configure_ensembl(access="https", reading_timeout=120, rest_attempts=5)`. An HTTPS proxy is read from `HTTPS_PROXY`.
- **Environment variables**: idtrack itself reads `IDTRACK_ENSEMBL_ACCESS` only (`auto` or `https`). The test suite reads `IDTRACK_SKIP_SLOW` (`0` runs the tests marked slow, as `--runslow` does). The tutorial notebooks read `IDTRACK_LOCAL_REPO` for their local repository (default `idtrack_cache`).
- **Logging**: idtrack logs through Python's `logging` under the logger `idtrack`; `idtrack.API.configure_logger("INFO")` shows its messages on the console (`idtrack build` does so).

## 3. Data: build and check a graph

A graph holds the identifier history of one organism from its first release in idtrack to a snapshot release. It is built once from Ensembl's tables of every release and then loaded from its file. With the environment of section 1 active, in its folder (the smallest organism first, 12 to 14 minutes):

```bash
export IDTRACK_ENSEMBL_ACCESS=https
idtrack build cynomolgus 115 --repository idtrack_cache
```

The last two lines it prints are the graph file and what it holds:

```text
idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz
macaca_fascicularis, releases 91 to 115, 24.1 MB, 700 s
```

The organisms are `homo_sapiens` (`human`), `mus_musculus` (`mouse`), `sus_scrofa` (`pig`), `macaca_mulatta` (`rhesus`) and `macaca_fascicularis` (`cynomolgus`); `idtrack build --help` lists the options (`--first-release`, `--rebuild`). Sizes and times vary from build to build and machine to machine (the cynomolgus graph has been 22.5 to 24.1 MB and its build from nothing 700 to 816 s). From tables already downloaded, a build takes 8 to 10 minutes (cynomolgus) to 2 h 16 to 2 h 35 (human) at a peak of 1.1 to 9.6 GB of memory; the documentation's page Time and memory (<https://idtrack.readthedocs.io/en/latest/performance.html>) gives each organism's build, peak memory, tables, graph file, load and conversion times.

Run a long build where it may run for hours (a batch job on a cluster, or `nohup`); an interrupted build starts again from the tables already downloaded.

Check a graph file: that this version reads it, that its record fits, and that its content gives the digest it records (seconds):

```bash
idtrack verify idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz
```

With the repository it was built in, `idtrack verify` also runs the graph's self-tests against those tables (cynomolgus under a minute, human about 8 minutes); the exit status is 0 when everything holds:

```bash
idtrack verify idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz --repository idtrack_cache
```

What a graph holds (organism, releases, assemblies, databases, the idtrack version and commit that built it):

```bash
idtrack info idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz
```

## 4. Use it

From the command line, with the graph of section 3 (to the snapshot release; `--final-database "VGNC Symbol"` for symbols, `--from-release` for identifiers of an older table, `--json` for the full results, `--file` for a list):

```bash
idtrack convert idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz TP53 BRCA1 MT-CO1 NOT_A_GENE
```

```text
query	status	targets	reason
TP53	converted	ENSMFAG00000038506.2
BRCA1	converted	ENSMFAG00000043956.2
MT-CO1	converted	ENSMFAG00000060073.1
NOT_A_GENE	not_found		not_in_graph
```

From Python, with the path behind an answer (`explain=True`): a gene id of release 91, on the assembly Macaca_fascicularis_5.0, that Ensembl's history ends; the search passes to the gene of the next assembly through the NCBI gene id both carried (a bridge):

```bash
python - <<'EOF'
from idtrack import API

api = API.load("idtrack_cache/graph_macaca_fascicularis_assembly6_min91_max115.npz")
result = api.convert("ENSMFAG00000000060", from_release=91, explain=True)
print(result.status, result.targets())
best = result.best[0]
for index in best.path + best.final_path:
    step = result.path.steps[index]
    print(step.kind, step.source, "->", step.target, "at", step.release)
EOF
```

```text
converted ('ENSMFAG00000054957.1',)
entry ENSMFAG00000000060 -> ENSMFAG00000000060.1 at 91
bridge_near ENSMFAG00000000060.1 -> 102131583 at 102
bridge_far 102131583 -> ENSMFAG00000054957.1 at 103
```

The documentation's Quickstart (`docs/usage.rst`), Command line (`docs/cli.rst`) and tutorials (`docs/_notebooks/`) go on from here; its page Limitations and known costs (`docs/limitations.rst`) says what this version does not do, with how often it matters.

## 5. Develop: tests and checks

In the clone, with the tools of section 1 on the path (`export PATH="$PWD/../idtrack-tools/bin:$PATH"` in a new shell).

The test suite (tests marked slow are skipped; about 30 minutes; it builds the graphs of pig and human from the cut Ensembl tables in `tests/data/real`, without network; tests that need Ensembl skip themselves when it cannot be reached):

```bash
make test
```

Its last line counts `2629 passed, 57 skipped` and the warnings of other packages (anndata's under pandas 3). One file, or the tests matching a name: `poetry run pytest tests/unit/test_api.py`, `poetry run pytest -k convert`. The tests marked slow (`--runslow`) download from Ensembl for hours; they are not part of a setup. Lint and formatting (black, isort, flake8 with the plugins of `.flake8`), and the documentation, every warning an error (into `docs/_build/html`):

```bash
make lint
make docs
```

The checks of CI are nox sessions, each in its own virtual environment with the versions of `poetry.lock` (the workflows under `.github/workflows` run them on every push):

```bash
nox --list-sessions
nox --session=pre-commit --python=3.12
nox --session=mypy --python=3.12
```

The sessions are `pre-commit`, `pip-audit`, `mypy`, `tests`, `typeguard`, `xdoctest`, `docs-build`, `coverage` and `docs` (a live preview), each for Python 3.12 to 3.15 where it applies; without `--session`, nox runs the default sessions on every interpreter it finds. Besides the nox sessions, the workflows build the sdist and the wheel and run the unit suite against the installed wheel (`.github/scripts/test_wheel.sh`, runnable locally: `PYTHON=python3.12 bash .github/scripts/test_wheel.sh dist/idtrack-*-py3-none-any.whl` after `make dist`; `PYTHON` names the interpreter of its virtual environment, `python3` if unset), and install the wheel against the oldest and newest dependency versions (`dependency_compatibility.yml`). A change is ready when the workflows pass; `docs/contributing.rst` says how to propose it.

### Files written by tools

Some files of the repository are written by scripts in `tools/` (not part of the package), never by hand: the externals YAML files the package ships (`idtrack/default_config/`), the cut Ensembl tables of the tests (`tests/data/real/<organism>.tables.json.xz`) and the expectations of the consumer-contract test (`tests/data/real/consumer_contract.json`). `tools/README.md` says what each tool writes, from what, how long it takes and how to check the result against the file in the repository. They must be written again:

- when an organism's selection of external databases changes (`idtrack/build/default_externals.py`) or a newer Ensembl release is to be listed: `tools/make_externals.py` for the organism, then the cut tables and the contract;
- when a test needs genes that the cut tables lack: the gene list, `tools/cut_coverage.py`, `tools/cut_tables.py`, then the contract's answers (`tools/contract_answers.py`, in a checkout of the tag `manuscript-2026-03`, commit `d4c2dda`, the last commit with the API before 0.1.0: the answers are those of the main consumer's calls as written for that API) and `tools/consumer_contract.py --answers ...`;
- when a change of the build changes the cut graphs (`tests/unit/test_consumer_contract.py` fails on the content digest): `tools/consumer_contract.py`.

The tools need a local repository with the organism's complete tables (one in which its graph was built, section 3); their tests (`tests/unit/test_tools.py`) run them on the cut tables.

## 6. Release

The version is written in one place, `version` in `[project]` of `pyproject.toml`; the package, the documentation and the graph files read it from the installed metadata. For a release:

1. Set the version in `pyproject.toml`, add its section to `docs/changes.rst` (what a user must do, the answers that change, what is new), set the install lines of section 1, of the README and of the installation page to the new tag, commit on `main`, and check that the workflows pass on that commit. idtrack, its web app (github.com/theislab/idtrack-web) and its benchmark repository (github.com/theislab/idtrack-reproducibility) carry one version number: a release sets the same version in all three, the web app pins the package's tag of that version, and the benchmark repository's default tree is that tag.
2. Build and check the distributions locally:

    ```bash
    make dist
    ../idtrack-tools/bin/pip install twine
    ../idtrack-tools/bin/twine check --strict dist/*
    ```

3. Tag the commit `v<version>` in each of the three repositories (for version 0.1.2, for example: `git tag v0.1.2`, `git push origin v0.1.2`; a pushed tag is never moved: a change after it is a new version) and publish a GitHub release of that tag. Publishing the release starts `.github/workflows/publish_package.yml`, which checks that the tag equals the version, builds and uploads to PyPI with the repository secret `PYPI_TOKEN` (an API token of the PyPI project). A version that is already on PyPI fails: a version is published once, so every check of this section is green before the release is published. Read the Docs builds the documentation of `main` on every push.

Steps 1 and 3 change `main`, the tags and PyPI; the maintainer does them.
