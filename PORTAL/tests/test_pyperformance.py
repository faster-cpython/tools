import json
from pathlib import Path


from jobs import _pyperformance, _utils


DATA_ROOT = Path(__file__).parent / "data"


def test_splitting_by_suite(tmp_path):
    """
    Load an input file with 62 benchmarks (59 from pyperformance and 3 from
    pyston) and make sure they are split into results files by suite with the
    correct number of benchmarks.
    """

    git_url = "https://github.com/faster-cpython/ideas"
    git_commit = "4cd693d"  # main on 2022-09-09
    datadir = "benchmark-results"
    repo_root = tmp_path / "ideas"
    base_filename = "cpython-3.11.0b1-8d32a5c8c4-fc_linux-b2cf916db80e"

    github_target = _utils.GitHubTarget.from_url(git_url)
    github_target.ensure_local(str(repo_root))

    results_repo = _pyperformance.PyperfResultsRepo.from_remote(
        git_url, str(repo_root), datadir=datadir
    )

    input_file = DATA_ROOT / f"{base_filename}.json"

    with open(input_file, "r") as fd:
        content = json.load(fd)
        assert len(content["benchmarks"]) == 62

    results_file = _pyperformance.PyperfResultsFile(str(input_file)).read()

    results_repo.add(
        results_file, branch=git_commit, push=False
    )

    with open(
        repo_root / datadir / f"{base_filename}-pyperformance.json", "r"
    ) as fd:
        content = json.load(fd)
        assert len(content["benchmarks"]) == 59

    with open(
        repo_root / datadir / f"{base_filename}-pyston.json", "r"
    ) as fd:
        content = json.load(fd)
        assert len(content["benchmarks"]) == 3
