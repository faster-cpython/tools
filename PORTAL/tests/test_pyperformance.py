import json
import re


from jobs import _pyperformance, _utils


from .helpers import DATA_ROOT, NEW_FILENAME, IDEAS_GIT_URL


def test_splitting_by_suite(tmp_path):
    """
    Load an input file with 62 benchmarks (59 from pyperformance and 3 from
    pyston) and make sure they are split into results files by suite with the
    correct number of benchmarks.
    """

    git_commit = "4cd693d"  # main on 2022-09-09
    datadir = "benchmark-results"
    repo_root = tmp_path / "ideas"
    base_filename = NEW_FILENAME.stem

    github_target = _utils.GitHubTarget.from_url(IDEAS_GIT_URL)
    github_target.ensure_local(str(repo_root))

    results_repo = _pyperformance.PyperfResultsRepo.from_remote(
        IDEAS_GIT_URL, str(repo_root), datadir=datadir
    )

    input_file = DATA_ROOT / f"{base_filename}.json"

    with open(input_file, "r") as fd:
        content = json.load(fd)
        assert len(content["benchmarks"]) == 62

    results_file = _pyperformance.PyperfResultsFile(str(input_file)).read()
    assert results_file.suite == _pyperformance.PyperfUploadID.MULTI_SUITE

    # This splits the suites by default
    results_repo.add(results_file, branch=git_commit, push=False)

    with open(repo_root / datadir / f"{base_filename}-pyperformance.json", "r") as fd:
        content = json.load(fd)
        assert len(content["benchmarks"]) == 59

    with open(repo_root / datadir / f"{base_filename}-pyston.json", "r") as fd:
        content = json.load(fd)
        assert len(content["benchmarks"]) == 3
        assert ["json", "pycparser", "thrift"] == sorted(
            [x["metadata"]["name"] for x in content["benchmarks"]]
        )


def test_paths_in_readme(tmp_path):
    """
    Make sure the paths to benchmarks in the README.md files are correct.
    """

    git_commit = "4cd693d"  # main on 2022-09-09
    datadir = "benchmark-results"
    repo_root = tmp_path / "ideas"
    base_filename = NEW_FILENAME.stem

    github_target = _utils.GitHubTarget.from_url(IDEAS_GIT_URL)
    github_target.ensure_local(str(repo_root))

    results_repo = _pyperformance.PyperfResultsRepo.from_remote(
        IDEAS_GIT_URL, str(repo_root), datadir=datadir
    )

    input_file = DATA_ROOT / f"{base_filename}.json"

    results_file = _pyperformance.PyperfResultsFile(str(input_file)).read()

    results_repo.add(results_file, branch=git_commit, push=False)

    find_url_to_results = re.compile(
        r"\]\((.*cpython-.*-(?:(?:pyperformance)|(?:pyston))\.json)\)"
    )

    # The top-level README.md file should have links that include
    # `benchmark-results`
    with open(repo_root / "README.md", "r") as fd:
        content = fd.read()
        for url in find_url_to_results.finditer(content):
            url = url.groups()[0]
            assert url.startswith(f"{datadir}/cpython-")

    # The `benchmark-results/README.md` file should have links that don't
    # include `benchmark-results`
    with open(repo_root / datadir / "README.md", "r") as fd:
        content = fd.read()
        for url in find_url_to_results.finditer(content):
            url = url.groups()[0]
            assert url.startswith("cpython-")


def test_not_hidden(tmp_path):
    git_commit = "4cd693d"  # main on 2022-09-09
    datadir = "benchmark-results"
    repo_root = tmp_path / "ideas"
    base_filename = NEW_FILENAME.stem

    github_target = _utils.GitHubTarget.from_url(IDEAS_GIT_URL)
    github_target.ensure_local(str(repo_root))

    results_repo = _pyperformance.PyperfResultsRepo.from_remote(
        IDEAS_GIT_URL, str(repo_root), datadir=datadir
    )

    input_file = DATA_ROOT / f"{base_filename}.json"

    results_file = _pyperformance.PyperfResultsFile(str(input_file)).read()

    results_repo.add(results_file, branch=git_commit, push=False)

    find_url_to_results = re.compile(
        r"\]\((.*cpython-.*-(?:(?:pyperformance)|(?:pyston))\.json)\)"
    )

    # The top-level README.md file should have links that include
    # `benchmark-results`
    with open(repo_root / "README.md", "r") as fd:
        content = fd.read()

    assert "<!--\npyperformance:" not in content
    assert "<!--\npyston:" not in content


def test_save_comparison_results(tmp_path):

    git_commit = "4cd693d"  # main on 2022-09-09
    datadir = "benchmark-results"
    repo_root = tmp_path / "ideas"
    base_filename = NEW_FILENAME.stem

    github_target = _utils.GitHubTarget.from_url(IDEAS_GIT_URL)
    github_target.ensure_local(str(repo_root))

    results_repo = _pyperformance.PyperfResultsRepo.from_remote(
        IDEAS_GIT_URL, str(repo_root), datadir=datadir, baseline="3.10.4"
    )

    input_file = DATA_ROOT / f"{base_filename}.json"

    results_file = _pyperformance.PyperfResultsFile(str(input_file)).read()

    results_repo.add(results_file, branch=git_commit, push=False)

    with open(repo_root / datadir / f"{base_filename}-pyperformance.md", "r") as fd:
        content = fd.read()
        re.match(content, r"Geometric mean\s+\|\s+\(ref\)\s+\|\s+1\.27x faster")

    find_url_to_results = re.compile(
        r"\]\((.*cpython-.*-(?:(?:pyperformance)|(?:pyston))\.md)\)"
    )

    # The `benchmark-results/README.md` file should have links that don't
    # include `benchmark-results`
    with open(repo_root / datadir / "README.md", "r") as fd:
        content = fd.read()
        for url in find_url_to_results.finditer(content):
            url = url.groups()[0]
            assert url.startswith("cpython-")
