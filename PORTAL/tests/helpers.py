import json
from pathlib import Path


from jobs import _utils


DATA_ROOT = Path(__file__).parent / "data"

NEW_FILENAME = DATA_ROOT / "cpython-3.11.0b1-8d32a5c8c4-fc_linux-b2cf916db80e.json"

CONFIG_FILENAME = DATA_ROOT / "jobs.json"

IDEAS_GIT_URL = "https://github.com/faster-cpython/ideas"


def setup_temp_env(tmp_path: Path):
    cfg_file = tmp_path / "jobs.json"
    results_repo_root = (tmp_path / "ideas")

    github_target = _utils.GitHubTarget.from_url(IDEAS_GIT_URL)
    github_target.ensure_local(str(results_repo_root))

    content = json.loads(CONFIG_FILENAME.read_text())
    content["data_dir"] = str(tmp_path / "BENCH")
    content["results_repo_root"] = str(results_repo_root)

    cfg_file.write_text(json.dumps(content))

    return ["--config", str(cfg_file)]
