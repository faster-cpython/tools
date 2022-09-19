from pathlib import Path


DATA_ROOT = Path(__file__).parent / "data"

NEW_FILENAME = DATA_ROOT / "cpython-3.11.0b1-8d32a5c8c4-fc_linux-b2cf916db80e.json"

CONFIG_FILENAME = DATA_ROOT / "jobs.json"

IDEAS_GIT_URL = "https://github.com/faster-cpython/ideas"

DEFAULT_ARGS = ["--config", str(CONFIG_FILENAME)]
