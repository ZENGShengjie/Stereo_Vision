"""Parameter persistence to JSON file."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PARAMS_FILE = DATA_DIR / "display_params.json"


def load_params() -> dict | None:
    if not PARAMS_FILE.exists():
        return None
    try:
        with PARAMS_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_params(params: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PARAMS_FILE.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
