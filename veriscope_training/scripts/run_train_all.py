from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from veriscope_training.config import AppConfig
from veriscope_training.pipelines.train_all import train_all_enabled_models


def main() -> int:
    config = AppConfig.load(PROJECT_ROOT)
    payload = train_all_enabled_models(config, include_transformers=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
