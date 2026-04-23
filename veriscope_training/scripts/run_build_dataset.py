from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from veriscope_training.config import AppConfig
from veriscope_training.pipelines.build_dataset import BuildDatasetOptions, build_processed_datasets


def main() -> int:
    config = AppConfig.load(PROJECT_ROOT)
    try:
        payload = build_processed_datasets(config, BuildDatasetOptions())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
