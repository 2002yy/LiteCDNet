from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def run_case(case_name: str) -> None:
    from main_ablation import main

    sys.argv.extend(["--ablation_case", case_name])
    main()
