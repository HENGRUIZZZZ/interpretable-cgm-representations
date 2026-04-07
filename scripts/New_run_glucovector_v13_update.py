from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.New_run_glucovector_v13 as v13


def main() -> None:
    v13.OUT_ROOT = os.path.join(REPO_ROOT, "New_paper1_results_glucovector_v13_update")
    v13.run_v13()


if __name__ == "__main__":
    main()

