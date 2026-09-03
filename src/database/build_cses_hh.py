#!/usr/bin/env python3
"""Build the household-grain CSES database table source."""

from pathlib import Path

from cses_hh_hl_common import run


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    final, audit, issues = run(project_root, "HH")
    print(f"output=data/exp/database/final_HH_CSES.parquet rows={len(final)} columns={len(final.columns)}")
    print(audit.to_string(index=False))
    print(f"issues={len(issues)} report=data/exp/database/cses_hh_data_issues.csv")
