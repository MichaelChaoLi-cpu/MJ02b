#!/usr/bin/env python3
"""Build the household-grain CSES housing database table source."""

from pathlib import Path

from cses_housing import run


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    final, audit, issues = run(project_root)
    print(f"output=data/exp/database/final_HO_CSES.parquet rows={len(final)} columns={len(final.columns)}")
    print(audit.to_string(index=False))
    print(f"issues={len(issues)} report=data/exp/database/cses_ho_data_issues.csv")
