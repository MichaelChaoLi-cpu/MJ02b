#!/usr/bin/env python3
"""Extend the heat-labor diagnostics with timing and mechanism checks.

All person-level analytical inputs are queried with SELECT from ``mda.public``
under a PostgreSQL read-only session. Outputs are local diagnostic artifacts.
The raw-date audit is not merged into the models because it has not been
published to PostgreSQL and database writes are outside this script's scope.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPL_CACHE = ROOT / "data" / "exp" / ".matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from scipy.stats import norm

from run_heat_labor_credibility_audit import fit_hdfe


DEFAULT_OUTPUT = Path("data/exp/experiments/heat-labor-extensions")
BASE_CONTROLS = ["age", "age_squared", "female", "rural", "precipitation_100mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def connection_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "options": "-c default_transaction_read_only=on",
    }
    if args.user:
        values["user"] = args.user
    if args.password:
        values["password"] = args.password
    return values


def load_data(connection: psycopg.Connection, schema: str) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT
            a.survey_wave, a.survey_year, a.survey_month,
            a.person_id, a.household_id, a.admin2_code,
            d.admin1_code,
            a.sex, a.age, a.urban_rural, a.analysis_weight,
            a.household_member_count, a.household_head_education_level,
            a.worked_past_week, a.weekly_hours_worked,
            a.weekly_hours_including_zero, a.log_monthly_salary_wages,
            a.days_wbmax_ge_26c, a.wet_bulb_max_mean_c,
            a.lag_days_wbmax_ge_26c, a.lead_days_wbmax_ge_26c,
            a.lag_wet_bulb_max_mean_c, a.lead_wet_bulb_max_mean_c,
            a.precipitation_month_sum_mm,
            a.temperature_location_fallback_used,
            a.electricity_spending_positive,
            a.monthly_electricity_expense_riel,
            a.floor_area_square_meters, a.rooms_used,
            a.has_toilet_facility, a.treats_drinking_water,
            e.main_occupation_source_code,
            e.main_industry_source_code,
            e.main_employment_status_source_code,
            ed.education_level_harmonized
        FROM {}.{} a
        LEFT JOIN {}.{} d USING (admin2_code)
        LEFT JOIN {}.{} e
          ON e.survey_wave = a.survey_wave
         AND e.person_id = a.person_id
        LEFT JOIN {}.{} ed
          ON ed.survey_wave = a.survey_wave
         AND ed.person_id = a.person_id
        WHERE a.analysis_weight > 0
        ORDER BY a.survey_year, a.person_id
        """
    ).format(
        sql.Identifier(schema), sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema), sql.Identifier("dim_admin2_cambodia"),
        sql.Identifier(schema), sql.Identifier("final_EC_CSES"),
        sql.Identifier(schema), sql.Identifier("final_ED_CSES"),
    )
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    numeric = [
        "survey_year", "survey_month", "sex", "age", "urban_rural",
        "analysis_weight", "household_member_count",
        "household_head_education_level", "worked_past_week",
        "weekly_hours_worked", "weekly_hours_including_zero",
        "log_monthly_salary_wages", "days_wbmax_ge_26c",
        "wet_bulb_max_mean_c", "lag_days_wbmax_ge_26c",
        "lead_days_wbmax_ge_26c", "lag_wet_bulb_max_mean_c",
        "lead_wet_bulb_max_mean_c",
        "precipitation_month_sum_mm", "electricity_spending_positive",
        "monthly_electricity_expense_riel", "floor_area_square_meters",
        "rooms_used", "has_toilet_facility", "treats_drinking_water",
        "main_employment_status_source_code", "education_level_harmonized",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["age_squared"] = frame["age"] ** 2
    frame["female"] = frame["sex"].eq(2).where(frame["sex"].isin([1, 2])).astype(float)
    frame["rural"] = frame["urban_rural"].eq(2).where(
        frame["urban_rural"].isin([1, 2])
    ).astype(float)
    frame["wb26_5days"] = frame["days_wbmax_ge_26c"] / 5.0
    frame["lag_wb26_5days"] = frame["lag_days_wbmax_ge_26c"] / 5.0
    frame["lead_wb26_5days"] = frame["lead_days_wbmax_ge_26c"] / 5.0
    frame["wbmean_1c"] = frame["wet_bulb_max_mean_c"]
    frame["lag_wbmean_1c"] = frame["lag_wet_bulb_max_mean_c"]
    frame["lead_wbmean_1c"] = frame["lead_wet_bulb_max_mean_c"]
    frame["precipitation_100mm"] = frame["precipitation_month_sum_mm"] / 100.0
    frame["admin2_calendar_month"] = (
        frame["admin2_code"].astype("string")
        + "-m"
        + frame["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    frame["low_education"] = frame["education_level_harmonized"].le(2).where(
        frame["education_level_harmonized"].between(0, 7)
    ).astype(float)
    frame["older_worker"] = frame["age"].ge(50).where(frame["age"].notna()).astype(float)
    valid_rooms = frame["rooms_used"].gt(0) & frame["household_member_count"].gt(0)
    frame["persons_per_room"] = (
        frame["household_member_count"] / frame["rooms_used"]
    ).where(valid_rooms)
    wave_medians = frame.groupby("survey_wave")["persons_per_room"].transform("median")
    frame["high_crowding"] = frame["persons_per_room"].gt(wave_medians).where(
        frame["persons_per_room"].notna() & wave_medians.notna()
    ).astype(float)

    occupation = frame["main_occupation_source_code"].astype("string")
    valid_occupation = occupation.str.fullmatch(r"\d{3}", na=False)
    frame["occupation_major_group"] = occupation.str[0].where(valid_occupation)
    frame["agriculture_occupation_candidate"] = (
        frame["occupation_major_group"].eq("6").where(valid_occupation).astype(float)
    )
    frame["manual_outdoor_occupation_candidate"] = (
        frame["occupation_major_group"].isin(list("6789")).where(valid_occupation).astype(float)
    )
    return frame


def timing_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame["worked_past_week"].notna()
    masks = [
        ("all_current_month_matches", "All nine currently matched waves", base),
        ("exclude_2019", "Exclude 2019 release", base & frame["survey_wave"].ne("2019")),
        ("exclude_2021", "Exclude 2021 release", base & frame["survey_wave"].ne("2021")),
        (
            "exclude_2019_2021",
            "Exclude releases with raw survey-year mismatch (2019, 2021)",
            base & ~frame["survey_wave"].isin(["2019", "2021"]),
        ),
        (
            "exclude_2011_12",
            "Exclude unresolved cross-year release (2011-12)",
            base & frame["survey_wave"].ne("2011-12"),
        ),
        (
            "exclude_known_timing_risk",
            "Exclude 2011-12, 2019, and 2021 timing-risk releases",
            base & ~frame["survey_wave"].isin(["2011-12", "2019", "2021"]),
        ),
        (
            "documented_calendar_year_only",
            "Only waves documented as January-December of nominal year",
            base & frame["survey_wave"].isin(["2009", "2013", "2014"]),
        ),
    ]
    rows: list[dict[str, object]] = []
    for specification, label, mask in masks:
        result, _model = fit_hdfe(
            frame,
            f"timing_{specification}",
            "worked_past_week",
            ["wb26_5days"],
            BASE_CONTROLS,
            sample_label=label,
            mask=mask,
        )
        row = result[0]
        selected = frame.loc[mask]
        row["timing_sample"] = specification
        row["included_waves"] = ", ".join(
            selected.sort_values("survey_year")["survey_wave"].drop_duplicates()
        )
        row["excluded_waves"] = ", ".join(
            wave for wave in frame.sort_values("survey_year")["survey_wave"].drop_duplicates()
            if wave not in set(selected["survey_wave"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def binary_effect_modification(
    frame: pd.DataFrame,
    specification: str,
    modifier: str,
    label_zero: str,
    label_one: str,
    *,
    outcome: str = "worked_past_week",
    controls: list[str] | None = None,
    mask: pd.Series | None = None,
    caveat: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = controls or BASE_CONTROLS
    selected_mask = pd.Series(True, index=frame.index) if mask is None else mask.copy()
    selected_mask &= frame[modifier].notna() & frame[outcome].notna()
    interaction = f"wb26_x_{modifier}"
    working = frame.copy()
    working[interaction] = working["wb26_5days"] * working[modifier]
    effective_controls = [control for control in controls if control != modifier]
    terms = ["wb26_5days", modifier, interaction]
    raw_rows, model = fit_hdfe(
        working,
        specification,
        outcome,
        terms,
        effective_controls,
        sample_label=f"nonmissing {modifier}",
        mask=selected_mask,
    )
    raw = pd.DataFrame(raw_rows)

    params = model.params
    covariance = model.cov
    slopes = []
    for value, label in [(0, label_zero), (1, label_one)]:
        estimate = float(params["wb26_5days"])
        variance = float(covariance.loc["wb26_5days", "wb26_5days"])
        if value == 1:
            estimate += float(params[interaction])
            variance += float(covariance.loc[interaction, interaction])
            variance += 2 * float(covariance.loc["wb26_5days", interaction])
        standard_error = float(np.sqrt(max(variance, 0)))
        z_value = estimate / standard_error if standard_error > 0 else np.nan
        subgroup_mask = selected_mask & working[modifier].eq(value)
        slopes.append({
            "specification": specification,
            "outcome": outcome,
            "modifier": modifier,
            "group_value": value,
            "group_label": label,
            "heat_slope": estimate,
            "standard_error": standard_error,
            "ci_lower_95": estimate - 1.96 * standard_error,
            "ci_upper_95": estimate + 1.96 * standard_error,
            "p_value": float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
            "interaction_coefficient": float(params[interaction]),
            "interaction_p_value": float(model.pvalues[interaction]),
            "n_obs_model": int(model.nobs),
            "n_rows_group": int(subgroup_mask.sum()),
            "caveat": caveat,
        })
    return pd.DataFrame(slopes), raw


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, group in frame.groupby("survey_wave", sort=False):
        workers = group["worked_past_week"].eq(1)
        rows.append({
            "survey_wave": wave,
            "person_rows": len(group),
            "worked_past_week_observed": int(group["worked_past_week"].notna().sum()),
            "own_education_available": int(group["low_education"].notna().sum()),
            "electricity_proxy_available": int(group["electricity_spending_positive"].notna().sum()),
            "crowding_proxy_available": int(group["high_crowding"].notna().sum()),
            "current_workers": int(workers.sum()),
            "occupation_available_among_workers": int((workers & group["occupation_major_group"].notna()).sum()),
            "occupation_coverage_among_workers": float(
                group.loc[workers, "occupation_major_group"].notna().mean()
            ) if workers.any() else 0.0,
            "hours_available_among_workers": int((workers & group["weekly_hours_worked"].notna()).sum()),
        })
    return pd.DataFrame(rows)


def run_extensions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timing = timing_sensitivity(frame)
    slope_tables = []
    raw_tables = []

    subgroup_specs = [
        (
            "distribution_female", "female", "Male", "Female",
            ["age", "age_squared", "rural", "precipitation_100mm"],
            "Sex is pre-exposure, but the interaction remains associational.",
        ),
        (
            "distribution_rural", "rural", "Urban", "Rural",
            ["age", "age_squared", "female", "precipitation_100mm"],
            "Residence is pre-exposure, but urban-rural coding and heat exposure are district-level.",
        ),
        (
            "distribution_low_education", "low_education", "Lower-secondary or above", "None/preschool/primary",
            BASE_CONTROLS,
            "Education is harmonized across waves; estimates are descriptive effect modification.",
        ),
        (
            "distribution_older_worker", "older_worker", "15-49", "50-64",
            ["female", "rural", "precipitation_100mm"],
            "Age-group differences are descriptive and may reflect occupation or cohort composition.",
        ),
        (
            "adaptation_electricity_spending", "electricity_spending_positive", "No positive electricity spending", "Positive electricity spending",
            BASE_CONTROLS,
            "Electricity spending is not verified access to cooling and may proxy income or urban infrastructure.",
        ),
        (
            "adaptation_high_crowding", "high_crowding", "At/below wave median persons per room", "Above wave median persons per room",
            BASE_CONTROLS,
            "Crowding is a housing-resource proxy, not a direct thermal-adaptation measure.",
        ),
    ]
    for name, modifier, zero, one, controls, caveat in subgroup_specs:
        slopes, raw = binary_effect_modification(
            frame, name, modifier, zero, one,
            controls=controls, caveat=caveat,
        )
        slope_tables.append(slopes)
        raw_tables.append(raw)

    worker_hours = frame["worked_past_week"].eq(1) & frame["weekly_hours_worked"].notna()
    for name, modifier, zero, one, caveat in [
        (
            "occupation_agriculture_hours", "agriculture_occupation_candidate",
            "Other occupation major groups", "Major group 6 candidate",
            "Occupation codes are wave-specific and observed only for workers; this is a worker-hours diagnostic, not employment participation evidence.",
        ),
        (
            "occupation_manual_outdoor_hours", "manual_outdoor_occupation_candidate",
            "Major groups 0-5", "Major groups 6-9 candidate",
            "The 6-9 grouping is a coarse manual/outdoor candidate, not a validated cross-wave exposure classification.",
        ),
    ]:
        slopes, raw = binary_effect_modification(
            frame, name, modifier, zero, one,
            outcome="weekly_hours_worked", controls=BASE_CONTROLS,
            mask=worker_hours, caveat=caveat,
        )
        slope_tables.append(slopes)
        raw_tables.append(raw)

    slopes = pd.concat(slope_tables, ignore_index=True)
    raw = pd.concat(raw_tables, ignore_index=True)
    coverage = coverage_summary(frame)
    return timing, slopes, raw, coverage


def plot_timing(timing: pd.DataFrame, path: Path) -> None:
    table = timing.reset_index(drop=True).iloc[::-1]
    labels = {
        "all_current_month_matches": "Current all-wave match",
        "exclude_2019": "Exclude 2019",
        "exclude_2021": "Exclude 2021",
        "exclude_2019_2021": "Exclude 2019 and 2021",
        "exclude_2011_12": "Exclude 2011-12",
        "exclude_known_timing_risk": "Exclude all known timing-risk waves",
        "documented_calendar_year_only": "Documented calendar-year waves only",
    }
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.errorbar(
        table["coefficient"] * 100,
        np.arange(len(table)),
        xerr=1.96 * table["standard_error"] * 100,
        fmt="o", color="#1d3557", ecolor="#457b9d", capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(np.arange(len(table)), table["timing_sample"].map(labels))
    axis.set_xlabel("Percentage-point change per 5 additional WBmax >= 26 C days")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_subgroups(slopes: pd.DataFrame, path: Path) -> None:
    work = slopes.loc[slopes["outcome"].eq("worked_past_week")].copy()
    dimensions = {
        "distribution_female": "Sex",
        "distribution_rural": "Residence",
        "distribution_low_education": "Education",
        "distribution_older_worker": "Age",
        "adaptation_electricity_spending": "Electricity proxy",
        "adaptation_high_crowding": "Crowding proxy",
    }
    work["label"] = work["specification"].map(dimensions) + " - " + work["group_label"]
    work = work.iloc[::-1].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.errorbar(
        work["heat_slope"] * 100,
        np.arange(len(work)),
        xerr=1.96 * work["standard_error"] * 100,
        fmt="o", color="#9b2226", ecolor="#bb3e03", capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_yticks(np.arange(len(work)), work["label"])
    axis.set_xlabel("Subgroup slope: percentage points per 5 additional hot days")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    output: Path,
    frame: pd.DataFrame,
    timing: pd.DataFrame,
    slopes: pd.DataFrame,
    raw: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "timing_sensitivity": timing,
        "subgroup_effects": slopes,
        "interaction_terms": raw,
        "mechanism_coverage": coverage,
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    plot_timing(timing, output / "timing_sensitivity.png")
    plot_subgroups(slopes, output / "subgroup_effects.png")

    all_model = timing.loc[timing["timing_sample"].eq("all_current_month_matches")].iloc[0]
    exclude_two = timing.loc[timing["timing_sample"].eq("exclude_2019_2021")].iloc[0]
    exclude_three = timing.loc[timing["timing_sample"].eq("exclude_known_timing_risk")].iloc[0]
    report = f"""# Heat-labor extension diagnostics

Status: local diagnostic evidence; not a formal AnaSOP output and not causal.

## Read-only contract

- Analytical inputs were obtained by SELECT from `mda.public`.
- PostgreSQL session: `default_transaction_read_only=on`.
- Database writes: none.
- Person rows read: {len(frame):,}.

## Timing sensitivity

The current month-matched estimate is {all_model.coefficient * 100:.3f}
percentage points per five WBmax >= 26 C days (p={all_model.p_value:.3f}).
Excluding 2019 and 2021, whose raw files contain survey years outside their
nominal release years, gives {exclude_two.coefficient * 100:.3f} percentage
points (p={exclude_two.p_value:.3f}). Excluding those waves plus the unresolved
2011-12 cross-year release gives {exclude_three.coefficient * 100:.3f}
percentage points (p={exclude_three.p_value:.3f}).

These are exclusion diagnostics, not corrected-date estimates. Correct
estimation requires the survey-date layer and 2020/2022 weather to be available
inside the database under explicit publication authorization.

## Mechanism boundaries

- Sex, residence, age, and harmonized education are distributional modifiers.
- Positive electricity spending and persons per room are resource proxies, not
  direct evidence of air conditioning or thermal adaptation.
- Occupation results condition on current workers and use unvalidated broad
  groups derived from wave-specific codes. They describe worker hours only and
  cannot explain entry into employment.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "database": "mda",
        "schema": "public",
        "database_access": "read-only SELECT",
        "input_tables": [
            "final_HEAT_LABOR_ANALYTIC", "final_EC_CSES", "final_ED_CSES",
            "dim_admin2_cambodia",
        ],
        "person_rows": len(frame),
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    with psycopg.connect(**connection_args(args)) as connection:
        frame = load_data(connection, args.schema)
        connection.rollback()
    if frame.empty:
        raise RuntimeError("Read-only database query returned no rows")
    timing, slopes, raw, coverage = run_extensions(frame)
    save_outputs(output, frame, timing, slopes, raw, coverage)
    print(f"read_only_input_rows={len(frame)}")
    print(f"timing_models={len(timing)} subgroup_slopes={len(slopes)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
