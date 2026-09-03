#!/usr/bin/env python3
"""Run provisional CSES district-month humid-heat/labor experiments.

All analytical input is queried from mda.public. Results are descriptive or
associational because exact/raw survey timing is not yet represented in the
database analytic table and the CSES is a repeated cross-section.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import statsmodels.formula.api as smf
from linearmodels.iv import AbsorbingLS
from psycopg import sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--output", default="data/exp/experiments/heat-labor-baseline")
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


def load_analysis_data(connection: psycopg.Connection, schema: str) -> pd.DataFrame:
    query = sql.SQL(
        """
        SELECT
            survey_wave, survey_year, survey_month, person_id, household_id,
            admin2_code, sex, age, urban_rural, analysis_weight,
            worked_past_week, weekly_hours_including_zero,
            log_monthly_salary_wages,
            days_wbmax_ge_26c, wet_bulb_max_mean_c,
            days_tmax_ge_35c,
            apparent_temperature_max_mean_c,
            precipitation_month_sum_mm,
            lag_days_wbmax_ge_26c, lead_days_wbmax_ge_26c,
            temperature_location_fallback_used
        FROM {}.{}
        WHERE analysis_weight > 0
        """
    ).format(sql.Identifier(schema), sql.Identifier("final_HEAT_LABOR_ANALYTIC"))
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=columns)
    numeric = [
        "survey_year", "survey_month", "sex", "age", "analysis_weight",
        "worked_past_week", "weekly_hours_including_zero",
        "log_monthly_salary_wages", "days_wbmax_ge_26c",
        "wet_bulb_max_mean_c", "days_tmax_ge_35c",
        "apparent_temperature_max_mean_c", "precipitation_month_sum_mm",
        "lag_days_wbmax_ge_26c", "lead_days_wbmax_ge_26c",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["year_month"] = (
        frame["survey_year"].astype("Int64").astype("string")
        + "-"
        + frame["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    frame["wb26_5days"] = frame["days_wbmax_ge_26c"] / 5.0
    frame["tmax35_5days"] = frame["days_tmax_ge_35c"] / 5.0
    frame["lag_wb26_5days"] = frame["lag_days_wbmax_ge_26c"] / 5.0
    frame["lead_wb26_5days"] = frame["lead_days_wbmax_ge_26c"] / 5.0
    frame["precipitation_100mm"] = frame["precipitation_month_sum_mm"] / 100.0
    frame["age_squared"] = frame["age"] ** 2
    frame["female"] = frame["sex"].eq(2).astype(float)
    frame["urban_rural_2"] = frame["urban_rural"].astype("string").eq("2").astype(float)
    frame["admin2_calendar_month"] = (
        frame["admin2_code"].astype("string")
        + "-m"
        + frame["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    return frame


BASE_CONTROLS = (
    "age + I(age ** 2) + C(sex) + C(urban_rural) "
    "+ C(admin2_code) + C(year_month)"
)

SPECS = [
    {
        "specification": "primary_work_participation",
        "outcome": "worked_past_week",
        "exposure": "wb26_5days",
        "sample": "working_age_all_waves",
        "formula": f"worked_past_week ~ wb26_5days + {BASE_CONTROLS}",
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "primary_total_weekly_hours",
        "outcome": "weekly_hours_including_zero",
        "exposure": "wb26_5days",
        "sample": "working_age_hours_observed",
        "formula": f"weekly_hours_including_zero ~ wb26_5days + {BASE_CONTROLS}",
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "rainfall_controlled_work_participation",
        "outcome": "worked_past_week",
        "exposure": "wb26_5days",
        "sample": "working_age_all_waves_with_rainfall",
        "formula": (
            f"worked_past_week ~ wb26_5days + precipitation_100mm + {BASE_CONTROLS}"
        ),
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
        "covariates": "age + age squared + sex + urban-rural + monthly precipitation",
    },
    {
        "specification": "primary_log_wage_conditional",
        "outcome": "log_monthly_salary_wages",
        "exposure": "wb26_5days",
        "sample": "positive_or_zero_reported_wage_2009_2014",
        "formula": f"log_monthly_salary_wages ~ wb26_5days + {BASE_CONTROLS}",
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "alternative_continuous_wet_bulb",
        "outcome": "worked_past_week",
        "exposure": "wet_bulb_max_mean_c",
        "sample": "working_age_all_waves",
        "formula": f"worked_past_week ~ wet_bulb_max_mean_c + {BASE_CONTROLS}",
        "terms": ["wet_bulb_max_mean_c"],
        "exposure_unit": "1 C higher monthly mean daily WBmax",
    },
    {
        "specification": "dry_bulb_comparison",
        "outcome": "worked_past_week",
        "exposure": "tmax35_5days",
        "sample": "working_age_all_waves",
        "formula": f"worked_past_week ~ tmax35_5days + {BASE_CONTROLS}",
        "terms": ["tmax35_5days"],
        "exposure_unit": "5 additional Tmax>=35C days in survey month",
    },
    {
        "specification": "lagged_month_exposure",
        "outcome": "worked_past_week",
        "exposure": "lag_wb26_5days",
        "sample": "working_age_with_previous_month",
        "formula": f"worked_past_week ~ lag_wb26_5days + {BASE_CONTROLS}",
        "terms": ["lag_wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in prior month",
    },
    {
        "specification": "future_month_placebo",
        "outcome": "worked_past_week",
        "exposure": "lead_wb26_5days",
        "sample": "working_age_with_next_month",
        "formula": f"worked_past_week ~ lead_wb26_5days + {BASE_CONTROLS}",
        "terms": ["lead_wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in future month",
    },
    {
        "specification": "distributed_current_and_lag",
        "outcome": "worked_past_week",
        "exposure": "wb26_5days_and_lag",
        "required_exposures": ["wb26_5days", "lag_wb26_5days"],
        "sample": "working_age_with_current_and_previous_month",
        "formula": (
            f"worked_past_week ~ wb26_5days + lag_wb26_5days + {BASE_CONTROLS}"
        ),
        "terms": ["wb26_5days", "lag_wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in current or prior month",
    },
    {
        "specification": "joint_current_and_future_placebo",
        "outcome": "worked_past_week",
        "exposure": "wb26_5days_and_lead",
        "required_exposures": ["wb26_5days", "lead_wb26_5days"],
        "sample": "working_age_with_current_and_next_month",
        "formula": (
            f"worked_past_week ~ wb26_5days + lead_wb26_5days + {BASE_CONTROLS}"
        ),
        "terms": ["wb26_5days", "lead_wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in current or future month",
    },
    {
        "specification": "hours_excluding_2011_release_gap",
        "outcome": "weekly_hours_including_zero",
        "exposure": "wb26_5days",
        "sample": "hours_observed_excluding_2011",
        "exclude_waves": ["2011-12"],
        "formula": f"weekly_hours_including_zero ~ wb26_5days + {BASE_CONTROLS}",
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "sex_heterogeneity",
        "outcome": "worked_past_week",
        "exposure": "wb26_5days",
        "sample": "working_age_nonmissing_sex",
        "formula": (
            "worked_past_week ~ wb26_5days * C(sex) + age + I(age ** 2) "
            "+ C(urban_rural) + C(admin2_code) + C(year_month)"
        ),
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days; interaction relative to sex=1",
        "include_prefix": "wb26_5days:C(sex)",
    },
    {
        "specification": "urban_rural_heterogeneity",
        "outcome": "worked_past_week",
        "exposure": "wb26_5days",
        "sample": "working_age_nonmissing_residence",
        "formula": (
            "worked_past_week ~ wb26_5days * C(urban_rural) + age + I(age ** 2) "
            "+ C(sex) + C(admin2_code) + C(year_month)"
        ),
        "terms": ["wb26_5days"],
        "exposure_unit": "5 additional WBmax>=26C days; residence interactions reported when estimable",
        "include_prefix": "wb26_5days:C(urban_rural)",
    },
]

HDFE_SPECS = [
    {
        "specification": "hdfe_local_seasonality_work",
        "outcome": "worked_past_week",
        "exposures": ["wb26_5days"],
        "sample": "working_age_all_waves",
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "hdfe_local_seasonality_work_rainfall",
        "outcome": "worked_past_week",
        "exposures": ["wb26_5days"],
        "weather_controls": ["precipitation_100mm"],
        "sample": "working_age_all_waves_with_rainfall",
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "hdfe_local_seasonality_hours",
        "outcome": "weekly_hours_including_zero",
        "exposures": ["wb26_5days"],
        "sample": "working_age_hours_observed",
        "exposure_unit": "5 additional WBmax>=26C days in survey month",
    },
    {
        "specification": "hdfe_local_seasonality_current_lag",
        "outcome": "worked_past_week",
        "exposures": ["wb26_5days", "lag_wb26_5days"],
        "sample": "working_age_with_current_and_previous_month",
        "exposure_unit": "5 additional WBmax>=26C days in current or prior month",
    },
    {
        "specification": "hdfe_local_seasonality_future_placebo",
        "outcome": "worked_past_week",
        "exposures": ["wb26_5days", "lead_wb26_5days"],
        "sample": "working_age_with_current_and_next_month",
        "exposure_unit": "5 additional WBmax>=26C days in current or future month",
    },
]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return math.nan
    return float(np.average(values[valid], weights=weights[valid]))


def fit_spec(frame: pd.DataFrame, spec: dict[str, object]) -> list[dict[str, object]]:
    required_exposures = [
        str(value) for value in spec.get("required_exposures", [spec["exposure"]])
    ]
    required = [
        str(spec["outcome"]), *required_exposures, "age", "sex",
        "urban_rural", "admin2_code", "year_month", "analysis_weight",
    ]
    subset = frame.dropna(subset=required).copy()
    if spec["specification"] == "primary_total_weekly_hours":
        subset = subset.loc[subset["weekly_hours_including_zero"].notna()].copy()
    if spec["specification"] == "primary_log_wage_conditional":
        subset = subset.loc[subset["survey_year"].ge(2009)].copy()
    if spec.get("exclude_waves"):
        subset = subset.loc[~subset["survey_wave"].isin(spec["exclude_waves"])].copy()
    if subset["admin2_code"].nunique() < 30:
        raise RuntimeError(f"Too few clusters for {spec['specification']}")

    model = smf.wls(
        formula=str(spec["formula"]),
        data=subset,
        weights=subset["analysis_weight"],
        missing="drop",
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": subset["admin2_code"], "use_correction": True},
    )
    terms = list(spec["terms"])
    prefix = spec.get("include_prefix")
    if prefix:
        terms.extend(term for term in model.params.index if term.startswith(str(prefix)))
    terms = list(dict.fromkeys(terms))

    results = []
    for term in terms:
        if term not in model.params.index:
            continue
        coefficient = float(model.params[term])
        standard_error = float(model.bse[term])
        results.append(
            {
                "experiment_status": "provisional",
                "specification": spec["specification"],
                "outcome": spec["outcome"],
                "exposure": spec["exposure"],
                "term": term,
                "sample": spec["sample"],
                "coefficient": coefficient,
                "standard_error": standard_error,
                "ci_lower_95": coefficient - 1.96 * standard_error,
                "ci_upper_95": coefficient + 1.96 * standard_error,
                "p_value": float(model.pvalues[term]),
                "n_obs": int(model.nobs),
                "n_clusters": int(subset["admin2_code"].nunique()),
                "weighted_outcome_mean": weighted_mean(
                    subset[str(spec["outcome"])], subset["analysis_weight"]
                ),
                "exposure_unit": spec["exposure_unit"],
                "fixed_effects": "admin2 + survey-year-by-calendar-month",
                "covariates": spec.get(
                    "covariates", "age + age squared + sex + urban-rural"
                ),
                "variance": "admin2-cluster-robust",
                "interpretation_limit": (
                    "Associational provisional estimate: survey timing is monthly, "
                    "the exact interview day is unavailable, and the design is repeated cross-section."
                ),
            }
        )
    if not results:
        raise RuntimeError(f"No requested coefficient found for {spec['specification']}")
    print(
        f"estimated={spec['specification']} n={int(model.nobs)} "
        f"clusters={subset['admin2_code'].nunique()}",
        flush=True,
    )
    return results


def fit_hdfe_spec(frame: pd.DataFrame, spec: dict[str, object]) -> list[dict[str, object]]:
    exposures = [str(value) for value in spec["exposures"]]
    weather_controls = [str(value) for value in spec.get("weather_controls", [])]
    controls = [
        "age", "age_squared", "female", "urban_rural_2", *weather_controls
    ]
    required = [
        str(spec["outcome"]), *exposures, *controls, "analysis_weight",
        "admin2_code", "admin2_calendar_month", "survey_year",
    ]
    subset = frame.dropna(subset=required).copy()
    exog = subset[exposures + controls].astype(float).copy()
    exog["constant"] = 1.0
    absorb = pd.DataFrame(
        {
            "admin2_calendar_month": subset["admin2_calendar_month"].astype("category"),
            "survey_year": subset["survey_year"].astype("Int64").astype("category"),
        },
        index=subset.index,
    )
    model = AbsorbingLS(
        dependent=subset[str(spec["outcome"])].astype(float),
        exog=exog,
        absorb=absorb,
        weights=subset["analysis_weight"].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=subset[["admin2_code"]],
        debiased=True,
    )
    results = []
    for term in exposures:
        coefficient = float(model.params[term])
        standard_error = float(model.std_errors[term])
        results.append(
            {
                "experiment_status": "provisional",
                "specification": spec["specification"],
                "outcome": spec["outcome"],
                "exposure": "+".join(exposures),
                "term": term,
                "sample": spec["sample"],
                "coefficient": coefficient,
                "standard_error": standard_error,
                "ci_lower_95": coefficient - 1.96 * standard_error,
                "ci_upper_95": coefficient + 1.96 * standard_error,
                "p_value": float(model.pvalues[term]),
                "n_obs": int(model.nobs),
                "n_clusters": int(subset["admin2_code"].nunique()),
                "weighted_outcome_mean": weighted_mean(
                    subset[str(spec["outcome"])], subset["analysis_weight"]
                ),
                "exposure_unit": spec["exposure_unit"],
                "fixed_effects": "admin2-by-calendar-month + survey year",
                "covariates": (
                    "age + age squared + sex + urban-rural"
                    + (" + monthly precipitation" if weather_controls else "")
                ),
                "variance": "admin2-cluster-robust",
                "interpretation_limit": (
                    "Associational provisional estimate: survey timing is monthly, "
                    "the exact interview day is unavailable, and the design is repeated cross-section."
                ),
            }
        )
    print(
        f"estimated={spec['specification']} n={int(model.nobs)} "
        f"clusters={subset['admin2_code'].nunique()}",
        flush=True,
    )
    return results


def sample_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, group in frame.groupby("survey_wave", sort=False):
        weights = group["analysis_weight"]
        rows.append(
            {
                "survey_wave": wave,
                "survey_year": int(group["survey_year"].iloc[0]),
                "person_rows": len(group),
                "admin2_clusters": group["admin2_code"].nunique(),
                "worked_observed": int(group["worked_past_week"].notna().sum()),
                "inclusive_hours_observed": int(
                    group["weekly_hours_including_zero"].notna().sum()
                ),
                "wage_observed": int(group["log_monthly_salary_wages"].notna().sum()),
                "weighted_work_rate": weighted_mean(group["worked_past_week"], weights),
                "weighted_weekly_hours": weighted_mean(
                    group["weekly_hours_including_zero"], weights
                ),
                "mean_wb26_days": float(group["days_wbmax_ge_26c"].mean()),
                "mean_precipitation_mm": float(
                    group["precipitation_month_sum_mm"].mean()
                ),
                "coastal_fallback_rows": int(
                    group["temperature_location_fallback_used"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("survey_year", ignore_index=True)


def write_experiment_artifacts(
    output: Path, estimates: pd.DataFrame, summary: pd.DataFrame
) -> None:
    estimates.to_csv(output / "provisional_estimates.csv", index=False)
    summary.to_csv(output / "sample_summary.csv", index=False)
    registry_rows = [
        {
            "specification": spec["specification"],
            "estimator": "statsmodels WLS",
            "outcome": spec["outcome"],
            "exposure": spec["exposure"],
            "sample": spec["sample"],
            "fixed_effects": "admin2 + survey-year-by-calendar-month",
            "formula_or_terms": spec["formula"],
        }
        for spec in SPECS
    ] + [
        {
            "specification": spec["specification"],
            "estimator": "linearmodels AbsorbingLS",
            "outcome": spec["outcome"],
            "exposure": "+".join(spec["exposures"]),
            "sample": spec["sample"],
            "fixed_effects": "admin2-by-calendar-month + survey year",
            "formula_or_terms": "+".join(
                [*spec["exposures"], *spec.get("weather_controls", [])]
            ),
        }
        for spec in HDFE_SPECS
    ]
    pd.DataFrame(registry_rows).to_csv(output / "model_registry.csv", index=False)

    def result(specification: str, term: str) -> pd.Series:
        match = estimates.loc[
            estimates["specification"].eq(specification)
            & estimates["term"].eq(term)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Expected one result for {specification}/{term}")
        return match.iloc[0]

    primary = result("primary_work_participation", "wb26_5days")
    strict = result("hdfe_local_seasonality_work", "wb26_5days")
    strict_rain = result(
        "hdfe_local_seasonality_work_rainfall", "wb26_5days"
    )
    hours = result("hdfe_local_seasonality_hours", "wb26_5days")
    future = result("hdfe_local_seasonality_future_placebo", "lead_wb26_5days")
    text = f"""# Provisional humid-heat/labor experiment

Status: **provisional diagnostic, not a final causal result**.

## Database input and coverage

- PostgreSQL-only input: `mda.public.final_HEAT_LABOR_ANALYTIC`
- Person rows: {int(summary['person_rows'].sum()):,}
- Waves: {', '.join(summary['survey_wave'].astype(str))}
- Climate: ERA5-Land district-month Tmax and WBmax plus ERA5 precipitation and apparent temperature; nine survey years through 2021
- Exposure unit: five additional days in the survey month with daily WBmax at least 26 C

## Main estimates

| specification | coefficient | 95% CI | p-value | n |
|---|---:|---:|---:|---:|
| District FE + survey-year-by-month FE, work participation | {primary.coefficient:.4f} | [{primary.ci_lower_95:.4f}, {primary.ci_upper_95:.4f}] | {primary.p_value:.3f} | {int(primary.n_obs):,} |
| District-by-calendar-month FE + year FE, work participation | {strict.coefficient:.4f} | [{strict.ci_lower_95:.4f}, {strict.ci_upper_95:.4f}] | {strict.p_value:.3f} | {int(strict.n_obs):,} |
| Same strict FE + monthly precipitation control | {strict_rain.coefficient:.4f} | [{strict_rain.ci_lower_95:.4f}, {strict_rain.ci_upper_95:.4f}] | {strict_rain.p_value:.3f} | {int(strict_rain.n_obs):,} |
| District-by-calendar-month FE + year FE, weekly hours | {hours.coefficient:.3f} | [{hours.ci_lower_95:.3f}, {hours.ci_upper_95:.3f}] | {hours.p_value:.3f} | {int(hours.n_obs):,} |
| Strict-FE future-month placebo, work participation | {future.coefficient:.4f} | [{future.ci_lower_95:.4f}, {future.ci_upper_95:.4f}] | {future.p_value:.3f} | {int(future.n_obs):,} |

## Current reading

The table reports both the strict local-seasonality estimate and its rainfall-controlled
counterpart. These are month-level associations; coefficient signs and uncertainty must
be interpreted together with the future-month placebo and outcome-coverage diagnostics.

## Required before final interpretation

- Retain a month-level associational estimand: exact interview dates are unavailable,
  while work outcomes refer to the previous seven days.
- Add a complete 1991-2020 climatology before using local percentile or anomaly fields.
- Audit occupation and housing category concordances before mechanism estimates.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = load_analysis_data(connection, args.schema)
        summary = sample_summary(frame)
        estimates = pd.DataFrame(
            [
                result
                for spec in SPECS
                for result in fit_spec(frame, spec)
            ]
            + [
                result
                for spec in HDFE_SPECS
                for result in fit_hdfe_spec(frame, spec)
            ]
        )
        write_experiment_artifacts(output, estimates, summary)
        connection.rollback()

    print(f"input_rows={len(frame)}")
    print(f"estimate_rows={len(estimates)}")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
