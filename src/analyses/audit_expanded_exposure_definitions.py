#!/usr/bin/env python3
"""Run an expanded, read-only heat-exposure definition experiment.

The experiment compares absolute humid- and dry-heat day counts, stricter
threshold variants, and continuous mean maximum wet-bulb and air temperature.
It also audits whether local historical P95 variables are estimable. Database
access is SELECT-only; every output is a temporary local diagnostic.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
import seaborn as sns
from psycopg import sql

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_credibility_audit import (
    DEMOGRAPHIC_CONTROLS,
    connection_args,
    fetch_frame,
    fit_hdfe,
    load_analysis_data,
)


DEFAULT_OUTPUT = Path("data/exp/exposure-definition-audit/expanded")

EXPOSURES = {
    "wb26_5days": {
        "source": "days_wbmax_ge_26c",
        "label": "WBmax >= 26 C days",
        "short": "WB26 days",
        "family": "humid heat",
        "kind": "threshold count",
        "increment": 5.0,
        "unit": "days",
    },
    "wb28_5days": {
        "source": "days_wbmax_ge_28c",
        "label": "WBmax >= 28 C days",
        "short": "WB28 days",
        "family": "humid heat",
        "kind": "strict threshold count",
        "increment": 5.0,
        "unit": "days",
    },
    "wbmax_mean_1c": {
        "source": "wet_bulb_max_mean_c",
        "label": "Mean daily WBmax",
        "short": "Mean WBmax",
        "family": "humid heat",
        "kind": "continuous",
        "increment": 1.0,
        "unit": "C",
    },
    "tmax35_5days": {
        "source": "days_tmax_ge_35c",
        "label": "Tmax >= 35 C days",
        "short": "Tmax35 days",
        "family": "dry heat",
        "kind": "threshold count",
        "increment": 5.0,
        "unit": "days",
    },
    "tmax37_5days": {
        "source": "days_tmax_ge_37c",
        "label": "Tmax >= 37 C days",
        "short": "Tmax37 days",
        "family": "dry heat",
        "kind": "strict threshold count",
        "increment": 5.0,
        "unit": "days",
    },
    "tmax_mean_1c": {
        "source": "temperature_2m_max_mean_c",
        "label": "Mean daily Tmax",
        "short": "Mean Tmax",
        "family": "dry heat",
        "kind": "continuous",
        "increment": 1.0,
        "unit": "C",
    },
}


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


def load_local_percentile_availability(
    connection: psycopg.Connection, schema: str
) -> pd.DataFrame:
    query = sql.SQL(
        """
        WITH analytical_cells AS (
            SELECT DISTINCT admin2_code, survey_year, survey_month
            FROM {}.{}
            WHERE analysis_weight > 0
        ), matched AS (
            SELECT
                a.admin2_code,
                a.survey_year,
                a.survey_month,
                c.days_tmax_above_local_month_p95,
                c.days_wbmax_above_local_month_p95,
                c.temperature_2m_max_anomaly_c,
                c.wet_bulb_max_anomaly_c,
                c.climatology_year_count,
                c.climatology_complete
            FROM analytical_cells a
            LEFT JOIN {}.{} c
              ON c.admin2_code = a.admin2_code
             AND c.reference_year = a.survey_year
             AND c.calendar_month = a.survey_month
        )
        SELECT
            count(*)::bigint AS analytical_cells,
            count(days_tmax_above_local_month_p95)::bigint AS tmax_p95_observed,
            count(days_wbmax_above_local_month_p95)::bigint AS wbmax_p95_observed,
            count(temperature_2m_max_anomaly_c)::bigint AS tmax_anomaly_observed,
            count(wet_bulb_max_anomaly_c)::bigint AS wbmax_anomaly_observed,
            min(climatology_year_count)::integer AS minimum_climatology_years,
            max(climatology_year_count)::integer AS maximum_climatology_years,
            count(*) FILTER (WHERE climatology_complete)::bigint
                AS complete_climatology_cells
        FROM matched
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema),
        sql.Identifier("final_CLIMATE_MONTHLY_ADMIN2"),
    )
    audit = fetch_frame(connection, query)
    record = audit.iloc[0]
    n_cells = int(record["analytical_cells"])
    candidates = [
        (
            "days_tmax_above_local_month_p95",
            "Days above local calendar-month Tmax P95",
            int(record["tmax_p95_observed"]),
        ),
        (
            "days_wbmax_above_local_month_p95",
            "Days above local calendar-month WBmax P95",
            int(record["wbmax_p95_observed"]),
        ),
        (
            "temperature_2m_max_anomaly_c",
            "Mean Tmax anomaly from local calendar-month climatology",
            int(record["tmax_anomaly_observed"]),
        ),
        (
            "wet_bulb_max_anomaly_c",
            "Mean WBmax anomaly from local calendar-month climatology",
            int(record["wbmax_anomaly_observed"]),
        ),
    ]
    rows = []
    for variable, label, observed in candidates:
        rows.append(
            {
                "candidate_variable": variable,
                "definition": label,
                "analytical_cells": n_cells,
                "observed_cells": observed,
                "missing_rate": 1.0 - observed / n_cells if n_cells else np.nan,
                "minimum_climatology_years": int(record["minimum_climatology_years"]),
                "maximum_climatology_years": int(record["maximum_climatology_years"]),
                "complete_climatology_cells": int(record["complete_climatology_cells"]),
                "required_climatology_years": 25,
                "estimation_status": (
                    "estimable" if observed == n_cells else "not estimable"
                ),
                "reason": (
                    "Complete 1991-2020-style climatology is unavailable; the database "
                    "requires at least 25 baseline years before publishing this field."
                    if observed < n_cells
                    else "Available for every analytical cell."
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wbmax_mean_1c"] = frame["wet_bulb_max_mean_c"]
    frame["tmax_mean_1c"] = frame["temperature_2m_max_mean_c"]
    return frame


def make_cells(frame: pd.DataFrame) -> pd.DataFrame:
    sources = [metadata["source"] for metadata in EXPOSURES.values()]
    columns = [
        "survey_wave",
        "survey_year",
        "survey_month",
        "admin2_code",
        "admin1_code",
        *sources,
    ]
    cells = frame[columns].drop_duplicates(
        ["survey_year", "survey_month", "admin2_code"]
    ).reset_index(drop=True)
    cells["admin2_calendar_month"] = (
        cells["admin2_code"].astype("string")
        + "-m"
        + cells["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    return cells


def summarize_support(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual_data = cells[
        ["survey_wave", "survey_year", "survey_month", "admin2_code"]
    ].copy()
    rows: list[dict[str, object]] = []
    for term, metadata in EXPOSURES.items():
        source = metadata["source"]
        values = cells[source]
        residuals = residualize_exposure(cells, source)
        residual_data[source] = values
        residual_data[f"residual_{source}"] = residuals
        observed = values.dropna()
        residual_observed = residuals.dropna()
        raw_sd = float(observed.std(ddof=0))
        residual_sd = float(residual_observed.std(ddof=0))
        rows.append(
            {
                "term": term,
                "exposure_definition": metadata["label"],
                "family": metadata["family"],
                "kind": metadata["kind"],
                "native_unit": metadata["unit"],
                "model_increment_native_units": metadata["increment"],
                "n_cells": int(len(cells)),
                "observed_cells": int(observed.size),
                "missing_rate": float(values.isna().mean()),
                "mean_native_units": float(observed.mean()),
                "raw_sd_native_units": raw_sd,
                "p25_native_units": float(observed.quantile(0.25)),
                "median_native_units": float(observed.median()),
                "p75_native_units": float(observed.quantile(0.75)),
                "p95_native_units": float(observed.quantile(0.95)),
                "zero_share": (
                    float(observed.eq(0).mean())
                    if "threshold" in metadata["kind"]
                    else np.nan
                ),
                "residual_sd_native_units": residual_sd,
                "residual_to_raw_sd_ratio": residual_sd / raw_sd if raw_sd else np.nan,
                "residual_sd_in_model_units": residual_sd / metadata["increment"],
            }
        )
    return pd.DataFrame(rows), residual_data


def run_models(frame: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    terms = list(EXPOSURES)
    complete_exposures = frame[terms].notna().all(axis=1)
    rain_controls = [*DEMOGRAPHIC_CONTROLS, "precipitation_100mm"]
    specifications = [
        (
            "primary_work_rainfall",
            "Work participation: demographics + rainfall",
            "worked_past_week",
            rain_controls,
            frame["worked_past_week"].notna() & complete_exposures,
        ),
        (
            "work_no_rainfall",
            "Work participation: demographics only",
            "worked_past_week",
            DEMOGRAPHIC_CONTROLS,
            frame["worked_past_week"].notna() & complete_exposures,
        ),
        (
            "work_no_demographics",
            "Work participation: rainfall only",
            "worked_past_week",
            ["precipitation_100mm"],
            frame["worked_past_week"].notna() & complete_exposures,
        ),
        (
            "weekly_hours_rainfall",
            "Weekly hours: demographics + rainfall",
            "weekly_hours_including_zero",
            rain_controls,
            frame["weekly_hours_including_zero"].notna() & complete_exposures,
        ),
        (
            "log_wage_rainfall",
            "Log wage: demographics + rainfall",
            "log_monthly_salary_wages",
            rain_controls,
            frame["log_monthly_salary_wages"].notna() & complete_exposures,
        ),
    ]
    for family, label, outcome, controls, mask in specifications:
        for term in terms:
            result, _model = fit_hdfe(
                frame,
                f"expanded_{family}_{term}",
                outcome,
                [term],
                controls,
                sample_label="common sample across six estimable exposure definitions",
                mask=mask,
            )
            record = result[0]
            record["model_family"] = family
            record["specification_label"] = label
            record["exposure_definition"] = EXPOSURES[term]["label"]
            rows.append(record)

    joint_specs = [
        (
            "joint_absolute_thresholds",
            "Work participation: WB26 and Tmax35 jointly",
            ["wb26_5days", "tmax35_5days"],
        ),
        (
            "joint_continuous_means",
            "Work participation: mean WBmax and mean Tmax jointly",
            ["wbmax_mean_1c", "tmax_mean_1c"],
        ),
        (
            "joint_humid_threshold_continuous",
            "Work participation: WB26 and mean WBmax jointly",
            ["wb26_5days", "wbmax_mean_1c"],
        ),
        (
            "joint_dry_threshold_continuous",
            "Work participation: Tmax35 and mean Tmax jointly",
            ["tmax35_5days", "tmax_mean_1c"],
        ),
    ]
    work_mask = frame["worked_past_week"].notna() & complete_exposures
    for family, label, joint_terms in joint_specs:
        result, _model = fit_hdfe(
            frame,
            f"expanded_{family}",
            "worked_past_week",
            joint_terms,
            rain_controls,
            sample_label="common sample; listed exposures entered jointly",
            mask=work_mask,
        )
        for record in result:
            record["model_family"] = family
            record["specification_label"] = label
            record["exposure_definition"] = EXPOSURES[record["term"]]["label"]
            rows.append(record)

    models = pd.DataFrame(rows)
    support_lookup = support.set_index("term")
    models["native_increment"] = models["term"].map(
        {term: metadata["increment"] for term, metadata in EXPOSURES.items()}
    )
    models["native_unit"] = models["term"].map(
        {term: metadata["unit"] for term, metadata in EXPOSURES.items()}
    )
    models["residual_sd_in_model_units"] = models["term"].map(
        support_lookup["residual_sd_in_model_units"]
    )
    outcome_scale = np.where(models["outcome"].eq("worked_past_week"), 100.0, 1.0)
    models["effect_display"] = models["coefficient"] * outcome_scale
    models["ci_lower_display"] = models["ci_lower_95"] * outcome_scale
    models["ci_upper_display"] = models["ci_upper_95"] * outcome_scale
    models["standardized_effect_display"] = (
        models["effect_display"] * models["residual_sd_in_model_units"]
    )
    models["standardized_ci_lower_display"] = (
        models["ci_lower_display"] * models["residual_sd_in_model_units"]
    )
    models["standardized_ci_upper_display"] = (
        models["ci_upper_display"] * models["residual_sd_in_model_units"]
    )
    models["display_unit"] = np.select(
        [
            models["outcome"].eq("worked_past_week"),
            models["outcome"].eq("weekly_hours_including_zero"),
            models["outcome"].eq("log_monthly_salary_wages"),
        ],
        ["percentage points", "weekly hours", "log points"],
        default="outcome units",
    )
    first_columns = [
        "model_family",
        "specification_label",
        "outcome",
        "term",
        "exposure_definition",
        "native_increment",
        "native_unit",
        "coefficient",
        "standard_error",
        "ci_lower_95",
        "ci_upper_95",
        "p_value",
        "effect_display",
        "ci_lower_display",
        "ci_upper_display",
        "residual_sd_in_model_units",
        "standardized_effect_display",
        "standardized_ci_lower_display",
        "standardized_ci_upper_display",
        "display_unit",
    ]
    remaining = [column for column in models.columns if column not in first_columns]
    return models[first_columns + remaining]


def make_primary_summary(
    models: pd.DataFrame, support: pd.DataFrame
) -> pd.DataFrame:
    primary = models.loc[models["model_family"].eq("primary_work_rainfall")].copy()
    primary = primary.merge(
        support[
            [
                "term",
                "family",
                "kind",
                "observed_cells",
                "zero_share",
                "residual_to_raw_sd_ratio",
            ]
        ],
        on="term",
        how="left",
        validate="one_to_one",
    )
    primary["direction"] = np.where(
        primary["standardized_effect_display"].lt(0), "adverse", "non-adverse"
    )
    primary["interval_excludes_zero"] = (
        primary["standardized_ci_upper_display"].lt(0)
        | primary["standardized_ci_lower_display"].gt(0)
    )
    columns = [
        "term",
        "exposure_definition",
        "family",
        "kind",
        "observed_cells",
        "zero_share",
        "residual_to_raw_sd_ratio",
        "standardized_effect_display",
        "standardized_ci_lower_display",
        "standardized_ci_upper_display",
        "p_value",
        "direction",
        "interval_excludes_zero",
        "n_obs",
        "n_admin2",
    ]
    return primary[columns]


def create_comparison_figure(
    support: pd.DataFrame,
    models: pd.DataFrame,
    output: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    order = list(EXPOSURES)
    labels = [EXPOSURES[term]["short"] for term in order]
    colors = [
        "#1D6F7A" if EXPOSURES[term]["family"] == "humid heat" else "#C66A2B"
        for term in order
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    support_plot = support.set_index("term").loc[order]
    axes[0].barh(
        np.arange(len(order)),
        support_plot["residual_to_raw_sd_ratio"],
        color=colors,
        alpha=0.9,
    )
    axes[0].set_yticks(np.arange(len(order)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Residual SD / raw SD")
    axes[0].set_ylabel("")
    axes[0].set_xlim(0, max(0.65, support_plot["residual_to_raw_sd_ratio"].max() * 1.15))
    axes[0].text(-0.12, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")

    primary = models.loc[models["model_family"].eq("primary_work_rainfall")]
    primary = primary.set_index("term").loc[order]
    y = np.arange(len(order))
    estimate = primary["standardized_effect_display"].to_numpy()
    lower = primary["standardized_ci_lower_display"].to_numpy()
    upper = primary["standardized_ci_upper_display"].to_numpy()
    for index, term in enumerate(order):
        axes[1].errorbar(
            estimate[index],
            y[index],
            xerr=np.array(
                [[estimate[index] - lower[index]], [upper[index] - estimate[index]]]
            ),
            fmt="o",
            capsize=3,
            color=colors[index],
        )
    axes[1].axvline(0, color="#4D4D4D", linewidth=1, linestyle="--")
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Work participation effect (pp / 1 within-FE SD)")
    axes[1].set_ylabel("")
    axes[1].text(-0.12, 1.03, "b", transform=axes[1].transAxes, fontweight="bold")

    forest = models.loc[
        models["model_family"].isin(
            ["primary_work_rainfall", "work_no_rainfall", "work_no_demographics"]
        )
    ].copy()
    spec_order = [
        "primary_work_rainfall",
        "work_no_rainfall",
        "work_no_demographics",
    ]
    spec_labels = ["Demographics + rainfall", "Demographics only", "Rainfall only"]
    spec_colors = ["#17324D", "#668FA3", "#9AABB3"]
    offsets = [0.20, 0.0, -0.20]
    for spec, spec_label, color, offset in zip(
        spec_order, spec_labels, spec_colors, offsets, strict=True
    ):
        subset = forest.loc[forest["model_family"].eq(spec)].set_index("term").loc[order]
        estimates = subset["standardized_effect_display"].to_numpy()
        lowers = subset["standardized_ci_lower_display"].to_numpy()
        uppers = subset["standardized_ci_upper_display"].to_numpy()
        axes[2].errorbar(
            estimates,
            y + offset,
            xerr=np.vstack([estimates - lowers, uppers - estimates]),
            fmt="o",
            capsize=2,
            color=color,
            label=spec_label,
        )
    axes[2].axvline(0, color="#4D4D4D", linewidth=1, linestyle="--")
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Work participation effect (pp / 1 within-FE SD)")
    axes[2].set_ylabel("")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")
    axes[2].text(-0.12, 1.03, "c", transform=axes[2].transAxes, fontweight="bold")

    fig.text(
        0.5,
        0.01,
        "Teal = humid-heat definitions; orange = dry-heat definitions. "
        "P95 historical thresholds are excluded because the required climatology is incomplete.",
        ha="center",
        fontsize=9,
        color="#5F6B73",
    )
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_summary_table_png(
    summary: pd.DataFrame,
    availability: pd.DataFrame,
    output: Path,
) -> None:
    display = summary.copy()
    display["Definition"] = display["exposure_definition"]
    display["Type"] = display["kind"].str.replace(" count", "", regex=False).str.title()
    display["Residual / Raw SD"] = display["residual_to_raw_sd_ratio"].map(
        lambda value: f"{value:.3f}"
    )
    display["Effect (pp / SD)"] = display["standardized_effect_display"].map(
        lambda value: f"{value:.3f}"
    )
    display["95% CI"] = display.apply(
        lambda row: (
            f"[{row['standardized_ci_lower_display']:.3f}, "
            f"{row['standardized_ci_upper_display']:.3f}]"
        ),
        axis=1,
    )
    display["p-value"] = display["p_value"].map(lambda value: f"{value:.3f}")
    display["Direction"] = display["direction"].str.title()
    display = display[
        [
            "Definition",
            "Type",
            "Residual / Raw SD",
            "Effect (pp / SD)",
            "95% CI",
            "p-value",
            "Direction",
        ]
    ]

    fig, axis = plt.subplots(figsize=(15.5, 5.0))
    axis.axis("off")
    axis.add_patch(
        plt.Rectangle((0, 0.89), 1, 0.11, transform=axis.transAxes, color="#17324D")
    )
    axis.text(
        0.012,
        0.945,
        "Expanded Heat-Exposure Definition Experiment",
        transform=axis.transAxes,
        va="center",
        ha="left",
        color="white",
        fontsize=16,
        fontweight="bold",
    )
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.25, 1, 0.64],
        colWidths=[0.22, 0.15, 0.13, 0.14, 0.15, 0.08, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#C9D4DA")
        if row == 0:
            cell.set_facecolor("#DCEBEA")
            cell.set_text_props(weight="bold", color="#243746")
        else:
            cell.set_facecolor("#EEF4F7" if row % 2 == 0 else "white")
            if column in [0, 1, 6]:
                cell.get_text().set_ha("left")
    unavailable = int(availability["estimation_status"].eq("not estimable").sum())
    notes = [
        "Notes: Effect is the work-participation change in percentage points per one fixed-effect residual standard deviation of exposure.",
        f"Local historical P95/anomaly candidates unavailable: {unavailable}/4; baseline coverage is 1-8 years versus the required 25 years.",
        "Models use a common sample, positive analysis weights, district-by-calendar-month and survey-year fixed effects, rainfall and demographic controls, and district-clustered inference.",
        "Database access was read-only SELECT; database writes: none. This remains a temporary diagnostic and does not revise AnaSOP.",
    ]
    for index, note in enumerate(notes):
        axis.text(
            0.01,
            0.19 - index * 0.045,
            note,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#5F6B73",
        )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate(
    availability: pd.DataFrame,
    support: pd.DataFrame,
    models: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, object]:
    if len(support) != 6 or support["term"].nunique() != 6:
        raise AssertionError("Expected six estimable exposure definitions")
    if not support["n_cells"].eq(3_325).all():
        raise AssertionError("Unexpected district-month cell count")
    if not availability["observed_cells"].eq(0).all():
        raise AssertionError("Historical-climatology fields unexpectedly contain values")
    primary = models.loc[models["model_family"].eq("primary_work_rainfall")]
    if len(primary) != 6 or primary["n_obs"].nunique() != 1:
        raise AssertionError("Primary exposure models do not use one common sample")
    for family in [
        "primary_work_rainfall",
        "work_no_rainfall",
        "work_no_demographics",
        "weekly_hours_rainfall",
        "log_wage_rainfall",
    ]:
        family_rows = models.loc[models["model_family"].eq(family)]
        if len(family_rows) != 6 or family_rows["n_obs"].nunique() != 1:
            raise AssertionError(f"Common sample contract failed for {family}")
    if summary["term"].nunique() != 6:
        raise AssertionError("Primary summary is incomplete")
    return {
        "district_month_cells": int(support.iloc[0]["n_cells"]),
        "estimable_exposure_definitions": 6,
        "unavailable_historical_candidates": int(
            availability["estimation_status"].eq("not estimable").sum()
        ),
        "model_rows": int(len(models)),
        "primary_common_sample_n": int(primary.iloc[0]["n_obs"]),
        "admin2_clusters": int(primary.iloc[0]["n_admin2"]),
    }


def write_readme(
    output: Path,
    availability: pd.DataFrame,
    support: pd.DataFrame,
    models: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    ordered = summary.set_index("term").loc[list(EXPOSURES)]
    lines = [
        "# Expanded heat-exposure definition experiment",
        "",
        "Status: temporary diagnostic; not a formal AnaSOP output.",
        "",
        "## Scope",
        "",
        "The experiment compares six estimable definitions under a common sample and model contract: WBmax >= 26 C days, WBmax >= 28 C days, mean daily WBmax, Tmax >= 35 C days, Tmax >= 37 C days, and mean daily Tmax.",
        "",
        "Historical local P95 and climatological anomaly variables cannot currently be estimated. All analytical cells are missing because district-calendar-month baseline coverage is only 1-8 years, below the database's 25-year completeness requirement. No short-baseline substitute was relabeled as historical climatology.",
        "",
        "## Primary work-participation screening",
        "",
        "Effects below are percentage points per one fixed-effect residual standard deviation of exposure:",
        "",
    ]
    for term, row in ordered.iterrows():
        lines.append(
            f"- {EXPOSURES[term]['label']}: {row['standardized_effect_display']:.3f} "
            f"(95% CI {row['standardized_ci_lower_display']:.3f} to "
            f"{row['standardized_ci_upper_display']:.3f}; p={row['p_value']:.3f}; "
            f"direction={row['direction']})."
        )
    lines.extend(
        [
            "",
            "## Diagnostic recommendation",
            "",
            "Retain WBmax >= 26 C days as the primary interpretable humid-heat exposure and use mean daily WBmax as its continuous robustness measure. These two humid-heat definitions have similar adverse standardized work-participation estimates, whereas all three dry-heat definitions are non-adverse in their separate primary models. In the joint continuous model, mean WBmax remains adverse and becomes larger in magnitude while mean Tmax is positive, reinforcing the distinction between humid heat and dry heat.",
            "",
            "Do not promote WBmax >= 28 C or Tmax >= 37 C to primary status: their zero shares are 98.9% and 87.4%, respectively. Retain Tmax >= 35 C as the literature-aligned dry-heat comparison. Wage estimates vary across humid-heat parameterizations, so wage should not yet carry the core narrative.",
            "",
            "The screen does not select a primary definition mechanically by p-value. A final choice should combine substantive interpretation, usable within-fixed-effect variation, direction and magnitude stability, and temporal/placebo behavior.",
            "",
            "## Design contract",
            "",
            "Every paired model uses the same observations, positive analysis weights, district-by-calendar-month and survey-year fixed effects, and debiased district-clustered standard errors. The primary specification controls for age, age squared, female, rural residence, and monthly precipitation.",
            "",
            "Database access was read-only SELECT; database writes: none.",
            "",
            "## Files",
            "",
            "- `Expanded_exposure_experiment.png`: identification support and standardized work-effect comparison.",
            "- `Expanded_exposure_experiment_summary.png`: rendered summary table for review.",
            "- `candidate_availability.csv`: local climatology availability audit.",
            "- `exposure_support.csv`: raw and within-fixed-effect support statistics.",
            "- `model_comparison.csv`: all separate and joint estimates.",
            "- `primary_standardized_effects.csv`: concise primary-model comparison.",
            "- `cell_exposure_residuals.csv`: district-month exposure and residual data.",
            "- `audit_manifest.json`: validation and provenance summary.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(**connection_args(args)) as connection:
        availability = load_local_percentile_availability(connection, args.schema)
        frame = load_analysis_data(connection, args.schema)
        connection.rollback()
    frame = prepare_frame(frame)
    cells = make_cells(frame)
    support, residuals = summarize_support(cells)
    models = run_models(frame, support)
    summary = make_primary_summary(models, support)

    availability.to_csv(output / "candidate_availability.csv", index=False)
    support.to_csv(output / "exposure_support.csv", index=False)
    residuals.to_csv(output / "cell_exposure_residuals.csv", index=False)
    models.to_csv(output / "model_comparison.csv", index=False)
    summary.to_csv(output / "primary_standardized_effects.csv", index=False)
    create_comparison_figure(
        support, models, output / "Expanded_exposure_experiment.png"
    )
    create_summary_table_png(
        summary, availability, output / "Expanded_exposure_experiment_summary.png"
    )
    validation = validate(availability, support, models, summary)
    write_readme(output, availability, support, models, summary)
    expected_outputs = [
        "Expanded_exposure_experiment.png",
        "Expanded_exposure_experiment_summary.png",
        "README.md",
        "audit_manifest.json",
        "candidate_availability.csv",
        "cell_exposure_residuals.csv",
        "exposure_support.csv",
        "model_comparison.csv",
        "primary_standardized_effects.csv",
    ]
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "temporary diagnostic; not a formal AnaSOP output",
        "database_access": "read-only SELECT; writes none",
        "source_tables": [
            f"{args.dbname}.{args.schema}.final_HEAT_LABOR_ANALYTIC",
            f"{args.dbname}.{args.schema}.final_CLIMATE_MONTHLY_ADMIN2",
        ],
        "code": "src/analyses/audit_expanded_exposure_definitions.py",
        "validation": validation,
        "outputs": expected_outputs,
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
