#!/usr/bin/env python3
"""Cumulative Estimate Credibility Checks.

Plan: Show cumulative humid-versus-dry heat specificity and future placebos,
leave-one-wave-out stability, and frozen sample, control, geography, and
clustering sensitivities.

Framework: AnaSOP Section 5 requires timing, specificity, influential-wave,
geography, and inference diagnostics. Section 6 freezes two cumulative humid-
heat definitions and matched two-month dry-heat comparisons. Section 7 Steps
6-7 use these checks to determine whether the cumulative association is stable
and more coherent for humid heat. All PostgreSQL inputs are queried under a
forced read-only session.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mj02b-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg
import seaborn as sns
from matplotlib.lines import Line2D
from psycopg import sql

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_credibility_audit import fit_hdfe


DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_cumulative_estimate_credibility_checks.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_cumulative_estimate_credibility_checks.png"
)
DEFAULT_ESTIMATES_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_cumulative_estimate_credibility_checks_estimates.csv"
)

NAVY = "#17324D"
TEAL = "#3C7F7B"
GOLD = "#D39A35"
SLATE = "#7B8991"
RUST = "#A85C3A"
GRID = "#D7E0E4"
TEXT = "#23313D"
MUTED = "#5C6B73"

DEMOGRAPHIC_CONTROLS = ["age", "age_squared", "female", "rural"]
BASE_CONTROLS = [*DEMOGRAPHIC_CONTROLS, "precipitation_100mm"]

HUMID_EXPOSURES = {
    "threshold": {
        "term": "wb26_current_lag_total_5days",
        "label": "Cumulative WB26 days",
    },
    "continuous": {
        "term": "wbmean_current_lag_average_1c",
        "label": "Two-month mean WBmax",
    },
}

SPECIFICITY_ROWS = [
    {
        "row_id": "humid_threshold",
        "label": "Cumulative WB26 days",
        "term": "wb26_current_lag_total_5days",
        "category": "humid",
        "marker": "o",
    },
    {
        "row_id": "future_threshold",
        "label": "Future WB26 days (placebo)",
        "term": "lead_wb26_5days",
        "category": "placebo",
        "marker": "o",
    },
    {
        "row_id": "dry_threshold",
        "label": "Cumulative Tmax35 days",
        "term": "tmax35_current_lag_total_5days",
        "category": "dry",
        "marker": "o",
    },
    {
        "row_id": "humid_continuous",
        "label": "Two-month mean WBmax",
        "term": "wbmean_current_lag_average_1c",
        "category": "humid",
        "marker": "D",
    },
    {
        "row_id": "future_continuous",
        "label": "Future mean WBmax (placebo)",
        "term": "lead_wbmean_1c",
        "category": "placebo",
        "marker": "D",
    },
    {
        "row_id": "dry_continuous",
        "label": "Two-month mean Tmax",
        "term": "tmaxmean_current_lag_average_1c",
        "category": "dry",
        "marker": "D",
    },
]

ROBUSTNESS_ROWS = [
    ("frozen_primary", "Frozen primary"),
    ("no_rainfall", "No rainfall control"),
    ("no_demographics", "No demographic controls"),
    ("balanced_districts", "Districts in all nine waves"),
    ("exclude_fallback", "Exclude coastal fallback"),
    ("province_clustering", "Province-clustered SE"),
    ("exclude_timing_risk", "Exclude timing-risk waves"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="mda")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument(
        "--estimates-output", type=Path, default=DEFAULT_ESTIMATES_OUTPUT
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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
            a.person_id, a.admin2_code, d.admin1_code,
            a.sex, a.age, a.urban_rural, a.analysis_weight,
            a.worked_past_week,
            a.days_wbmax_ge_26c, a.lag_days_wbmax_ge_26c,
            a.lead_days_wbmax_ge_26c,
            a.wet_bulb_max_mean_c, a.lag_wet_bulb_max_mean_c,
            a.lead_wet_bulb_max_mean_c,
            a.days_tmax_ge_35c, a.temperature_2m_max_mean_c,
            lag_climate.days_tmax_ge_35c AS lag_days_tmax_ge_35c,
            lag_climate.temperature_2m_max_mean_c
                AS lag_temperature_2m_max_mean_c,
            a.precipitation_month_sum_mm,
            a.temperature_location_fallback_used
        FROM {}.{} AS a
        LEFT JOIN {}.{} AS d USING (admin2_code)
        LEFT JOIN {}.{} AS lag_climate
          ON lag_climate.admin2_code = a.admin2_code
         AND make_date(lag_climate.reference_year, lag_climate.calendar_month, 1)
             = (make_date(a.survey_year, a.survey_month, 1)
                - interval '1 month')::date
        WHERE a.analysis_weight > 0
        ORDER BY a.survey_year, a.person_id
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier("final_HEAT_LABOR_ANALYTIC"),
        sql.Identifier(schema),
        sql.Identifier("dim_admin2_cambodia"),
        sql.Identifier(schema),
        sql.Identifier("final_CLIMATE_MONTHLY_ADMIN2"),
    )
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description]
        frame = pd.DataFrame(cursor.fetchall(), columns=columns)
    numeric = [
        "survey_year",
        "survey_month",
        "sex",
        "age",
        "urban_rural",
        "analysis_weight",
        "worked_past_week",
        "days_wbmax_ge_26c",
        "lag_days_wbmax_ge_26c",
        "lead_days_wbmax_ge_26c",
        "wet_bulb_max_mean_c",
        "lag_wet_bulb_max_mean_c",
        "lead_wet_bulb_max_mean_c",
        "days_tmax_ge_35c",
        "lag_days_tmax_ge_35c",
        "temperature_2m_max_mean_c",
        "lag_temperature_2m_max_mean_c",
        "precipitation_month_sum_mm",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["age_squared"] = frame["age"] ** 2
    frame["female"] = frame["sex"].eq(2).where(frame["sex"].isin([1, 2])).astype(float)
    frame["rural"] = frame["urban_rural"].eq(2).where(
        frame["urban_rural"].isin([1, 2])
    ).astype(float)
    frame["precipitation_100mm"] = frame["precipitation_month_sum_mm"] / 100.0
    frame["wb26_5days"] = frame["days_wbmax_ge_26c"] / 5.0
    frame["lead_wb26_5days"] = frame["lead_days_wbmax_ge_26c"] / 5.0
    frame["wb26_current_lag_total_5days"] = (
        frame["days_wbmax_ge_26c"] + frame["lag_days_wbmax_ge_26c"]
    ) / 5.0
    frame["wbmean_current_lag_average_1c"] = (
        frame["wet_bulb_max_mean_c"] + frame["lag_wet_bulb_max_mean_c"]
    ) / 2.0
    frame["lead_wbmean_1c"] = frame["lead_wet_bulb_max_mean_c"]
    frame["tmax35_current_lag_total_5days"] = (
        frame["days_tmax_ge_35c"] + frame["lag_days_tmax_ge_35c"]
    ) / 5.0
    frame["tmaxmean_current_lag_average_1c"] = (
        frame["temperature_2m_max_mean_c"]
        + frame["lag_temperature_2m_max_mean_c"]
    ) / 2.0
    frame["admin2_calendar_month"] = (
        frame["admin2_code"].astype("string")
        + "-m"
        + frame["survey_month"].astype("Int64").astype("string").str.zfill(2)
    )
    district_wave_count = frame.groupby("admin2_code")["survey_wave"].transform("nunique")
    frame["balanced_district"] = district_wave_count.eq(9)
    frame["fallback_excluded"] = ~frame[
        "temperature_location_fallback_used"
    ].fillna(False).astype(bool)
    return frame


def required_mask(
    frame: pd.DataFrame,
    exposures: list[str],
    controls: list[str] | None = None,
) -> pd.Series:
    controls = controls or BASE_CONTROLS
    required = [
        "worked_past_week",
        *exposures,
        *controls,
        "analysis_weight",
        "admin2_calendar_month",
        "survey_year",
        "admin2_code",
        "admin1_code",
    ]
    return frame[required].notna().all(axis=1) & frame["analysis_weight"].gt(0)


def reference_support(
    frame: pd.DataFrame,
    mask: pd.Series,
    terms: list[str],
) -> pd.Series:
    columns = [
        "survey_year",
        "survey_month",
        "admin2_code",
        "admin2_calendar_month",
        *terms,
    ]
    selected = frame.loc[mask, columns].copy()
    key = ["survey_year", "survey_month", "admin2_code"]
    for term in terms:
        within_cell_unique = selected.groupby(key, observed=True)[term].nunique(dropna=False)
        if int(within_cell_unique.max()) != 1:
            raise AssertionError(f"Exposure varies within district-year-month: {term}")
    cells = selected.drop_duplicates(key).reset_index(drop=True)
    support = {}
    for term in terms:
        support[term] = float(residualize_exposure(cells, term).dropna().std(ddof=0))
    return pd.Series(support, name="within_fe_sd")


def fit_one(
    frame: pd.DataFrame,
    *,
    panel: str,
    specification: str,
    specification_label: str,
    exposure_family: str,
    term: str,
    controls: list[str],
    mask: pd.Series,
    within_fe_sd: float,
    cluster_columns: list[str] | None = None,
) -> dict[str, object]:
    records, model = fit_hdfe(
        frame,
        f"credibility_{panel}_{specification}_{exposure_family}",
        "worked_past_week",
        [term],
        controls,
        sample_label=specification_label,
        cluster_columns=cluster_columns,
        mask=mask,
    )
    record = records[0]
    scale = 100.0 * within_fe_sd
    return {
        "panel": panel,
        "specification": specification,
        "specification_label": specification_label,
        "exposure_family": exposure_family,
        "term": term,
        "raw_effect_pp": 100.0 * float(record["coefficient"]),
        "raw_ci_lower_pp": 100.0 * float(record["ci_lower_95"]),
        "raw_ci_upper_pp": 100.0 * float(record["ci_upper_95"]),
        "within_fe_sd": within_fe_sd,
        "standardized_effect_pp": scale * float(record["coefficient"]),
        "standardized_ci_lower_pp": scale * float(record["ci_lower_95"]),
        "standardized_ci_upper_pp": scale * float(record["ci_upper_95"]),
        "p_value": float(record["p_value"]),
        "n_obs": int(model.nobs),
        "n_admin2": int(record["n_admin2"]),
        "n_provinces": int(record["n_provinces"]),
        "controls": record["controls"],
        "fixed_effects": record["fixed_effects"],
        "cluster_level": record["cluster_level"],
    }


def estimate_specificity(frame: pd.DataFrame) -> list[dict[str, object]]:
    terms = [row["term"] for row in SPECIFICITY_ROWS]
    mask = required_mask(frame, terms)
    support = reference_support(frame, mask, terms)
    rows = []
    for metadata in SPECIFICITY_ROWS:
        rows.append(
            {
                **fit_one(
                    frame,
                    panel="specificity",
                    specification=metadata["row_id"],
                    specification_label=metadata["label"],
                    exposure_family=metadata["category"],
                    term=metadata["term"],
                    controls=BASE_CONTROLS,
                    mask=mask,
                    within_fe_sd=float(support[metadata["term"]]),
                ),
                "row_label": metadata["label"],
                "marker": metadata["marker"],
                "category": metadata["category"],
            }
        )
    return rows


def estimate_leave_one_wave_out(frame: pd.DataFrame) -> list[dict[str, object]]:
    terms = [metadata["term"] for metadata in HUMID_EXPOSURES.values()]
    base_mask = required_mask(frame, terms)
    support = reference_support(frame, base_mask, terms)
    wave_order = (
        frame.loc[base_mask, ["survey_wave", "survey_year"]]
        .drop_duplicates()
        .sort_values(["survey_year", "survey_wave"])["survey_wave"]
        .tolist()
    )
    rows = []
    for wave in wave_order:
        mask = base_mask & frame["survey_wave"].ne(wave)
        for family, metadata in HUMID_EXPOSURES.items():
            rows.append(
                {
                    **fit_one(
                        frame,
                        panel="leave_one_wave_out",
                        specification=f"exclude_{wave}",
                        specification_label=f"Exclude {wave}",
                        exposure_family=family,
                        term=metadata["term"],
                        controls=BASE_CONTROLS,
                        mask=mask,
                        within_fe_sd=float(support[metadata["term"]]),
                    ),
                    "row_label": f"Exclude {wave}",
                    "omitted_wave": wave,
                }
            )
    return rows


def estimate_robustness(frame: pd.DataFrame) -> list[dict[str, object]]:
    terms = [metadata["term"] for metadata in HUMID_EXPOSURES.values()]
    base_mask = required_mask(frame, terms)
    support = reference_support(frame, base_mask, terms)
    specifications = {
        "frozen_primary": {
            "mask": base_mask,
            "controls": BASE_CONTROLS,
            "clusters": None,
        },
        "no_rainfall": {
            "mask": base_mask,
            "controls": DEMOGRAPHIC_CONTROLS,
            "clusters": None,
        },
        "no_demographics": {
            "mask": base_mask,
            "controls": ["precipitation_100mm"],
            "clusters": None,
        },
        "balanced_districts": {
            "mask": base_mask & frame["balanced_district"],
            "controls": BASE_CONTROLS,
            "clusters": None,
        },
        "exclude_fallback": {
            "mask": base_mask & frame["fallback_excluded"],
            "controls": BASE_CONTROLS,
            "clusters": None,
        },
        "province_clustering": {
            "mask": base_mask,
            "controls": BASE_CONTROLS,
            "clusters": ["admin1_code"],
        },
        "exclude_timing_risk": {
            "mask": base_mask
            & ~frame["survey_wave"].isin(["2011-12", "2019", "2021"]),
            "controls": BASE_CONTROLS,
            "clusters": None,
        },
    }
    labels = dict(ROBUSTNESS_ROWS)
    rows = []
    for specification, _label in ROBUSTNESS_ROWS:
        settings = specifications[specification]
        for family, metadata in HUMID_EXPOSURES.items():
            rows.append(
                {
                    **fit_one(
                        frame,
                        panel="robustness",
                        specification=specification,
                        specification_label=labels[specification],
                        exposure_family=family,
                        term=metadata["term"],
                        controls=settings["controls"],
                        mask=settings["mask"],
                        within_fe_sd=float(support[metadata["term"]]),
                        cluster_columns=settings["clusters"],
                    ),
                    "row_label": labels[specification],
                }
            )
    return rows


def estimate_all(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        *estimate_specificity(frame),
        *estimate_leave_one_wave_out(frame),
        *estimate_robustness(frame),
    ]
    estimates = pd.DataFrame(rows)
    if estimates.empty:
        raise AssertionError("No credibility estimates were produced")
    return estimates


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT, labelsize=8.5)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    sns.despine(ax=axis, top=True, right=True)


def paired_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    row_order: list[str],
    row_labels: list[str],
) -> None:
    styles = {
        "threshold": {"color": TEAL, "marker": "o", "label": "Cumulative WB26 days"},
        "continuous": {"color": NAVY, "marker": "D", "label": "Two-month mean WBmax"},
    }
    y = np.arange(len(row_order))[::-1]
    for family, offset in [("threshold", 0.10), ("continuous", -0.10)]:
        subset = data.loc[data["exposure_family"].eq(family)].set_index(
            "specification"
        ).loc[row_order]
        estimate = subset["standardized_effect_pp"].to_numpy(float)
        lower = subset["standardized_ci_lower_pp"].to_numpy(float)
        upper = subset["standardized_ci_upper_pp"].to_numpy(float)
        axis.errorbar(
            estimate,
            y + offset,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt=styles[family]["marker"],
            color=styles[family]["color"],
            markerfacecolor=styles[family]["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.0,
            capsize=2.5,
            linewidth=1.25,
            label=styles[family]["label"],
            zorder=3,
        )
    axis.axvline(0, color=MUTED, linestyle="--", linewidth=1.0)
    bounds = data[["standardized_ci_lower_pp", "standardized_ci_upper_pp"]].to_numpy(
        float
    )
    lower_limit = min(-0.05, float(np.nanmin(bounds)))
    upper_limit = max(0.05, float(np.nanmax(bounds)))
    span = upper_limit - lower_limit
    axis.set_xlim(lower_limit - 0.08 * span, upper_limit + 0.08 * span)
    axis.set_yticks(y, row_labels)


def sample_label_for_pair(data: pd.DataFrame, specification: str) -> str:
    selected = data.loc[data["specification"].eq(specification)].set_index(
        "exposure_family"
    )
    threshold_n = int(selected.loc["threshold", "n_obs"])
    continuous_n = int(selected.loc["continuous", "n_obs"])
    if threshold_n == continuous_n:
        return f"N={threshold_n:,}"
    return f"N={threshold_n:,}/{continuous_n:,}"


def order_specifications(estimates: pd.DataFrame) -> pd.DataFrame:
    specificity_order = [
        "humid_threshold",
        "humid_continuous",
        "future_threshold",
        "future_continuous",
        "dry_threshold",
        "dry_continuous",
    ]
    parts: list[pd.DataFrame] = []
    specificity = estimates.loc[estimates["panel"].eq("specificity")].set_index(
        "specification",
        drop=False,
    )
    for specification in specificity_order:
        parts.append(specificity.loc[[specification]])

    leave_one_out = estimates.loc[
        estimates["panel"].eq("leave_one_wave_out")
    ]
    for specification in leave_one_out["specification"].drop_duplicates():
        pair = leave_one_out.loc[
            leave_one_out["specification"].eq(specification)
        ].set_index("exposure_family", drop=False)
        parts.extend([pair.loc[["threshold"]], pair.loc[["continuous"]]])

    robustness = estimates.loc[estimates["panel"].eq("robustness")]
    for specification, _label in ROBUSTNESS_ROWS:
        pair = robustness.loc[
            robustness["specification"].eq(specification)
        ].set_index("exposure_family", drop=False)
        parts.extend([pair.loc[["threshold"]], pair.loc[["continuous"]]])

    ordered = pd.concat(parts, ignore_index=True)
    if len(ordered) != len(estimates) or len(ordered) != 38:
        raise AssertionError("Specification curve must contain all 38 estimates")
    ordered["curve_index"] = np.arange(len(ordered))
    return ordered


def curve_style(row: pd.Series) -> dict[str, object]:
    if row["panel"] == "specificity":
        category = str(row["category"])
        marker = str(row["marker"])
        if category == "humid":
            color = TEAL if marker == "o" else NAVY
        elif category == "placebo":
            color = SLATE
        else:
            color = RUST
        facecolor = "white" if category == "placebo" else color
        return {
            "color": color,
            "marker": marker,
            "facecolor": facecolor,
        }
    family = str(row["exposure_family"])
    return {
        "color": TEAL if family == "threshold" else NAVY,
        "marker": "o" if family == "threshold" else "D",
        "facecolor": TEAL if family == "threshold" else NAVY,
    }


def attribute_keys(row: pd.Series) -> set[str]:
    keys: set[str] = set()
    if row["panel"] == "specificity":
        specification = str(row["specification"])
        if specification == "humid_threshold":
            keys.add("wb26")
        elif specification == "humid_continuous":
            keys.add("wbmax")
        elif specification.startswith("future_"):
            keys.add("future")
        elif specification.startswith("dry_"):
            keys.add("dry")
        keys.add("specificity")
        keys.add("specificity_sample")
    else:
        keys.add("wb26" if row["exposure_family"] == "threshold" else "wbmax")
        if row["panel"] == "leave_one_wave_out":
            keys.add("leave_one_wave_out")
        else:
            keys.add("robustness")
            mapping = {
                "frozen_primary": "frozen_primary",
                "no_rainfall": "no_rainfall",
                "no_demographics": "no_demographics",
                "balanced_districts": "balanced_districts",
                "exclude_fallback": "exclude_fallback",
                "province_clustering": "province_clustering",
                "exclude_timing_risk": "exclude_timing_risk",
            }
            keys.add(mapping[str(row["specification"])])
    return keys


def build_figure(estimates: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9.5,
            "text.color": TEXT,
            "axes.edgecolor": MUTED,
        }
    )
    ordered = order_specifications(estimates)
    figure = plt.figure(figsize=(18.0, 9.4))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.12, 1.28],
        left=0.15,
        right=0.99,
        bottom=0.13,
        top=0.95,
        hspace=0.07,
    )
    estimate_axis = figure.add_subplot(grid[0, 0])
    matrix_axis = figure.add_subplot(grid[1, 0], sharex=estimate_axis)

    block_spans = [
        (-0.5, 5.5, "#F8FAFA", "Specificity and placebo", 2.5),
        (5.5, 23.5, "#FFFFFF", "Leave-one-wave-out", 14.5),
        (23.5, 37.5, "#F8FAFA", "Frozen robustness", 30.5),
    ]
    for start, end, color, label, center in block_spans:
        estimate_axis.axvspan(start, end, color=color, zorder=0)
        matrix_axis.axvspan(start, end, color=color, zorder=0)
        estimate_axis.text(
            center,
            0.985,
            label,
            transform=estimate_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.6,
            color=MUTED,
            fontweight="bold",
        )
    for boundary in [5.5, 23.5]:
        estimate_axis.axvline(boundary, color=GRID, linewidth=1.0, zorder=1)
        matrix_axis.axvline(boundary, color=GRID, linewidth=1.0, zorder=1)

    for row in ordered.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        style = curve_style(row_series)
        estimate = float(row.standardized_effect_pp)
        lower = float(row.standardized_ci_lower_pp)
        upper = float(row.standardized_ci_upper_pp)
        estimate_axis.errorbar(
            int(row.curve_index),
            estimate,
            yerr=np.array([[estimate - lower], [upper - estimate]]),
            fmt=style["marker"],
            color=style["color"],
            markerfacecolor=style["facecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            markersize=5.7,
            capsize=2.2,
            linewidth=1.15,
            zorder=3,
        )
    estimate_axis.axhline(0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    estimate_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.85)
    estimate_axis.set_axisbelow(True)
    estimate_axis.set_ylabel(
        "Work participation difference\n(pp per reference within-FE SD)"
    )
    estimate_axis.set_xlim(-0.5, 37.5)
    estimate_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    estimate_axis.tick_params(axis="y", colors=TEXT, labelsize=8.5)
    sns.despine(ax=estimate_axis, top=True, right=True, bottom=True)
    estimate_axis.text(
        0.0,
        1.075,
        "a",
        transform=estimate_axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
    )
    estimate_axis.text(
        0.025,
        1.075,
        "All frozen estimates and 95% confidence intervals",
        transform=estimate_axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    cumulative_validation = ordered.loc[
        ordered["panel"].isin(["leave_one_wave_out", "robustness"])
    ]
    estimate_axis.text(
        0.985,
        0.94,
        (
            "Cumulative humid-heat validation\n"
            f"{int((cumulative_validation['standardized_effect_pp'] < 0).sum())}/"
            f"{len(cumulative_validation)} estimates adverse"
        ),
        transform=estimate_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color=NAVY,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": GRID,
            "boxstyle": "round,pad=0.34",
            "alpha": 0.95,
        },
    )

    attributes = [
        ("wb26", "Cumulative WB26 days", TEAL),
        ("wbmax", "Two-month mean WBmax", NAVY),
        ("future", "Future exposure placebo", SLATE),
        ("dry", "Dry-heat comparison", RUST),
        ("specificity_sample", "Common specificity sample", MUTED),
        ("specificity", "Specificity / placebo family", MUTED),
        ("leave_one_wave_out", "Leave-one-wave-out family", MUTED),
        ("robustness", "Robustness family", MUTED),
        ("frozen_primary", "Frozen primary", GOLD),
        ("no_rainfall", "No rainfall control", GOLD),
        ("no_demographics", "No demographic controls", GOLD),
        ("balanced_districts", "Districts in all nine waves", GOLD),
        ("exclude_fallback", "Exclude coastal fallback", GOLD),
        ("province_clustering", "Province-clustered SE", GOLD),
        ("exclude_timing_risk", "Exclude timing-risk waves", GOLD),
    ]
    y_positions = np.arange(len(attributes))[::-1]
    y_lookup = {key: y for (key, _label, _color), y in zip(attributes, y_positions)}
    for row in ordered.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        keys = attribute_keys(row_series)
        for key in keys:
            color = next(color for item_key, _label, color in attributes if item_key == key)
            matrix_axis.scatter(
                int(row.curve_index),
                y_lookup[key],
                s=25 if key in {"wb26", "wbmax", "future", "dry"} else 19,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
    matrix_axis.set_yticks(
        y_positions,
        [label for _key, label, _color in attributes],
    )
    matrix_axis.tick_params(axis="y", labelsize=7.7, colors=TEXT, length=0)
    matrix_axis.set_ylim(-1.8, len(attributes) - 0.35)
    matrix_axis.set_xlim(-0.5, 37.5)
    matrix_axis.set_xticks([])
    matrix_axis.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.55)
    matrix_axis.set_axisbelow(True)
    for y in [10.5, 6.5]:
        matrix_axis.axhline(y, color=GRID, linewidth=0.9, zorder=1)
    sns.despine(ax=matrix_axis, top=True, right=True, bottom=True)
    matrix_axis.text(
        0.0,
        1.03,
        "b",
        transform=matrix_axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
    )
    matrix_axis.text(
        0.025,
        1.03,
        "Specification attributes aligned to Panel A",
        transform=matrix_axis.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )

    specificity_centers = [(0.5, "Humid"), (2.5, "Future"), (4.5, "Dry")]
    for x, label in specificity_centers:
        matrix_axis.text(
            x,
            -0.78,
            label,
            ha="center",
            va="top",
            fontsize=6.8,
            color=MUTED,
            fontweight="bold",
        )
    loo = ordered.loc[ordered["panel"].eq("leave_one_wave_out")]
    for specification in loo["specification"].drop_duplicates():
        indices = loo.loc[loo["specification"].eq(specification), "curve_index"]
        label = specification.removeprefix("exclude_").replace("2011-12", "11-12")
        matrix_axis.text(
            float(indices.mean()),
            -0.78,
            label.replace("20", "", 1),
            ha="center",
            va="top",
            fontsize=6.5,
            color=MUTED,
            fontweight="bold",
        )
    robustness = ordered.loc[ordered["panel"].eq("robustness")]
    short_labels = {
        "frozen_primary": "Primary",
        "no_rainfall": "No rain",
        "no_demographics": "No demo",
        "balanced_districts": "Balanced",
        "exclude_fallback": "No coast",
        "province_clustering": "Province",
        "exclude_timing_risk": "Timing",
    }
    for specification, label in short_labels.items():
        indices = robustness.loc[
            robustness["specification"].eq(specification),
            "curve_index",
        ]
        matrix_axis.text(
            float(indices.mean()),
            -0.78,
            label,
            ha="center",
            va="top",
            fontsize=6.2,
            color=MUTED,
            fontweight="bold",
            rotation=28,
        )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=TEAL,
                markerfacecolor=TEAL,
                markersize=5.8,
                linewidth=1.2,
                label="Cumulative WB26 days",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color=NAVY,
                markerfacecolor=NAVY,
                markersize=5.5,
                linewidth=1.2,
                label="Two-month mean WBmax",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=SLATE,
                markerfacecolor="white",
                markersize=5.8,
                linewidth=1.2,
                label="Future exposure placebo",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=RUST,
                markerfacecolor=RUST,
                markersize=5.8,
                linewidth=1.2,
                label="Dry-heat comparison",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
        fontsize=8.2,
        ncol=4,
        handletextpad=0.6,
        columnspacing=1.7,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate(estimates: pd.DataFrame, output: Path) -> None:
    specificity = estimates.loc[estimates["panel"].eq("specificity")]
    leave_one_out = estimates.loc[estimates["panel"].eq("leave_one_wave_out")]
    robustness = estimates.loc[estimates["panel"].eq("robustness")]
    if len(specificity) != 6 or specificity["n_obs"].nunique() != 1:
        raise AssertionError("Specificity panel must contain six common-sample estimates")
    if len(leave_one_out) != 18:
        raise AssertionError("Leave-one-wave-out panel must contain 18 estimates")
    if len(robustness) != 2 * len(ROBUSTNESS_ROWS):
        raise AssertionError("Robustness panel has an unexpected estimate count")
    for panel in [leave_one_out, robustness]:
        if panel.groupby("specification")["n_obs"].nunique().gt(1).any():
            raise AssertionError("Paired humid-heat definitions do not share N")
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError("Credibility figure is missing or empty")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    estimates_output = resolve_path(args.estimates_output)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = load_data(connection, args.schema)
        connection.rollback()
    frame = prepare_frame(frame)
    estimates = estimate_all(frame)
    build_figure(estimates, output)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    if review_output.resolve() != output.resolve():
        shutil.copy2(output, review_output)
    estimates_output.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(estimates_output, index=False)
    validate(estimates, output)
    print(f"estimated_models={len(estimates)}")
    for panel in ["specificity", "leave_one_wave_out", "robustness"]:
        subset = estimates.loc[estimates["panel"].eq(panel)]
        print(
            f"{panel}: models={len(subset)} "
            f"negative={int(subset['standardized_effect_pp'].lt(0).sum())}"
        )
    print(f"saved_figure={output.relative_to(ROOT)}")
    print(f"saved_review_copy={review_output.relative_to(ROOT)}")
    print(f"saved_estimates={estimates_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
