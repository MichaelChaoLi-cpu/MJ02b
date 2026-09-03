#!/usr/bin/env python3
"""Cumulative Humid Heat by Education.

Plan: Show higher- and low-education cumulative humid-heat slopes, leave-one-
wave-out interaction stability, and frozen interaction sensitivities under
threshold and continuous exposure definitions.

Framework: AnaSOP Section 6 defines the higher-education slope as the reference
coefficient and the low-education difference as the cumulative-heat interaction.
Section 7 Steps 5 and 7 require both group-specific slopes and interaction
stability under the two frozen cumulative exposure definitions. Multiplicity
values are read from the frozen 82-row audit rather than recomputed within the
figure. All PostgreSQL inputs are queried under a forced read-only session.
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
from scipy.stats import norm

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_credibility_audit import fit_hdfe
from run_heat_labor_extensions import BASE_CONTROLS, connection_args, load_data


DEFAULT_OUTPUT = (
    ROOT
    / "data/results/figures"
    / "Figure_cumulative_humid_heat_by_education.png"
)
DEFAULT_REVIEW_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_cumulative_humid_heat_by_education.png"
)
DEFAULT_ESTIMATES_OUTPUT = (
    ROOT
    / "data/exp/figure-table-review"
    / "Figure_cumulative_humid_heat_by_education_estimates.csv"
)
DEFAULT_MULTIPLICITY_AUDIT = ROOT / "data/exp/storyline-search/target_tests.csv"

NAVY = "#17324D"
TEAL = "#3C7F7B"
GOLD = "#D39A35"
GRID = "#D7E0E4"
TEXT = "#23313D"
MUTED = "#5C6B73"

ROWS = [
    ("higher_education", "Lower-secondary or above"),
    ("low_education", "None, preschool, or primary"),
    ("interaction", "Difference: low minus higher"),
]

EXPOSURES = {
    "threshold": {
        "term": "wb26_current_lag_total_5days",
        "label": "Two-month cumulative WB26 days",
        "x_label": "Work participation difference (percentage points)\nper within-FE SD of cumulative WB26 days",
    },
    "continuous": {
        "term": "wbmean_current_lag_average_1c",
        "label": "Two-month mean maximum wet-bulb temperature",
        "x_label": "Work participation difference (percentage points)\nper within-FE SD of two-month mean WBmax",
    },
}

SENSITIVITY_ROWS = [
    ("frozen_common", "Frozen common sample"),
    ("definition_specific", "Definition-specific sample"),
    ("exclude_timing_risk", "Exclude timing-risk waves"),
    ("balanced_districts", "Districts in all nine waves"),
    ("exclude_fallback", "Exclude coastal fallback"),
    ("province_clustering", "Province-clustered SE"),
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
    parser.add_argument(
        "--multiplicity-audit", type=Path, default=DEFAULT_MULTIPLICITY_AUDIT
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wb26_current_lag_total_5days"] = (
        frame["wb26_5days"] + frame["lag_wb26_5days"]
    )
    frame["wbmean_current_lag_average_1c"] = (
        frame["wbmean_1c"] + frame["lag_wbmean_1c"]
    ) / 2.0
    district_wave_count = frame.groupby("admin2_code")["survey_wave"].transform(
        "nunique"
    )
    frame["balanced_district"] = district_wave_count.eq(9)
    frame["fallback_excluded"] = ~frame[
        "temperature_location_fallback_used"
    ].fillna(False).astype(bool)
    return frame


def common_sample_mask(frame: pd.DataFrame) -> pd.Series:
    exposure_terms = [metadata["term"] for metadata in EXPOSURES.values()]
    required = [
        "worked_past_week",
        "low_education",
        *exposure_terms,
        *BASE_CONTROLS,
        "analysis_weight",
        "admin2_calendar_month",
        "survey_year",
        "admin2_code",
    ]
    return frame[required].notna().all(axis=1) & frame["analysis_weight"].gt(0)


def definition_sample_mask(frame: pd.DataFrame, term: str) -> pd.Series:
    required = [
        "worked_past_week",
        "low_education",
        term,
        *BASE_CONTROLS,
        "analysis_weight",
        "admin2_calendar_month",
        "survey_year",
        "admin2_code",
        "admin1_code",
    ]
    return frame[required].notna().all(axis=1) & frame["analysis_weight"].gt(0)


def load_interaction_adjustments(path: Path) -> pd.DataFrame:
    audit = pd.read_csv(path)
    terms = {
        "threshold": "wb26_current_lag_total_5days_x_low_education",
        "continuous": "wbmean_current_lag_average_1c_x_low_education",
    }
    selected = audit.loc[
        audit["analysis_domain"].eq("cumulative_heterogeneity")
        & audit["test_family"].eq("cumulative_heterogeneity_interactions")
        & audit["term"].isin(terms.values())
        & audit["is_target_test"].eq(True)
    ].copy()
    if len(selected) != 2 or selected["term"].nunique() != 2:
        raise AssertionError("Expected two registered education interactions")
    inverse = {term: family for family, term in terms.items()}
    selected["exposure_family"] = selected["term"].map(inverse)
    return selected.set_index("exposure_family")


def common_cell_support(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    exposure_terms = [metadata["term"] for metadata in EXPOSURES.values()]
    cell_columns = [
        "survey_year",
        "survey_month",
        "admin2_code",
        "admin2_calendar_month",
        *exposure_terms,
    ]
    selected = frame.loc[mask, cell_columns].copy()
    key = ["survey_year", "survey_month", "admin2_code"]
    for term in exposure_terms:
        within_cell_unique = selected.groupby(key, observed=True)[term].nunique(dropna=False)
        if int(within_cell_unique.max()) != 1:
            raise AssertionError(f"Exposure varies within district-year-month: {term}")
    cells = selected.drop_duplicates(key).reset_index(drop=True)
    residual_sd = {}
    for term in exposure_terms:
        residual = residualize_exposure(cells, term).dropna()
        residual_sd[term] = float(residual.std(ddof=0))
    return pd.Series(residual_sd, name="within_fe_sd")


def linear_combination(
    params: pd.Series,
    covariance: pd.DataFrame,
    terms: dict[str, float],
) -> tuple[float, float, float, float, float]:
    estimate = sum(float(weight) * float(params[term]) for term, weight in terms.items())
    variance = 0.0
    for term_a, weight_a in terms.items():
        for term_b, weight_b in terms.items():
            variance += (
                float(weight_a)
                * float(weight_b)
                * float(covariance.loc[term_a, term_b])
            )
    standard_error = float(np.sqrt(max(variance, 0.0)))
    p_value = float(2.0 * norm.sf(abs(estimate / standard_error)))
    return (
        estimate,
        standard_error,
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
        p_value,
    )


def fit_education_model(
    frame: pd.DataFrame,
    *,
    family: str,
    term: str,
    specification: str,
    sample_label: str,
    mask: pd.Series,
    cluster_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, object]:
    interaction = f"{term}_x_low_education"
    working = frame.copy()
    working[interaction] = working[term] * working["low_education"]
    _records, model = fit_hdfe(
        working,
        f"education_{specification}_{family}",
        "worked_past_week",
        [term, "low_education", interaction],
        BASE_CONTROLS,
        sample_label=sample_label,
        cluster_columns=cluster_columns,
        mask=mask,
    )
    selected = working.loc[mask].copy()
    return selected, model


def estimate_models(
    frame: pd.DataFrame,
    adjustments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mask = common_sample_mask(frame)
    support = common_cell_support(frame, mask)
    rows: list[dict[str, object]] = []
    for family, metadata in EXPOSURES.items():
        term = metadata["term"]
        interaction = f"{term}_x_low_education"
        selected, model = fit_education_model(
            frame,
            family=family,
            term=term,
            specification="common",
            sample_label="common cumulative exposure and education sample",
            mask=mask,
        )
        if int(model.nobs) != int(mask.sum()):
            raise AssertionError("Education models do not use the frozen common sample")
        combinations = {
            "higher_education": {term: 1.0},
            "low_education": {term: 1.0, interaction: 1.0},
            "interaction": {interaction: 1.0},
        }
        within_fe_sd = float(support[term])
        for row_type, row_label in ROWS:
            estimate, standard_error, lower, upper, p_value = linear_combination(
                model.params, model.cov, combinations[row_type]
            )
            family_q = np.nan
            global_q = np.nan
            if row_type == "interaction" and adjustments is not None:
                registered = adjustments.loc[family]
                if not np.isclose(p_value, float(registered["p_value"]), atol=5e-5):
                    raise AssertionError(
                        f"Frozen interaction p-value mismatch: {family}"
                    )
                family_q = float(registered["family_bh_q_value"])
                global_q = float(registered["global_bh_q_value"])
            rows.append(
                {
                    "panel": "main",
                    "specification": "frozen_common",
                    "specification_label": "Frozen common sample",
                    "exposure_family": family,
                    "exposure_label": metadata["label"],
                    "row_type": row_type,
                    "row_label": row_label,
                    "raw_effect_pp": 100.0 * estimate,
                    "raw_standard_error_pp": 100.0 * standard_error,
                    "raw_ci_lower_pp": 100.0 * lower,
                    "raw_ci_upper_pp": 100.0 * upper,
                    "within_fe_sd": within_fe_sd,
                    "standardized_effect_pp": 100.0 * estimate * within_fe_sd,
                    "standardized_ci_lower_pp": 100.0 * lower * within_fe_sd,
                    "standardized_ci_upper_pp": 100.0 * upper * within_fe_sd,
                    "p_value": p_value,
                    "education_family_q_value": family_q,
                    "global_bh_q_value": global_q,
                    "n_obs": int(model.nobs),
                    "n_admin2": int(selected["admin2_code"].nunique()),
                    "n_provinces": int(selected["admin1_code"].nunique()),
                    "sample": "common cumulative exposure and education sample",
                    "controls": "+".join(BASE_CONTROLS),
                    "fixed_effects": "admin2-by-calendar-month + survey year",
                    "cluster_level": "admin2_code",
                }
            )
    estimates = pd.DataFrame(rows)
    if estimates["n_obs"].nunique() != 1:
        raise AssertionError("Education estimates do not share one observation count")
    return estimates


def interaction_record(
    *,
    frame: pd.DataFrame,
    family: str,
    term: str,
    panel: str,
    specification: str,
    specification_label: str,
    mask: pd.Series,
    within_fe_sd: float,
    cluster_columns: list[str] | None = None,
    omitted_wave: str | None = None,
) -> dict[str, object]:
    selected, model = fit_education_model(
        frame,
        family=family,
        term=term,
        specification=specification,
        sample_label=specification_label,
        mask=mask,
        cluster_columns=cluster_columns,
    )
    interaction = f"{term}_x_low_education"
    estimate, standard_error, lower, upper, p_value = linear_combination(
        model.params, model.cov, {interaction: 1.0}
    )
    return {
        "panel": panel,
        "specification": specification,
        "specification_label": specification_label,
        "exposure_family": family,
        "exposure_label": EXPOSURES[family]["label"],
        "row_type": "interaction",
        "row_label": "Difference: low minus higher",
        "raw_effect_pp": 100.0 * estimate,
        "raw_standard_error_pp": 100.0 * standard_error,
        "raw_ci_lower_pp": 100.0 * lower,
        "raw_ci_upper_pp": 100.0 * upper,
        "within_fe_sd": within_fe_sd,
        "standardized_effect_pp": 100.0 * estimate * within_fe_sd,
        "standardized_ci_lower_pp": 100.0 * lower * within_fe_sd,
        "standardized_ci_upper_pp": 100.0 * upper * within_fe_sd,
        "p_value": p_value,
        "education_family_q_value": np.nan,
        "global_bh_q_value": np.nan,
        "n_obs": int(model.nobs),
        "n_admin2": int(selected["admin2_code"].nunique()),
        "n_provinces": int(selected["admin1_code"].nunique()),
        "sample": specification_label,
        "controls": "+".join(BASE_CONTROLS),
        "fixed_effects": "admin2-by-calendar-month + survey year",
        "cluster_level": "+".join(cluster_columns or ["admin2_code"]),
        "omitted_wave": omitted_wave,
    }


def estimate_stability(frame: pd.DataFrame) -> pd.DataFrame:
    common_mask = common_sample_mask(frame)
    support = common_cell_support(frame, common_mask)
    wave_order = (
        frame.loc[common_mask, ["survey_wave", "survey_year"]]
        .drop_duplicates()
        .sort_values(["survey_year", "survey_wave"])["survey_wave"]
        .tolist()
    )
    rows: list[dict[str, object]] = []
    for wave in wave_order:
        for family, metadata in EXPOSURES.items():
            rows.append(
                interaction_record(
                    frame=frame,
                    family=family,
                    term=metadata["term"],
                    panel="leave_one_wave_out",
                    specification=f"exclude_{wave}",
                    specification_label=f"Exclude {wave}",
                    mask=common_mask & frame["survey_wave"].ne(wave),
                    within_fe_sd=float(support[metadata["term"]]),
                    omitted_wave=wave,
                )
            )

    for specification, label in SENSITIVITY_ROWS:
        for family, metadata in EXPOSURES.items():
            if specification == "definition_specific":
                mask = definition_sample_mask(frame, metadata["term"])
            elif specification == "exclude_timing_risk":
                mask = common_mask & ~frame["survey_wave"].isin(
                    ["2011-12", "2019", "2021"]
                )
            elif specification == "balanced_districts":
                mask = common_mask & frame["balanced_district"]
            elif specification == "exclude_fallback":
                mask = common_mask & frame["fallback_excluded"]
            else:
                mask = common_mask
            clusters = ["admin1_code"] if specification == "province_clustering" else None
            rows.append(
                interaction_record(
                    frame=frame,
                    family=family,
                    term=metadata["term"],
                    panel="sensitivity",
                    specification=specification,
                    specification_label=label,
                    mask=mask,
                    within_fe_sd=float(support[metadata["term"]]),
                    cluster_columns=clusters,
                )
            )
    stability = pd.DataFrame(rows)
    if len(stability) != 18 + 2 * len(SENSITIVITY_ROWS):
        raise AssertionError("Education stability output has an unexpected row count")
    return stability


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT, labelsize=9)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    sns.despine(ax=axis, top=True, right=True)


def p_threshold_display(value: float) -> tuple[float, str] | None:
    """Map an exact p value to its strongest approved display threshold."""
    if not np.isfinite(value) or value >= 0.10:
        return None
    if value < 0.01:
        return 0.01, r"$p < 0.01$"
    if value < 0.05:
        return 0.05, r"$p < 0.05$"
    return 0.10, r"$p < 0.10$"


def paired_forest(
    axis: plt.Axes,
    data: pd.DataFrame,
    row_order: list[str],
    row_labels: list[str],
) -> None:
    styles = {
        "threshold": {"color": TEAL, "marker": "o", "label": "Cumulative WB26 days"},
        "continuous": {"color": NAVY, "marker": "D", "label": "Two-month mean WBmax"},
    }
    y_positions = np.arange(len(row_order))[::-1]
    for family, offset in [("threshold", 0.11), ("continuous", -0.11)]:
        subset = data.loc[data["exposure_family"].eq(family)].set_index(
            "specification"
        ).loc[row_order]
        estimate = subset["standardized_effect_pp"].to_numpy(float)
        lower = subset["standardized_ci_lower_pp"].to_numpy(float)
        upper = subset["standardized_ci_upper_pp"].to_numpy(float)
        axis.errorbar(
            estimate,
            y_positions + offset,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt=styles[family]["marker"],
            color=styles[family]["color"],
            markerfacecolor=styles[family]["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.4,
            capsize=2.4,
            linewidth=1.2,
            label=styles[family]["label"],
            zorder=3,
        )
    bounds = data[["standardized_ci_lower_pp", "standardized_ci_upper_pp"]].to_numpy(
        float
    )
    lower_limit = min(-0.05, float(np.nanmin(bounds)))
    upper_limit = max(0.05, float(np.nanmax(bounds)))
    span = upper_limit - lower_limit
    axis.set_xlim(lower_limit - 0.08 * span, upper_limit + 0.08 * span)
    axis.set_yticks(y_positions, row_labels)
    axis.axvline(0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    axis.set_xlabel(
        "Low-minus-higher education difference (pp)\nper within-FE SD"
    )
    style_axis(axis)


def sample_label_for_pair(data: pd.DataFrame, specification: str) -> str:
    selected = data.loc[data["specification"].eq(specification)].set_index(
        "exposure_family"
    )
    threshold_n = int(selected.loc["threshold", "n_obs"])
    continuous_n = int(selected.loc["continuous", "n_obs"])
    if threshold_n == continuous_n:
        return f"N={threshold_n:,}"
    return f"N={threshold_n:,}/{continuous_n:,}"


def panel_header(axis: plt.Axes, label: str, header: str) -> None:
    axis.text(
        0.0,
        1.045,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="top",
    )
    axis.text(
        0.075,
        1.045,
        header,
        transform=axis.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )


def draw_leave_one_out_summary(
    axis: plt.Axes,
    leave_one_out: pd.DataFrame,
    main: pd.DataFrame,
) -> None:
    styles = {
        "threshold": {"color": TEAL, "marker": "o", "label": "WB26 days"},
        "continuous": {"color": NAVY, "marker": "D", "label": "Mean WBmax"},
    }
    y_positions = {"threshold": 1.0, "continuous": 0.0}
    main_interactions = main.loc[main["row_type"].eq("interaction")].set_index(
        "exposure_family"
    )
    for family in ["threshold", "continuous"]:
        subset = leave_one_out.loc[
            leave_one_out["exposure_family"].eq(family)
        ].sort_values("omitted_wave")
        values = subset["standardized_effect_pp"].to_numpy(float)
        y = y_positions[family]
        jitter = np.linspace(-0.075, 0.075, len(values))
        axis.plot(
            [values.min(), values.max()],
            [y, y],
            color=styles[family]["color"],
            linewidth=3.2,
            alpha=0.22,
            solid_capstyle="round",
            zorder=1,
        )
        axis.scatter(
            values,
            y + jitter,
            s=28,
            color=styles[family]["color"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )
        frozen = main_interactions.loc[family]
        estimate = float(frozen["standardized_effect_pp"])
        lower = float(frozen["standardized_ci_lower_pp"])
        upper = float(frozen["standardized_ci_upper_pp"])
        axis.errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - lower], [upper - estimate]]),
            fmt=styles[family]["marker"],
            color=styles[family]["color"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=7.0,
            capsize=3.0,
            linewidth=1.35,
            zorder=4,
        )
        adverse = int((values < 0).sum())
        axis.text(
            0.985,
            y,
            f"{adverse}/{len(values)} adverse",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.0,
            color=styles[family]["color"],
            fontweight="bold",
        )
    axis.axvline(0, color=MUTED, linestyle="--", linewidth=1.0, zorder=0)
    axis.set_yticks(
        [y_positions["threshold"], y_positions["continuous"]],
        [styles["threshold"]["label"], styles["continuous"]["label"]],
    )
    axis.set_ylim(-0.55, 1.55)
    axis.set_xlabel("Low-minus-higher difference (pp per within-FE SD)")
    style_axis(axis)


def draw_multiplicity_ladder(
    axis: plt.Axes,
    main: pd.DataFrame,
) -> None:
    styles = {
        "threshold": {
            "color": TEAL,
            "marker": "o",
            "label": "Cumulative WB26 days",
            "offset": 0.075,
        },
        "continuous": {
            "color": NAVY,
            "marker": "D",
            "label": "Two-month mean WBmax",
            "offset": -0.075,
        },
    }
    interactions = main.loc[main["row_type"].eq("interaction")].set_index(
        "exposure_family"
    )
    metric_rows = [
        ("p_value", "Nominal threshold", 2.0),
        ("education_family_q_value", "Family BH q", 1.0),
        ("global_bh_q_value", "Global BH q", 0.0),
    ]
    axis.axvspan(0.0, 0.10, color="#E8F3EF", alpha=0.82, zorder=0)
    axis.axvline(
        0.10,
        color=TEAL,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        zorder=1,
    )
    for family in ["threshold", "continuous"]:
        style = styles[family]
        values_list: list[float] = []
        display_labels: list[str] = []
        for metric, _label, _y in metric_rows:
            raw_value = float(interactions.loc[family, metric])
            if metric == "p_value":
                threshold = p_threshold_display(raw_value)
                if threshold is None:
                    values_list.append(np.nan)
                    display_labels.append("")
                else:
                    values_list.append(threshold[0])
                    display_labels.append(threshold[1])
            else:
                values_list.append(raw_value)
                display_labels.append(f"{raw_value:.3f}")
        values = np.array(values_list)
        y_values = np.array([y for _metric, _label, y in metric_rows]) + style[
            "offset"
        ]
        valid = np.isfinite(values)
        axis.plot(
            values[valid],
            y_values[valid],
            color=style["color"],
            linewidth=1.6,
            alpha=0.9,
            zorder=2,
        )
        axis.scatter(
            values[valid],
            y_values[valid],
            s=48,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        for value, y, display_label in zip(
            values, y_values, display_labels, strict=True
        ):
            if not np.isfinite(value):
                continue
            axis.text(
                value + 0.006,
                y,
                display_label,
                ha="left",
                va="center",
                fontsize=8.0,
                color=style["color"],
                fontweight="bold",
            )
    axis.set_xlim(0.0, 0.235)
    axis.set_ylim(-0.52, 2.52)
    axis.set_yticks(
        [row[2] for row in metric_rows],
        [row[1] for row in metric_rows],
    )
    axis.set_xticks(np.arange(0.0, 0.21, 0.05))
    axis.set_xlabel("Canonical p threshold or adjusted q value")
    axis.grid(axis="x", color=GRID, linewidth=0.75, alpha=0.85)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", colors=TEXT, labelsize=8.4)
    sns.despine(ax=axis, top=True, right=True)
    axis.text(
        0.098,
        2.43,
        "10% threshold",
        ha="right",
        va="top",
        fontsize=7.5,
        color=TEAL,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.055,
        (
            f"Common sample: N = {int(main['n_obs'].iloc[0]):,}; "
            f"districts = {int(main['n_admin2'].iloc[0])}. "
            "Family-level support does not survive global adjustment."
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=NAVY,
        fontweight="bold",
        bbox={
            "facecolor": "#FFF8E4",
            "edgecolor": "#E4C36F",
            "boxstyle": "round,pad=0.35",
        },
    )


def build_figure(estimates: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 9.2,
            "text.color": TEXT,
            "axes.edgecolor": MUTED,
        }
    )
    figure = plt.figure(figsize=(15.2, 8.7))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.03, 1.38],
        height_ratios=[0.92, 1.23],
        left=0.12,
        right=0.985,
        bottom=0.12,
        top=0.95,
        wspace=0.32,
        hspace=0.45,
    )
    main_axis = figure.add_subplot(grid[0, 0])
    leave_one_out_axis = figure.add_subplot(grid[0, 1])
    sensitivity_axis = figure.add_subplot(grid[1, 0])
    matrix_axis = figure.add_subplot(grid[1, 1])

    main = estimates.loc[estimates["panel"].eq("main")].copy()
    leave_one_out = estimates.loc[
        estimates["panel"].eq("leave_one_wave_out")
    ].copy()
    sensitivity = estimates.loc[estimates["panel"].eq("sensitivity")].copy()
    row_order = [item[0] for item in ROWS]
    labels = [item[1] for item in ROWS]
    y_positions = np.arange(len(row_order))[::-1]
    all_bounds = main[
        ["standardized_ci_lower_pp", "standardized_ci_upper_pp"]
    ].to_numpy()
    extent = max(0.25, float(np.nanmax(np.abs(all_bounds))))
    x_limit = 1.16 * extent
    main_styles = {
        "threshold": {"color": TEAL, "marker": "o", "label": "Cumulative WB26 days"},
        "continuous": {"color": NAVY, "marker": "D", "label": "Two-month mean WBmax"},
    }
    main_axis.axhspan(-0.40, 0.40, color="#FFF1C9", alpha=0.78, zorder=0)
    for family, offset in [("threshold", 0.11), ("continuous", -0.11)]:
        subset = main.loc[main["exposure_family"].eq(family)].set_index(
            "row_type"
        ).loc[row_order]
        for y, row_type in zip(y_positions + offset, row_order, strict=True):
            row = subset.loc[row_type]
            estimate = float(row["standardized_effect_pp"])
            lower = float(row["standardized_ci_lower_pp"])
            upper = float(row["standardized_ci_upper_pp"])
            main_axis.errorbar(
                estimate,
                y,
                xerr=np.array([[estimate - lower], [upper - estimate]]),
                fmt=main_styles[family]["marker"],
                color=main_styles[family]["color"],
                markerfacecolor=main_styles[family]["color"],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=6.2,
                capsize=3.0,
                linewidth=1.5,
                label=main_styles[family]["label"] if row_type == row_order[0] else None,
                zorder=3,
            )
    main_axis.axvline(0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    main_axis.axhline(0.5, color=GRID, linestyle=":", linewidth=1.0, zorder=0)
    main_axis.set_xlim(-x_limit, 0.22 * x_limit)
    main_axis.set_yticks(y_positions, labels)
    main_axis.set_xlabel("Work participation difference (pp per within-FE SD)")
    style_axis(main_axis)
    for tick in main_axis.get_yticklabels():
        if tick.get_text() == "None, preschool, or primary":
            tick.set_color(GOLD)
        if tick.get_text() == "Difference: low minus higher":
            tick.set_fontweight("bold")
            tick.set_color(NAVY)
    panel_header(main_axis, "a", "Education-group slopes and interaction")

    draw_leave_one_out_summary(leave_one_out_axis, leave_one_out, main)
    panel_header(
        leave_one_out_axis,
        "b",
        "Leave-one-wave-out interaction stability",
    )

    sensitivity_order = [item[0] for item in SENSITIVITY_ROWS]
    sensitivity_labels = [item[1] for item in SENSITIVITY_ROWS]
    paired_forest(
        sensitivity_axis,
        sensitivity,
        sensitivity_order,
        sensitivity_labels,
    )
    sensitivity_axis.set_xlabel(
        "Low-minus-higher education difference (pp per within-FE SD)"
    )
    sensitivity_axis.tick_params(axis="y", labelsize=7.8)
    panel_header(sensitivity_axis, "c", "Frozen sensitivity checks")

    draw_multiplicity_ladder(matrix_axis, main)
    panel_header(matrix_axis, "d", "Multiplicity ladder for education interaction")

    interaction_bounds = estimates.loc[
        ~estimates["panel"].eq("main"),
        ["standardized_ci_lower_pp", "standardized_ci_upper_pp"],
    ].to_numpy(float)
    interaction_lower = min(-0.05, float(np.nanmin(interaction_bounds)))
    interaction_upper = max(0.05, float(np.nanmax(interaction_bounds)))
    interaction_span = interaction_upper - interaction_lower
    shared_limits = (
        interaction_lower - 0.06 * interaction_span,
        interaction_upper + 0.06 * interaction_span,
    )
    leave_one_out_axis.set_xlim(*shared_limits)
    sensitivity_axis.set_xlim(*shared_limits)

    handles, legend_labels = main_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        frameon=False,
        fontsize=8.5,
        ncol=2,
        handletextpad=0.7,
        columnspacing=2.2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def validate(estimates: pd.DataFrame, output: Path) -> None:
    main = estimates.loc[estimates["panel"].eq("main")]
    if len(main) != 6:
        raise AssertionError(f"Expected six main education estimates, observed {len(main)}")
    interactions = main.loc[main["row_type"].eq("interaction")]
    if interactions[["education_family_q_value", "global_bh_q_value"]].isna().any().any():
        raise AssertionError("Frozen interaction multiplicity values are missing")
    if len(estimates.loc[estimates["panel"].eq("leave_one_wave_out")]) != 18:
        raise AssertionError("Expected 18 leave-one-wave-out interaction estimates")
    if len(estimates.loc[estimates["panel"].eq("sensitivity")]) != 12:
        raise AssertionError("Expected 12 frozen sensitivity interaction estimates")
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError("Education figure is missing or empty")


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output)
    review_output = resolve_path(args.review_output)
    estimates_output = resolve_path(args.estimates_output)
    multiplicity_audit = resolve_path(args.multiplicity_audit)
    adjustments = load_interaction_adjustments(multiplicity_audit)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = load_data(connection, args.schema)
        connection.rollback()
    frame = prepare_frame(frame)
    main_estimates = estimate_models(frame, adjustments)
    stability_estimates = estimate_stability(frame)
    estimates = pd.concat([main_estimates, stability_estimates], ignore_index=True)
    build_figure(estimates, output)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    if review_output.resolve() != output.resolve():
        shutil.copy2(output, review_output)
    estimates_output.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(estimates_output, index=False)
    validate(estimates, output)
    print(f"common_sample_n={int(main_estimates['n_obs'].iloc[0])}")
    print(f"districts={int(main_estimates['n_admin2'].iloc[0])}")
    for row in main_estimates.itertuples(index=False):
        family_q = getattr(row, "education_family_q_value")
        global_q = getattr(row, "global_bh_q_value")
        q_text = f"{family_q:.6f}" if np.isfinite(family_q) else "na"
        global_q_text = f"{global_q:.6f}" if np.isfinite(global_q) else "na"
        print(
            f"{row.exposure_family}_{row.row_type}: "
            f"standardized_effect_pp={row.standardized_effect_pp:.6f} "
            f"p={row.p_value:.6f} family_q={q_text} global_q={global_q_text}"
        )
    stability = estimates.loc[~estimates["panel"].eq("main")]
    print(
        f"stability_rows={len(stability)} "
        f"adverse_interactions={int(stability['standardized_effect_pp'].lt(0).sum())}"
    )
    print(f"saved_figure={output.relative_to(ROOT)}")
    print(f"saved_review_copy={review_output.relative_to(ROOT)}")
    print(f"saved_estimates={estimates_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
