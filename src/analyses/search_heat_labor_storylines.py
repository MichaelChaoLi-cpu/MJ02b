#!/usr/bin/env python3
"""Search a transparent, read-only universe of heat-labor storylines.

This exploratory program preserves every tested specification and evaluates
timing, outcomes, heterogeneity, adaptation proxies, occupation, placebos, and
leave-one-wave-out stability. It applies Benjamini-Hochberg corrections and
creates a diagnostic storyline scorecard. It does not modify PostgreSQL or
AnaSOP and must not be used to hide null specifications.
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
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from audit_exposure_definition_comparison import residualize_exposure
from run_heat_labor_credibility_audit import fit_hdfe
from run_heat_labor_extensions import BASE_CONTROLS, connection_args, load_data


DEFAULT_OUTPUT = Path("data/exp/storyline-search")

EXPOSURES = {
    "wb26": {
        "label": "WBmax >= 26 C days",
        "short": "WB26 days",
        "current": "wb26_5days",
        "lag": "lag_wb26_5days",
        "lead": "lead_wb26_5days",
        "cumulative": "wb26_current_lag_total_5days",
        "unit": "5 threshold-exceedance days",
    },
    "wbmean": {
        "label": "Mean daily WBmax",
        "short": "Mean WBmax",
        "current": "wbmean_1c",
        "lag": "lag_wbmean_1c",
        "lead": "lead_wbmean_1c",
        "cumulative": "wbmean_current_lag_average_1c",
        "unit": "1 C",
    },
}

MODIFIERS = {
    "female": {
        "label": "Female vulnerability",
        "zero": "Male",
        "one": "Female",
        "expected_sign": -1,
        "domain": "distribution",
        "caveat": "Sex interaction is associational.",
        "extra_controls": [],
    },
    "low_education": {
        "label": "Low-education vulnerability",
        "zero": "Lower-secondary or above",
        "one": "None/preschool/primary",
        "expected_sign": -1,
        "domain": "distribution",
        "caveat": "Education is harmonized but may proxy occupation and income.",
        "extra_controls": [],
    },
    "female_low_education": {
        "label": "Female with low education (composite contrast)",
        "zero": "All other observed groups",
        "one": "Female with low education",
        "expected_sign": -1,
        "domain": "intersectional distribution",
        "caveat": "Composite group is exploratory and not a causal mechanism.",
        "extra_controls": ["low_education"],
    },
    "rural": {
        "label": "Rural vulnerability",
        "zero": "Urban",
        "one": "Rural",
        "expected_sign": -1,
        "domain": "distribution",
        "caveat": "Urban-rural coding and exposure are district-level.",
        "extra_controls": [],
    },
    "older_worker": {
        "label": "Older-worker vulnerability",
        "zero": "Age 15-49",
        "one": "Age 50-64",
        "expected_sign": -1,
        "domain": "distribution",
        "caveat": "Age differences may reflect occupation and cohort composition.",
        "extra_controls": [],
    },
    "electricity_spending_positive": {
        "label": "Electricity-spending buffering",
        "zero": "No positive electricity spending",
        "one": "Positive electricity spending",
        "expected_sign": 1,
        "domain": "adaptation proxy",
        "caveat": "Spending is not verified electricity access or cooling ownership.",
        "extra_controls": [],
    },
    "high_crowding": {
        "label": "Crowding vulnerability",
        "zero": "At/below wave median crowding",
        "one": "Above wave median crowding",
        "expected_sign": -1,
        "domain": "housing-resource proxy",
        "caveat": "Crowding is a resource proxy rather than direct thermal adaptation.",
        "extra_controls": [],
    },
}

OCCUPATION_MODIFIERS = {
    "agriculture_occupation_candidate": {
        "label": "Agricultural-worker hours vulnerability",
        "zero": "Other occupation groups",
        "one": "Agriculture major-group candidate",
        "expected_sign": -1,
        "domain": "occupation",
        "caveat": "Occupation is observed among workers and codes are not fully validated across waves.",
    },
    "manual_outdoor_occupation_candidate": {
        "label": "Manual/outdoor-worker hours vulnerability",
        "zero": "Major groups 0-5",
        "one": "Major groups 6-9 candidate",
        "expected_sign": -1,
        "domain": "occupation",
        "caveat": "The 6-9 grouping is a coarse outdoor/manual proxy among workers.",
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


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["wb26_current_lag_total_5days"] = (
        frame["wb26_5days"] + frame["lag_wb26_5days"]
    )
    frame["wbmean_current_lag_average_1c"] = (
        frame[["wbmean_1c", "lag_wbmean_1c"]].mean(axis=1)
    ).where(frame[["wbmean_1c", "lag_wbmean_1c"]].notna().all(axis=1))
    frame["hours_observed"] = frame["weekly_hours_including_zero"].notna().astype(float)
    frame["wage_observed"] = frame["log_monthly_salary_wages"].notna().astype(float)
    frame["female_low_education"] = (
        frame["female"].eq(1) & frame["low_education"].eq(1)
    ).where(frame[["female", "low_education"]].notna().all(axis=1)).astype(float)
    return frame


def build_cell_frame(frame: pd.DataFrame) -> pd.DataFrame:
    exposure_terms = sorted(
        {
            metadata[window]
            for metadata in EXPOSURES.values()
            for window in ["current", "lag", "lead", "cumulative"]
        }
    )
    columns = [
        "survey_wave",
        "survey_year",
        "survey_month",
        "admin2_code",
        *exposure_terms,
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


def calculate_term_support(cells: pd.DataFrame) -> pd.DataFrame:
    terms = sorted(
        {
            metadata[window]
            for metadata in EXPOSURES.values()
            for window in ["current", "lag", "lead", "cumulative"]
        }
    )
    rows = []
    for term in terms:
        residual = residualize_exposure(cells, term)
        observed = cells[term].dropna()
        residual_observed = residual.dropna()
        rows.append(
            {
                "term": term,
                "n_cells": int(observed.size),
                "mean": float(observed.mean()),
                "raw_sd": float(observed.std(ddof=0)),
                "residual_sd": float(residual_observed.std(ddof=0)),
                "missing_rate": float(cells[term].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def result_scale(outcome: str) -> float:
    return 100.0 if outcome in ["worked_past_week", "hours_observed", "wage_observed"] else 1.0


def enrich_results(
    records: list[dict[str, object]],
    *,
    model_id: str,
    analysis_domain: str,
    test_family: str,
    role: str,
    exposure_family: str,
    target_terms: list[str],
    term_support: pd.Series,
    standardization_term: str | None = None,
) -> list[dict[str, object]]:
    enriched = []
    for record in records:
        term = str(record["term"])
        scale = result_scale(str(record["outcome"]))
        sd = float(term_support.get(standardization_term or term, np.nan))
        record.update(
            {
                "model_id": model_id,
                "analysis_domain": analysis_domain,
                "test_family": test_family,
                "role": role,
                "exposure_family": exposure_family,
                "is_target_test": term in target_terms,
                "effect_display": float(record["coefficient"]) * scale,
                "ci_lower_display": float(record["ci_lower_95"]) * scale,
                "ci_upper_display": float(record["ci_upper_95"]) * scale,
                "term_within_fe_sd": sd,
                "standardized_effect_display": (
                    float(record["coefficient"]) * scale * sd if np.isfinite(sd) else np.nan
                ),
                "standardized_ci_lower_display": (
                    float(record["ci_lower_95"]) * scale * sd if np.isfinite(sd) else np.nan
                ),
                "standardized_ci_upper_display": (
                    float(record["ci_upper_95"]) * scale * sd if np.isfinite(sd) else np.nan
                ),
            }
        )
        enriched.append(record)
    return enriched


def fit_and_collect(
    frame: pd.DataFrame,
    all_rows: list[dict[str, object]],
    registry: list[dict[str, object]],
    *,
    model_id: str,
    analysis_domain: str,
    test_family: str,
    role: str,
    exposure_family: str,
    outcome: str,
    terms: list[str],
    controls: list[str],
    mask: pd.Series,
    sample_label: str,
    target_terms: list[str],
    term_support: pd.Series,
    standardization_term: str | None = None,
) -> object:
    registry.append(
        {
            "model_id": model_id,
            "analysis_domain": analysis_domain,
            "test_family": test_family,
            "role": role,
            "exposure_family": exposure_family,
            "outcome": outcome,
            "terms": "+".join(terms),
            "controls": "+".join(controls),
            "sample": sample_label,
            "target_terms": "+".join(target_terms),
            "status": "planned",
        }
    )
    try:
        records, model = fit_hdfe(
            frame,
            model_id,
            outcome,
            terms,
            controls,
            sample_label=sample_label,
            mask=mask,
        )
        all_rows.extend(
            enrich_results(
                records,
                model_id=model_id,
                analysis_domain=analysis_domain,
                test_family=test_family,
                role=role,
                exposure_family=exposure_family,
                target_terms=target_terms,
                term_support=term_support,
                standardization_term=standardization_term,
            )
        )
        registry[-1]["status"] = "estimated"
        registry[-1]["n_obs"] = int(model.nobs)
        return model
    except Exception as error:
        registry[-1]["status"] = "failed"
        registry[-1]["error"] = str(error)
        return None


def interaction_model(
    frame: pd.DataFrame,
    all_rows: list[dict[str, object]],
    registry: list[dict[str, object]],
    slopes: list[dict[str, object]],
    *,
    exposure_family: str,
    exposure_term: str,
    modifier: str,
    metadata: dict[str, object],
    outcome: str,
    domain: str,
    controls: list[str],
    mask: pd.Series,
    term_support: pd.Series,
) -> None:
    interaction = f"{exposure_term}_x_{modifier}"
    working = frame.copy()
    working[interaction] = working[exposure_term] * working[modifier]
    effective_controls = [item for item in controls if item != modifier]
    terms = [exposure_term, modifier, interaction]
    model_id = f"{domain}_{exposure_family}_{modifier}"
    model = fit_and_collect(
        working,
        all_rows,
        registry,
        model_id=model_id,
        analysis_domain=domain,
        test_family=f"{domain}_interactions",
        role="effect_modification",
        exposure_family=exposure_family,
        outcome=outcome,
        terms=terms,
        controls=effective_controls,
        mask=mask & working[modifier].notna() & working[outcome].notna(),
        sample_label=f"nonmissing {modifier}",
        target_terms=[interaction],
        term_support=term_support,
        standardization_term=exposure_term,
    )
    if model is None:
        return
    params = model.params
    covariance = model.cov
    sd = float(term_support.get(exposure_term, np.nan))
    scale = result_scale(outcome)
    for value, group_label in [(0, metadata["zero"]), (1, metadata["one"])]:
        estimate = float(params[exposure_term])
        variance = float(covariance.loc[exposure_term, exposure_term])
        if value == 1:
            estimate += float(params[interaction])
            variance += float(covariance.loc[interaction, interaction])
            variance += 2 * float(covariance.loc[exposure_term, interaction])
        standard_error = float(np.sqrt(max(variance, 0)))
        p_value = float(2 * norm.sf(abs(estimate / standard_error)))
        slopes.append(
            {
                "model_id": model_id,
                "analysis_domain": domain,
                "exposure_family": exposure_family,
                "exposure_term": exposure_term,
                "modifier": modifier,
                "storyline": metadata["label"],
                "group_value": value,
                "group_label": group_label,
                "slope": estimate,
                "standard_error": standard_error,
                "ci_lower_95": estimate - 1.96 * standard_error,
                "ci_upper_95": estimate + 1.96 * standard_error,
                "p_value": p_value,
                "effect_display": estimate * scale,
                "standardized_effect_display": estimate * scale * sd,
                "n_obs": int(model.nobs),
                "caveat": metadata["caveat"],
            }
        )


def run_specification_universe(
    frame: pd.DataFrame, term_support_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, object]] = []
    registry: list[dict[str, object]] = []
    slopes: list[dict[str, object]] = []
    term_support = term_support_frame.set_index("term")["residual_sd"]
    work_mask = frame["worked_past_week"].notna()

    control_specs = [
        ("primary", BASE_CONTROLS),
        ("demographics_only", ["age", "age_squared", "female", "rural"]),
        ("rainfall_only", ["precipitation_100mm"]),
    ]
    for exposure_family, metadata in EXPOSURES.items():
        current = metadata["current"]
        for label, controls in control_specs:
            fit_and_collect(
                frame,
                all_rows,
                registry,
                model_id=f"current_work_{exposure_family}_{label}",
                analysis_domain="average_effect",
                test_family="current_work_controls",
                role="primary" if label == "primary" else "control_sensitivity",
                exposure_family=exposure_family,
                outcome="worked_past_week",
                terms=[current],
                controls=controls,
                mask=work_mask,
                sample_label="all working-age observations",
                target_terms=[current],
                term_support=term_support,
            )

    for exposure_family, metadata in EXPOSURES.items():
        for window in ["lag", "cumulative", "lead"]:
            term = metadata[window]
            fit_and_collect(
                frame,
                all_rows,
                registry,
                model_id=f"timing_work_{exposure_family}_{window}",
                analysis_domain="timing",
                test_family="timing_single",
                role="future_placebo" if window == "lead" else "timing_exposure",
                exposure_family=exposure_family,
                outcome="worked_past_week",
                terms=[term],
                controls=BASE_CONTROLS,
                mask=work_mask & frame[term].notna(),
                sample_label=f"{window} exposure observed",
                target_terms=[term],
                term_support=term_support,
            )
        for adjacent, role in [("lag", "distributed_lag"), ("lead", "placebo_joint")]:
            terms = [metadata["current"], metadata[adjacent]]
            fit_and_collect(
                frame,
                all_rows,
                registry,
                model_id=f"timing_work_{exposure_family}_current_plus_{adjacent}",
                analysis_domain="timing",
                test_family="timing_joint",
                role=role,
                exposure_family=exposure_family,
                outcome="worked_past_week",
                terms=terms,
                controls=BASE_CONTROLS,
                mask=work_mask & frame[terms].notna().all(axis=1),
                sample_label=f"current and {adjacent} exposure observed",
                target_terms=terms,
                term_support=term_support,
            )

    substantive_outcomes = [
        ("weekly_hours_including_zero", "inclusive_hours"),
        ("weekly_hours_worked", "worker_hours"),
        ("log_monthly_salary_wages", "log_wage"),
    ]
    for exposure_family, metadata in EXPOSURES.items():
        for window in ["current", "cumulative"]:
            term = metadata[window]
            for outcome, outcome_label in substantive_outcomes:
                fit_and_collect(
                    frame,
                    all_rows,
                    registry,
                    model_id=f"outcome_{outcome_label}_{exposure_family}_{window}",
                    analysis_domain="outcomes",
                    test_family=f"outcome_{outcome_label}",
                    role="secondary_outcome",
                    exposure_family=exposure_family,
                    outcome=outcome,
                    terms=[term],
                    controls=BASE_CONTROLS,
                    mask=frame[outcome].notna() & frame[term].notna(),
                    sample_label=f"{outcome_label}; {window} exposure observed",
                    target_terms=[term],
                    term_support=term_support,
                )
        for outcome in ["hours_observed", "wage_observed"]:
            term = metadata["current"]
            fit_and_collect(
                frame,
                all_rows,
                registry,
                model_id=f"selection_{outcome}_{exposure_family}",
                analysis_domain="selection",
                test_family="outcome_selection",
                role="selection_diagnostic",
                exposure_family=exposure_family,
                outcome=outcome,
                terms=[term],
                controls=BASE_CONTROLS,
                mask=frame[term].notna(),
                sample_label=f"selection outcome {outcome}",
                target_terms=[term],
                term_support=term_support,
            )

    for modifier, metadata in MODIFIERS.items():
        controls = [*BASE_CONTROLS, *metadata["extra_controls"]]
        for exposure_family, exposure_metadata in EXPOSURES.items():
            interaction_model(
                frame,
                all_rows,
                registry,
                slopes,
                exposure_family=exposure_family,
                exposure_term=exposure_metadata["current"],
                modifier=modifier,
                metadata=metadata,
                outcome="worked_past_week",
                domain="heterogeneity",
                controls=controls,
                mask=work_mask,
                term_support=term_support,
            )

    vulnerability_mask = (
        work_mask
        & frame["female"].notna()
        & frame["low_education"].notna()
    )
    for exposure_family, exposure_metadata in EXPOSURES.items():
        exposure_term = exposure_metadata["current"]
        working = frame.copy()
        female_interaction = f"{exposure_term}_x_female"
        education_interaction = f"{exposure_term}_x_low_education"
        triple_interaction = f"{exposure_term}_x_female_x_low_education"
        working[female_interaction] = working[exposure_term] * working["female"]
        working[education_interaction] = working[exposure_term] * working["low_education"]
        working[triple_interaction] = (
            working[exposure_term]
            * working["female"]
            * working["low_education"]
        )
        vulnerability_controls = [
            "age", "age_squared", "rural", "precipitation_100mm"
        ]
        fit_and_collect(
            working,
            all_rows,
            registry,
            model_id=f"joint_vulnerability_{exposure_family}_female_education",
            analysis_domain="heterogeneity",
            test_family="joint_vulnerability",
            role="joint_vulnerability",
            exposure_family=exposure_family,
            outcome="worked_past_week",
            terms=[
                exposure_term,
                "female",
                "low_education",
                female_interaction,
                education_interaction,
            ],
            controls=vulnerability_controls,
            mask=vulnerability_mask,
            sample_label="nonmissing sex and harmonized education",
            target_terms=[female_interaction, education_interaction],
            term_support=term_support,
            standardization_term=exposure_term,
        )
        fit_and_collect(
            working,
            all_rows,
            registry,
            model_id=f"full_factorial_{exposure_family}_female_education",
            analysis_domain="heterogeneity",
            test_family="joint_vulnerability",
            role="intersection_incremental",
            exposure_family=exposure_family,
            outcome="worked_past_week",
            terms=[
                exposure_term,
                "female",
                "low_education",
                "female_low_education",
                female_interaction,
                education_interaction,
                triple_interaction,
            ],
            controls=vulnerability_controls,
            mask=vulnerability_mask,
            sample_label="full factorial sex-by-education vulnerability model",
            target_terms=[triple_interaction],
            term_support=term_support,
            standardization_term=exposure_term,
        )

    for modifier in ["female", "low_education"]:
        metadata = MODIFIERS[modifier]
        for exposure_family, exposure_metadata in EXPOSURES.items():
            cumulative_term = exposure_metadata["cumulative"]
            interaction_model(
                frame,
                all_rows,
                registry,
                slopes,
                exposure_family=exposure_family,
                exposure_term=cumulative_term,
                modifier=modifier,
                metadata=metadata,
                outcome="worked_past_week",
                domain="cumulative_heterogeneity",
                controls=BASE_CONTROLS,
                mask=work_mask & frame[cumulative_term].notna(),
                term_support=term_support,
            )

    worker_mask = frame["worked_past_week"].eq(1) & frame["weekly_hours_worked"].notna()
    for modifier, metadata in OCCUPATION_MODIFIERS.items():
        for exposure_family, exposure_metadata in EXPOSURES.items():
            interaction_model(
                frame,
                all_rows,
                registry,
                slopes,
                exposure_family=exposure_family,
                exposure_term=exposure_metadata["current"],
                modifier=modifier,
                metadata=metadata,
                outcome="weekly_hours_worked",
                domain="occupation",
                controls=BASE_CONTROLS,
                mask=worker_mask,
                term_support=term_support,
            )

    waves = frame.sort_values("survey_year")["survey_wave"].drop_duplicates().tolist()
    for exposure_family, metadata in EXPOSURES.items():
        term = metadata["current"]
        for wave in waves:
            fit_and_collect(
                frame,
                all_rows,
                registry,
                model_id=f"loo_{exposure_family}_exclude_{wave}",
                analysis_domain="wave_stability",
                test_family="leave_one_wave_out",
                role="stability_diagnostic",
                exposure_family=exposure_family,
                outcome="worked_past_week",
                terms=[term],
                controls=BASE_CONTROLS,
                mask=work_mask & frame["survey_wave"].ne(wave),
                sample_label=f"all waves except {wave}",
                target_terms=[term],
                term_support=term_support,
            )
            all_rows[-1]["excluded_wave"] = wave

    results = pd.DataFrame(all_rows)
    registry_frame = pd.DataFrame(registry)
    slopes_frame = pd.DataFrame(slopes)
    return results, registry_frame, slopes_frame


def apply_multiplicity(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["global_bh_q_value"] = np.nan
    results["family_bh_q_value"] = np.nan
    eligible = (
        results["is_target_test"].fillna(False)
        & results["p_value"].notna()
        & ~results["analysis_domain"].eq("wave_stability")
    )
    if eligible.any():
        results.loc[eligible, "global_bh_q_value"] = multipletests(
            results.loc[eligible, "p_value"], method="fdr_bh"
        )[1]
    for family, indexes in results.loc[eligible].groupby("test_family").groups.items():
        results.loc[indexes, "family_bh_q_value"] = multipletests(
            results.loc[indexes, "p_value"], method="fdr_bh"
        )[1]
    return results


def interaction_tests(results: pd.DataFrame) -> pd.DataFrame:
    interactions = results.loc[
        results["role"].eq("effect_modification")
        & results["analysis_domain"].isin(["heterogeneity", "occupation"])
        & results["is_target_test"]
    ].copy()
    interactions["modifier"] = interactions["term"].str.split("_x_", n=1).str[1]
    interactions["storyline"] = interactions["modifier"].map(
        {
            **{key: value["label"] for key, value in MODIFIERS.items()},
            **{key: value["label"] for key, value in OCCUPATION_MODIFIERS.items()},
        }
    )
    interactions["expected_sign"] = interactions["modifier"].map(
        {
            **{key: value["expected_sign"] for key, value in MODIFIERS.items()},
            **{key: value["expected_sign"] for key, value in OCCUPATION_MODIFIERS.items()},
        }
    )
    interactions["sign_matches_story"] = np.sign(
        interactions["coefficient"]
    ).eq(interactions["expected_sign"])
    return interactions


def score_storylines(
    results: pd.DataFrame,
    slopes: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    current = results.loc[
        results["analysis_domain"].eq("average_effect") & results["is_target_test"]
    ]
    primary = current.loc[current["role"].eq("primary")]
    loo = results.loc[
        results["analysis_domain"].eq("wave_stability") & results["is_target_test"]
    ]
    leads = results.loc[
        results["role"].eq("future_placebo") & results["is_target_test"]
    ].set_index("exposure_family")
    current_primary = primary.set_index("exposure_family")
    placebo_pairs = []
    for exposure_family in EXPOSURES:
        current_effect = float(current_primary.loc[exposure_family, "standardized_effect_display"])
        lead_effect = float(leads.loc[exposure_family, "standardized_effect_display"])
        placebo_pairs.append(current_effect < 0 and current_effect < lead_effect)
    direction = float(current["standardized_effect_display"].lt(0).mean())
    replication = float(primary["standardized_effect_display"].lt(0).mean())
    stability = float(loo["standardized_effect_display"].lt(0).mean())
    placebo = float(np.mean(placebo_pairs))
    corrected = float(primary["family_bh_q_value"].lt(0.10).any())
    score = 3 * direction + 2 * replication + 2 * stability + 2 * placebo + corrected
    rows.append(
        {
            "candidate_storyline": "Average humid heat reduces work participation",
            "domain": "average effect",
            "direction_consistency": direction,
            "cross_definition_replication": replication,
            "stability_or_group_support": stability,
            "placebo_or_interaction_support": placebo,
            "multiplicity_signal": corrected,
            "diagnostic_score": score,
            "maximum_score": 10.0,
            "minimum_nominal_p": float(primary["p_value"].min()),
            "minimum_family_q": float(primary["family_bh_q_value"].min()),
            "caveat": "Average estimates remain imprecise and timing is month-level.",
        }
    )

    cumulative = results.loc[
        results["model_id"].str.contains("timing_work_.*_cumulative", regex=True)
        & results["is_target_test"]
    ]
    cumulative_interactions = results.loc[
        results["analysis_domain"].eq("cumulative_heterogeneity")
        & results["is_target_test"]
        & results["term"].str.endswith("_x_low_education")
    ].copy()
    cumulative_low_education_slopes = slopes.loc[
        slopes["analysis_domain"].eq("cumulative_heterogeneity")
        & slopes["modifier"].eq("low_education")
        & slopes["group_value"].eq(1)
    ]
    direction = float(cumulative_interactions["coefficient"].lt(0).mean())
    replication = float(direction == 1.0)
    group_support = float(
        cumulative_low_education_slopes["standardized_effect_display"].lt(0).mean()
    )
    cumulative_main_support = float(
        cumulative["standardized_effect_display"].lt(0).mean()
    )
    corrected = float(
        cumulative_interactions["family_bh_q_value"].lt(0.10).any()
    )
    integrated_score = (
        2 * direction
        + 2 * replication
        + 2 * group_support
        + 2 * cumulative_main_support
        + 2 * corrected
    )
    rows.append(
        {
            "candidate_storyline": "Cumulative humid heat disproportionately affects low-education workers",
            "domain": "integrated timing and vulnerability",
            "direction_consistency": direction,
            "cross_definition_replication": replication,
            "stability_or_group_support": group_support,
            "placebo_or_interaction_support": cumulative_main_support,
            "multiplicity_signal": corrected,
            "diagnostic_score": integrated_score,
            "maximum_score": 10.0,
            "minimum_nominal_p": float(cumulative_interactions["p_value"].min()),
            "minimum_family_q": float(
                cumulative_interactions["family_bh_q_value"].min()
            ),
            "caveat": "Integrated result remains exploratory; education may proxy occupation and cumulative month windows overlap.",
        }
    )

    distributed = results.loc[
        results["role"].eq("distributed_lag") & results["is_target_test"]
    ]
    direction = float(cumulative["standardized_effect_display"].lt(0).mean())
    replication = direction
    temporal = float(distributed["standardized_effect_display"].lt(0).mean())
    placebo = float(np.mean(placebo_pairs))
    corrected = float(cumulative["family_bh_q_value"].lt(0.10).any())
    score = 3 * direction + 2 * replication + 2 * temporal + 2 * placebo + corrected
    rows.append(
        {
            "candidate_storyline": "Humid heat has cumulative or lagged labor effects",
            "domain": "timing",
            "direction_consistency": direction,
            "cross_definition_replication": replication,
            "stability_or_group_support": temporal,
            "placebo_or_interaction_support": placebo,
            "multiplicity_signal": corrected,
            "diagnostic_score": score,
            "maximum_score": 10.0,
            "minimum_nominal_p": float(cumulative["p_value"].min()),
            "minimum_family_q": float(cumulative["family_bh_q_value"].min()),
            "caveat": "Lag exposure is month-level and cumulative windows overlap.",
        }
    )

    interactions = interaction_tests(results)
    slope_one = slopes.loc[slopes["group_value"].eq(1)].copy()
    metadata_all = {**MODIFIERS, **OCCUPATION_MODIFIERS}
    for modifier, metadata in metadata_all.items():
        if modifier in ["female", "low_education"]:
            tests = results.loc[
                results["role"].eq("joint_vulnerability")
                & results["is_target_test"]
                & results["term"].str.endswith(f"_x_{modifier}")
            ].copy()
            tests["sign_matches_story"] = np.sign(tests["coefficient"]).eq(
                metadata["expected_sign"]
            )
        elif modifier == "female_low_education":
            tests = results.loc[
                results["role"].eq("intersection_incremental")
                & results["is_target_test"]
            ].copy()
            tests["sign_matches_story"] = np.sign(tests["coefficient"]).eq(
                metadata["expected_sign"]
            )
        else:
            tests = interactions.loc[interactions["modifier"].eq(modifier)].copy()
        group_slopes = slope_one.loc[slope_one["modifier"].eq(modifier)]
        direction = float(tests["sign_matches_story"].mean())
        replication = float(direction == 1.0)
        expected_group_slope = (
            group_slopes["standardized_effect_display"].lt(0).mean()
            if metadata["expected_sign"] == -1
            else group_slopes["standardized_effect_display"].gt(
                slopes.loc[
                    slopes["modifier"].eq(modifier) & slopes["group_value"].eq(0),
                    "standardized_effect_display",
                ].to_numpy()
            ).mean()
        )
        nominal = float(tests["p_value"].lt(0.05).any())
        corrected = float(tests["family_bh_q_value"].lt(0.10).any())
        score = (
            2 * direction
            + replication
            + 3 * float(expected_group_slope)
            + nominal
            + 3 * corrected * float(expected_group_slope)
        )
        rows.append(
            {
                "candidate_storyline": (
                    "Incremental female-by-low-education vulnerability"
                    if modifier == "female_low_education"
                    else metadata["label"]
                ),
                "domain": metadata["domain"],
                "direction_consistency": direction,
                "cross_definition_replication": replication,
                "stability_or_group_support": float(expected_group_slope),
                "placebo_or_interaction_support": nominal,
                "multiplicity_signal": corrected,
                "diagnostic_score": score,
                "maximum_score": 10.0,
                "minimum_nominal_p": float(tests["p_value"].min()),
                "minimum_family_q": float(tests["family_bh_q_value"].min()),
                "caveat": metadata["caveat"],
            }
        )

    scorecard = pd.DataFrame(rows).sort_values(
        ["diagnostic_score", "minimum_family_q"], ascending=[False, True], ignore_index=True
    )
    scorecard["rank"] = np.arange(1, len(scorecard) + 1)
    scorecard["evidence_grade"] = "weak"
    exploratory = (
        scorecard["diagnostic_score"].ge(5)
        & scorecard["minimum_nominal_p"].lt(0.10)
    )
    supported = (
        scorecard["diagnostic_score"].ge(7)
        & scorecard["minimum_family_q"].lt(0.10)
        & scorecard["stability_or_group_support"].eq(1)
    )
    scorecard.loc[exploratory, "evidence_grade"] = "exploratory"
    scorecard.loc[supported, "evidence_grade"] = "family-supported"
    columns = ["rank", *[column for column in scorecard.columns if column != "rank"]]
    return scorecard[columns]


def plot_timing(results: pd.DataFrame, output: Path) -> None:
    rows = []
    for exposure_family, metadata in EXPOSURES.items():
        current = results.loc[
            results["model_id"].eq(f"current_work_{exposure_family}_primary")
            & results["is_target_test"]
        ].iloc[0]
        rows.append((exposure_family, "Current month", current))
        for window, label in [
            ("lag", "Previous month"),
            ("cumulative", "Current + previous"),
            ("lead", "Future-month placebo"),
        ]:
            row = results.loc[
                results["model_id"].eq(f"timing_work_{exposure_family}_{window}")
                & results["is_target_test"]
            ].iloc[0]
            rows.append((exposure_family, label, row))
    table = pd.DataFrame(
        [
            {
                "exposure_family": family,
                "window": window,
                "estimate": row["standardized_effect_display"],
                "lower": row["standardized_ci_lower_display"],
                "upper": row["standardized_ci_upper_display"],
            }
            for family, window, row in rows
        ]
    )
    windows = ["Current month", "Previous month", "Current + previous", "Future-month placebo"]
    colors = {"wb26": "#1D6F7A", "wbmean": "#8A5A9E"}
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(windows))[::-1]
    for family, offset in [("wb26", 0.10), ("wbmean", -0.10)]:
        subset = table.loc[table["exposure_family"].eq(family)].set_index("window").loc[windows]
        estimate = subset["estimate"].to_numpy()
        lower = subset["lower"].to_numpy()
        upper = subset["upper"].to_numpy()
        axis.errorbar(
            estimate,
            y + offset,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt="o",
            capsize=3,
            color=colors[family],
            label=EXPOSURES[family]["label"],
        )
    axis.axvline(0, color="#4D4D4D", linestyle="--", linewidth=1)
    axis.set_yticks(y, windows)
    axis.set_xlabel("Work participation effect (percentage points / 1 within-FE SD)")
    axis.legend(frameon=False, loc="lower left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_interactions(results: pd.DataFrame, output: Path) -> None:
    table = interaction_tests(results)
    order = [*MODIFIERS, *OCCUPATION_MODIFIERS]
    labels = [
        ({**MODIFIERS, **OCCUPATION_MODIFIERS}[item]["label"]) for item in order
    ]
    colors = {"wb26": "#1D6F7A", "wbmean": "#8A5A9E"}
    fig, axis = plt.subplots(figsize=(11.5, 7.2))
    y = np.arange(len(order))[::-1]
    for family, offset in [("wb26", 0.10), ("wbmean", -0.10)]:
        subset = table.loc[table["exposure_family"].eq(family)].set_index("modifier").loc[order]
        estimate = subset["standardized_effect_display"].to_numpy()
        lower = subset["standardized_ci_lower_display"].to_numpy()
        upper = subset["standardized_ci_upper_display"].to_numpy()
        axis.errorbar(
            estimate,
            y + offset,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt="o",
            capsize=3,
            color=colors[family],
            label=EXPOSURES[family]["label"],
        )
    axis.axvline(0, color="#4D4D4D", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Interaction effect (outcome units / 1 within-FE SD)")
    axis.legend(frameon=False, loc="lower left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cumulative_vulnerability(results: pd.DataFrame, output: Path) -> None:
    table = results.loc[
        results["analysis_domain"].eq("cumulative_heterogeneity")
        & results["is_target_test"]
    ].copy()
    table["modifier"] = table["term"].str.split("_x_", n=1).str[1]
    order = ["female", "low_education"]
    labels = ["Female differential", "Low-education differential"]
    colors = {"wb26": "#1D6F7A", "wbmean": "#8A5A9E"}
    fig, axis = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(order))[::-1]
    for family, offset in [("wb26", 0.10), ("wbmean", -0.10)]:
        subset = table.loc[table["exposure_family"].eq(family)].set_index("modifier").loc[order]
        estimate = subset["standardized_effect_display"].to_numpy()
        lower = subset["standardized_ci_lower_display"].to_numpy()
        upper = subset["standardized_ci_upper_display"].to_numpy()
        axis.errorbar(
            estimate,
            y + offset,
            xerr=np.vstack([estimate - lower, upper - estimate]),
            fmt="o",
            capsize=3,
            color=colors[family],
            label=EXPOSURES[family]["label"],
        )
    axis.axvline(0, color="#4D4D4D", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Cumulative-exposure interaction (percentage points / 1 within-FE SD)")
    axis.legend(frameon=False, loc="lower left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scorecard(scorecard: pd.DataFrame, output: Path) -> None:
    table = scorecard.sort_values("diagnostic_score", ascending=True)
    colors = table["evidence_grade"].map(
        {"family-supported": "#1D6F7A", "exploratory": "#D29B3A", "weak": "#A7B1B7"}
    )
    fig, axis = plt.subplots(figsize=(11.5, 7.0))
    axis.barh(table["candidate_storyline"], table["diagnostic_score"], color=colors)
    axis.axvline(7, color="#1D6F7A", linestyle="--", linewidth=1)
    axis.axvline(5, color="#D29B3A", linestyle="--", linewidth=1)
    axis.set_xlim(0, 10.2)
    axis.set_xlabel("Transparent diagnostic score (0-10; not a p-value)")
    axis.set_ylabel("")
    for index, value in enumerate(table["diagnostic_score"]):
        axis.text(value + 0.08, index, f"{value:.1f}", va="center", fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_scorecard_table(scorecard: pd.DataFrame, output: Path) -> None:
    display = scorecard.head(10).copy()
    display["Candidate Storyline"] = display["candidate_storyline"]
    display["Candidate Storyline"] = display["Candidate Storyline"].replace(
        {
            "Cumulative humid heat disproportionately affects low-education workers":
                "Cumulative humid heat x low-education vulnerability"
        }
    )
    display["Score"] = display["diagnostic_score"].map(lambda value: f"{value:.1f}/10")
    display["Grade"] = display["evidence_grade"].str.title()
    display["Min p"] = display["minimum_nominal_p"].map(lambda value: f"{value:.3f}")
    display["Min family q"] = display["minimum_family_q"].map(lambda value: f"{value:.3f}")
    display = display[["rank", "Candidate Storyline", "Score", "Grade", "Min p", "Min family q"]]
    display.columns = ["Rank", "Candidate Storyline", "Score", "Grade", "Min p", "Min family q"]

    fig, axis = plt.subplots(figsize=(15.2, 7.1))
    axis.axis("off")
    axis.add_patch(plt.Rectangle((0, 0.90), 1, 0.10, transform=axis.transAxes, color="#17324D"))
    axis.text(
        0.012,
        0.95,
        "Heat-Labor Storyline Search Scorecard",
        transform=axis.transAxes,
        va="center",
        color="white",
        fontsize=16,
        fontweight="bold",
    )
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.20, 1, 0.70],
        colWidths=[0.07, 0.43, 0.12, 0.12, 0.12, 0.14],
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
            if column == 1:
                cell.get_text().set_ha("left")
    notes = [
        "Notes: The score rewards direction consistency, replication across WB26 and continuous WBmax, stability/group support, placebo or interaction evidence, and multiplicity-adjusted signal.",
        "The score is a transparent screening aid, not an inferential statistic. All tested specifications remain in model_results.csv.",
        "Database access was read-only SELECT; writes: none. AnaSOP was not modified.",
    ]
    for index, note in enumerate(notes):
        axis.text(0.01, 0.14 - index * 0.045, note, transform=axis.transAxes, fontsize=8.5, color="#5F6B73")
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate(
    results: pd.DataFrame,
    registry: pd.DataFrame,
    slopes: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> dict[str, object]:
    if registry["status"].eq("failed").any():
        failed = registry.loc[registry["status"].eq("failed"), ["model_id", "error"]]
        raise AssertionError(f"Failed models:\n{failed.to_string(index=False)}")
    target = results.loc[results["is_target_test"]]
    if len(target) < 70:
        raise AssertionError(f"Specification universe unexpectedly small: {len(target)} target rows")
    if results.loc[results["role"].eq("primary"), "n_obs"].nunique() != 1:
        raise AssertionError("Primary exposure definitions do not use a common sample")
    if len(slopes) != 44:
        raise AssertionError(f"Expected 44 subgroup slopes, observed {len(slopes)}")
    if len(scorecard) != 12:
        raise AssertionError(f"Expected 12 storyline candidates, observed {len(scorecard)}")
    if target.loc[~target["analysis_domain"].eq("wave_stability"), "global_bh_q_value"].isna().any():
        raise AssertionError("Missing global multiplicity correction")
    bh_eligible = target.loc[~target["analysis_domain"].eq("wave_stability")]
    stability_diagnostics = target.loc[target["analysis_domain"].eq("wave_stability")]
    if len(bh_eligible) != 64 or len(stability_diagnostics) != 18:
        raise AssertionError(
            "Expected an 82-row audit with 64 BH-eligible tests and "
            "18 leave-one-wave-out stability diagnostics"
        )
    return {
        "planned_models": int(len(registry)),
        "estimated_models": int(registry["status"].eq("estimated").sum()),
        "result_rows": int(len(results)),
        "target_test_rows": int(len(target)),
        "bh_eligible_test_rows": int(len(bh_eligible)),
        "stability_diagnostic_rows": int(len(stability_diagnostics)),
        "subgroup_slope_rows": int(len(slopes)),
        "storyline_candidates": int(len(scorecard)),
        "primary_common_sample_n": int(results.loc[results["role"].eq("primary"), "n_obs"].iloc[0]),
    }


def write_readme(
    output: Path,
    scorecard: pd.DataFrame,
    validation: dict[str, object],
    results: pd.DataFrame,
) -> None:
    top = scorecard.head(3)
    minimum_global_q = float(
        results.loc[
            results["is_target_test"]
            & ~results["analysis_domain"].eq("wave_stability"),
            "global_bh_q_value",
        ].min()
    )
    lines = [
        "# Transparent heat-labor storyline search",
        "",
        "Status: exploratory diagnostic; not a formal AnaSOP output and not causal.",
        "",
        "## Search contract",
        "",
        "The search preserves all defensible timing, outcome, heterogeneity, adaptation-proxy, occupation, placebo, and wave-stability specifications. Benjamini-Hochberg q-values are reported globally and within test families. Candidate stories are ranked by a transparent diagnostic score rather than by the smallest p-value.",
        "",
        f"Estimated models: {validation['estimated_models']}; audit rows: {validation['target_test_rows']} ({validation['bh_eligible_test_rows']} BH-eligible substantive or exploratory tests and {validation['stability_diagnostic_rows']} leave-one-wave-out stability diagnostics); subgroup slopes: {validation['subgroup_slope_rows']}.",
        "",
        f"No BH-eligible test survives a 10 percent false-discovery threshold across the 64-test multiplicity universe; the minimum global q-value is {minimum_global_q:.3f}. The 18 leave-one-wave-out rows are stability diagnostics outside the BH calculation. The label `family-supported` means support after correction only within a prespecified test family and remains exploratory overall.",
        "",
        "## Leading candidates for human review",
        "",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- Rank {row.rank}: {row.candidate_storyline} — score {row.diagnostic_score:.1f}/10, "
            f"grade {row.evidence_grade}, minimum nominal p={row.minimum_nominal_p:.3f}, "
            f"minimum within-family q={row.minimum_family_q:.3f}. Caveat: {row.caveat}"
        )
    lines.extend(
        [
            "",
            "These rankings are screening evidence, not permission to discard lower-ranked or null results. The final story should be selected only after inspecting the timing, interaction, and full-model tables together.",
            "",
            "## Files",
            "",
            "- `specification_universe.csv`: every planned and estimated model.",
            "- `model_results.csv`: all coefficient rows, including nuisance terms.",
            "- `target_tests.csv`: 82 focal audit rows retained under the legacy filename; q-values are populated only for the 64 BH-eligible tests, while 18 leave-one-wave-out rows are stability diagnostics.",
            "- `subgroup_slopes.csv`: group-specific marginal heat slopes.",
            "- `storyline_scorecard.csv`: transparent candidate ranking.",
            "- `timing_and_placebo_search.png`, `heterogeneity_search.png`, `cumulative_vulnerability_search.png`, `storyline_scores.png`, and `Storyline_scorecard.png`: review graphics.",
            "",
            "Database access was read-only SELECT; database writes: none. AnaSOP was not modified.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(**connection_args(args)) as connection:
        frame = load_data(connection, args.schema)
        connection.rollback()
    frame = prepare_frame(frame)
    cells = build_cell_frame(frame)
    term_support = calculate_term_support(cells)
    results, registry, slopes = run_specification_universe(frame, term_support)
    results = apply_multiplicity(results)
    scorecard = score_storylines(results, slopes)

    registry.to_csv(output / "specification_universe.csv", index=False)
    results.to_csv(output / "model_results.csv", index=False)
    results.loc[results["is_target_test"]].to_csv(output / "target_tests.csv", index=False)
    slopes.to_csv(output / "subgroup_slopes.csv", index=False)
    term_support.to_csv(output / "exposure_term_support.csv", index=False)
    scorecard.to_csv(output / "storyline_scorecard.csv", index=False)
    plot_timing(results, output / "timing_and_placebo_search.png")
    plot_interactions(results, output / "heterogeneity_search.png")
    plot_cumulative_vulnerability(
        results, output / "cumulative_vulnerability_search.png"
    )
    plot_scorecard(scorecard, output / "storyline_scores.png")
    render_scorecard_table(scorecard, output / "Storyline_scorecard.png")
    validation = validate(results, registry, slopes, scorecard)
    write_readme(output, scorecard, validation, results)
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "exploratory diagnostic; not a formal AnaSOP output",
        "database_access": "read-only SELECT; writes none",
        "source_tables": [
            "final_HEAT_LABOR_ANALYTIC",
            "final_EC_CSES",
            "final_ED_CSES",
            "dim_admin2_cambodia",
        ],
        "code": "src/analyses/search_heat_labor_storylines.py",
        "validation": validation,
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
