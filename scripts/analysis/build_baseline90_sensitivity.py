"""Fresh-overlap sensitivity for the frozen 90-minute baseline threshold.

This script deliberately keeps the propensity-score ridge grid frozen in
``config/weighting.yml``.  A selected fallback candidate is still estimated and
reported when a scientific weighting gate is missed, but the failed gate is
never relabelled as a technical pipeline failure.

The primary 180-minute weights are read only for their lineage hash and are
never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "python"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_weighting_phase as weighting  # noqa: E402
from build_confirmatory_phase import (  # noqa: E402
    fit_hdfe,
    model_rows,
    weighted_summary_scale,
)


THRESHOLD = 90
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "robustness"

COHORT_PATH = PROCESSED / "analysis_sample_baseline_only.csv"
PANEL_PATH = PROCESSED / "design_player_match.csv"
PRIMARY_WEIGHTS_PATH = PROCESSED / "analysis_weights.csv"
WEIGHTS_OUT = PROCESSED / "analysis_weights_baseline90.csv"

WEIGHT_CONFIG = ROOT / "config" / "weighting.yml"
ROBUSTNESS_CONFIG = ROOT / "config" / "robustness.yml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(
        f"wrote {path.relative_to(ROOT)}: "
        f"{len(frame):,} rows x {len(frame.columns)} cols"
    )


def build_risk90(cohort: pd.DataFrame) -> pd.DataFrame:
    """Rebuild eligibility and same-nation support at the 90-minute cutoff."""

    cohort = cohort.copy()
    cohort["eligible_baseline_only_90"] = (
        cohort["outfield"].eq(True)
        & cohort["pre_games_ok"].eq(True)
        & cohort["pre_minutes_played"].fillna(0).ge(THRESHOLD)
        & cohort["baseline_identity_valid"].eq(True)
    )
    cohort["eligible_qualified_risk_set_90"] = (
        cohort["eligible_baseline_only_90"]
        & cohort["qualified_national_team"].eq(True)
    )

    support = (
        cohort.loc[cohort["eligible_qualified_risk_set_90"]]
        .groupby("national_team_eligible", dropna=False)["treated_world_cup"]
        .agg(
            nation_treated_count_90=lambda values: int(
                values.fillna(0).eq(1).sum()
            ),
            nation_control_count_90=lambda values: int(
                values.fillna(0).eq(0).sum()
            ),
        )
        .reset_index()
    )
    cohort = cohort.merge(
        support,
        on="national_team_eligible",
        how="left",
        validate="many_to_one",
    )
    cohort["eligible_same_nation_support_90"] = (
        cohort["eligible_qualified_risk_set_90"]
        & cohort["nation_treated_count_90"].fillna(0).gt(0)
        & cohort["nation_control_count_90"].fillna(0).gt(0)
    )

    risk = cohort.loc[cohort["eligible_qualified_risk_set_90"]].copy()
    # select_overlap_weights exports this frozen-schema column.  Its value must
    # be recomputed for the expanded risk set instead of copied from Phase 4.
    risk["eligible_same_nation_support"] = risk[
        "eligible_same_nation_support_90"
    ]
    risk["baseline_minutes_threshold"] = THRESHOLD
    return risk


def prepare_estimation_panel(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    estimation = panel.loc[
        panel["transfermarkt_player_id"].isin(
            weights["transfermarkt_player_id"]
        )
    ].merge(
        weights[["transfermarkt_player_id", "analysis_weight"]],
        on="transfermarkt_player_id",
        how="inner",
        validate="many_to_one",
    )
    estimation["club_game_id"] = (
        estimation["club_id"].astype(str)
        + "|"
        + estimation["game_id"].astype(str)
    )
    event_times = list(range(-8, 0)) + list(range(1, 9))
    return estimation.loc[
        estimation["relative_club_match"].isin(event_times)
    ].copy()


def fit_primary_sensitivity(
    estimation: pd.DataFrame,
    selected: pd.Series,
    risk: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    treatment = estimation["treated_world_cup"].to_numpy(float)
    acute = estimation["relative_club_match"].between(1, 4).to_numpy(float)
    consolidated = (
        estimation["relative_club_match"].between(5, 8).to_numpy(float)
    )
    scale, baseline = weighted_summary_scale(estimation, "minutes_share")
    fit = fit_hdfe(
        estimation,
        "minutes_share",
        {
            "treated_x_acute": treatment * acute,
            "treated_x_consolidated": treatment * consolidated,
        },
        ["transfermarkt_player_id", "club_game_id"],
    )
    rows = pd.DataFrame(
        model_rows(
            fit,
            "minutes_share",
            "baseline_minutes_ge_90_fresh_overlap",
            scale,
            baseline,
            0.05,
        )
    )
    weight_gate_passed = bool(selected["passes_selection_gates"])
    rows["baseline_minutes_threshold"] = THRESHOLD
    rows["selected_ridge"] = float(selected["ridge"])
    rows["weight_gate_passed"] = weight_gate_passed
    rows["n_treated"] = int(risk["treated_world_cup"].eq(1).sum())
    rows["n_controls"] = int(risk["treated_world_cup"].eq(0).sum())
    rows["interpretation"] = (
        "sensitivity_with_weighting_gate_passed"
        if weight_gate_passed
        else "diagnostic_sensitivity_weighting_gate_not_met"
    )
    return rows, fit


def scientific_gate_summary(
    selected: pd.Series,
    balance: pd.DataFrame,
    pretrend_joint: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    selection_gates = config["selection_gates"]
    holdout = pretrend_joint.loc[
        pretrend_joint["outcome"].eq("minutes_share")
        & pretrend_joint["diagnostic_window"].eq("holdout_window")
    ]
    holdout_present = len(holdout) == 1
    holdout_equivalence = bool(
        holdout_present
        and holdout.iloc[0]["all_pointwise_equivalence_passed"]
    )
    holdout_joint_p = (
        float(holdout.iloc[0]["simulation_p_value"])
        if holdout_present
        else np.nan
    )

    variance_ratios = (
        pd.to_numeric(balance["variance_ratio_after"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    max_abs_smd = float(balance["smd_after"].abs().max())
    max_ks = float(balance["ks_after"].max())
    minimum_variance_ratio = (
        float(variance_ratios.min()) if len(variance_ratios) else np.nan
    )
    maximum_variance_ratio = (
        float(variance_ratios.max()) if len(variance_ratios) else np.nan
    )
    weight_gate_passed = bool(selected["passes_selection_gates"])
    strong_language_ready = weight_gate_passed and holdout_equivalence

    rows = [
        {
            "gate": "frozen_overlap_candidate_passes_all_gates",
            "observed": weight_gate_passed,
            "expected": True,
            "passed": weight_gate_passed,
            "role": "weighting_diagnostic",
        },
        {
            "gate": "treated_ess_fraction",
            "observed": float(selected["treated_ess_fraction"]),
            "expected": (
                f">={selection_gates['min_ess_fraction_each_group']}"
            ),
            "passed": float(selected["treated_ess_fraction"])
            >= float(selection_gates["min_ess_fraction_each_group"]),
            "role": "weighting_diagnostic",
        },
        {
            "gate": "control_ess_fraction",
            "observed": float(selected["control_ess_fraction"]),
            "expected": (
                f">={selection_gates['min_ess_fraction_each_group']}"
            ),
            "passed": float(selected["control_ess_fraction"])
            >= float(selection_gates["min_ess_fraction_each_group"]),
            "role": "weighting_diagnostic",
        },
        {
            "gate": "treated_support_fraction",
            "observed": float(selected["treated_support_fraction"]),
            "expected": (
                f">={selection_gates['min_treated_support_fraction']}"
            ),
            "passed": float(selected["treated_support_fraction"])
            >= float(selection_gates["min_treated_support_fraction"]),
            "role": "weighting_diagnostic",
        },
        {
            "gate": "maximum_normalized_weight",
            "observed": float(selected["max_normalized_weight"]),
            "expected": f"<={selection_gates['max_normalized_weight']}",
            "passed": float(selected["max_normalized_weight"])
            <= float(selection_gates["max_normalized_weight"]),
            "role": "weighting_diagnostic",
        },
        {
            "gate": "all_diagnostic_max_abs_smd",
            "observed": max_abs_smd,
            "expected": "<=0.10",
            "passed": max_abs_smd <= 0.10,
            "role": "balance_diagnostic",
        },
        {
            "gate": "all_diagnostic_max_ks",
            "observed": max_ks,
            "expected": "<=0.10",
            "passed": max_ks <= 0.10,
            "role": "balance_diagnostic",
        },
        {
            "gate": "all_diagnostic_variance_ratio_range",
            "observed": (
                f"{minimum_variance_ratio:.12g}--"
                f"{maximum_variance_ratio:.12g}"
            ),
            "expected": "0.50--2.00",
            "passed": bool(
                len(variance_ratios)
                and minimum_variance_ratio >= 0.50
                and maximum_variance_ratio <= 2.00
            ),
            "role": "balance_diagnostic",
        },
        {
            "gate": "holdout_joint_p_above_0_05",
            "observed": holdout_joint_p,
            "expected": ">=0.05",
            "passed": bool(
                holdout_present and holdout_joint_p >= 0.05
            ),
            "role": "diagnostic_not_proof",
        },
        {
            "gate": "holdout_pointwise_equivalence",
            "observed": holdout_equivalence,
            "expected": True,
            "passed": holdout_equivalence,
            "role": "required_for_strong_causal_language",
        },
        {
            "gate": "results_reportable_as_diagnostic_sensitivity",
            "observed": True,
            "expected": True,
            "passed": True,
            "role": "reporting_decision",
        },
        {
            "gate": "strong_causal_language_ready",
            "observed": strong_language_ready,
            "expected": True,
            "passed": strong_language_ready,
            "role": "reporting_decision",
        },
    ]
    return pd.DataFrame(rows)


def technical_qa(
    risk: pd.DataFrame,
    panel: pd.DataFrame,
    estimation: pd.DataFrame,
    weights: pd.DataFrame,
    candidates: pd.DataFrame,
    pretrend: pd.DataFrame,
    pretrend_joint: pd.DataFrame,
    results: pd.DataFrame,
    fit: dict[str, object],
    primary_weights_hash_before: str,
    primary_weights_hash_after: str,
) -> pd.DataFrame:
    treatment_values = set(
        pd.to_numeric(risk["treated_world_cup"], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    strata_group_counts = risk.groupby("weighting_stratum")[
        "treated_world_cup"
    ].nunique()
    risk_ids = set(risk["transfermarkt_player_id"])
    weight_ids = set(weights["transfermarkt_player_id"])
    estimation_ids = set(estimation["transfermarkt_player_id"])
    observations_per_player = estimation.groupby(
        "transfermarkt_player_id"
    ).size()
    finite_result_columns = [
        "estimate",
        "standard_error_player",
        "standard_error_double_cluster",
    ]
    tests = [
        (
            "output_path_does_not_replace_primary_weights",
            str(WEIGHTS_OUT.relative_to(ROOT)),
            f"!={PRIMARY_WEIGHTS_PATH.relative_to(ROOT)}",
            WEIGHTS_OUT.resolve() != PRIMARY_WEIGHTS_PATH.resolve(),
        ),
        (
            "risk_set_unique_player",
            len(risk),
            risk["transfermarkt_player_id"].nunique(),
            risk["transfermarkt_player_id"].is_unique,
        ),
        (
            "treatment_binary_and_complete",
            sorted(treatment_values),
            "[0.0, 1.0]",
            treatment_values == {0.0, 1.0}
            and risk["treated_world_cup"].notna().all(),
        ),
        (
            "both_treatment_groups_present",
            int(risk["treated_world_cup"].nunique()),
            2,
            risk["treated_world_cup"].nunique() == 2,
        ),
        (
            "all_strata_have_both_groups",
            int(strata_group_counts.eq(2).sum()),
            len(strata_group_counts),
            bool(strata_group_counts.eq(2).all()),
        ),
        (
            "weights_unique_player",
            len(weights),
            weights["transfermarkt_player_id"].nunique(),
            weights["transfermarkt_player_id"].is_unique,
        ),
        (
            "weight_ids_equal_risk_ids",
            len(weight_ids.symmetric_difference(risk_ids)),
            0,
            weight_ids == risk_ids,
        ),
        (
            "one_selected_candidate",
            int(candidates["selected"].sum()),
            1,
            candidates["selected"].sum() == 1,
        ),
        (
            "selected_propensity_model_converged",
            int(candidates.loc[candidates["selected"], "model_converged"].sum()),
            1,
            bool(
                candidates.loc[
                    candidates["selected"], "model_converged"
                ].iloc[0]
            ),
        ),
        (
            "selected_calibration_converged",
            int(
                candidates.loc[
                    candidates["selected"], "calibration_converged"
                ].sum()
            ),
            1,
            bool(
                candidates.loc[
                    candidates["selected"], "calibration_converged"
                ].iloc[0]
            ),
        ),
        (
            "finite_positive_analysis_weights",
            int(np.isfinite(weights["analysis_weight"]).sum()),
            len(weights),
            bool(
                np.isfinite(weights["analysis_weight"]).all()
                and weights["analysis_weight"].gt(0).all()
            ),
        ),
        (
            "propensity_strictly_between_zero_and_one",
            int(
                weights["propensity_score"].between(
                    0, 1, inclusive="neither"
                ).sum()
            ),
            len(weights),
            bool(
                weights["propensity_score"]
                .between(0, 1, inclusive="neither")
                .all()
            ),
        ),
        (
            "estimation_ids_equal_weight_ids",
            len(estimation_ids.symmetric_difference(weight_ids)),
            0,
            estimation_ids == weight_ids,
        ),
        (
            "sixteen_rows_per_player",
            (
                f"{int(observations_per_player.min())}--"
                f"{int(observations_per_player.max())}"
            ),
            "16--16",
            bool(observations_per_player.eq(16).all()),
        ),
        (
            "unique_player_relative_match",
            int(
                estimation.duplicated(
                    ["transfermarkt_player_id", "relative_club_match"]
                ).sum()
            ),
            0,
            not estimation.duplicated(
                ["transfermarkt_player_id", "relative_club_match"]
            ).any(),
        ),
        (
            "model_uses_all_weighted_players",
            int(fit["players"]),
            len(weights),
            int(fit["players"]) == len(weights),
        ),
        (
            "hdfe_converged",
            float(fit["fe_error"]),
            "<1e-8",
            float(fit["fe_error"]) < 1e-8,
        ),
        (
            "finite_model_results",
            int(np.isfinite(results[finite_result_columns]).sum().sum()),
            results[finite_result_columns].size,
            bool(np.isfinite(results[finite_result_columns]).all().all()),
        ),
        (
            "pretrend_uses_pre_only",
            int(pretrend["uses_post_treatment_outcomes"].eq(0).sum()),
            len(pretrend),
            bool(
                len(pretrend)
                and pretrend["uses_post_treatment_outcomes"].eq(0).all()
            ),
        ),
        (
            "primary_holdout_pretrend_present",
            int(
                (
                    pretrend_joint["outcome"].eq("minutes_share")
                    & pretrend_joint["diagnostic_window"].eq(
                        "holdout_window"
                    )
                ).sum()
            ),
            1,
            (
                pretrend_joint["outcome"].eq("minutes_share")
                & pretrend_joint["diagnostic_window"].eq("holdout_window")
            ).sum()
            == 1,
        ),
        (
            "source_panel_contains_all_risk_players",
            len(
                risk_ids.difference(
                    set(panel["transfermarkt_player_id"].unique())
                )
            ),
            0,
            risk_ids.issubset(
                set(panel["transfermarkt_player_id"].unique())
            ),
        ),
        (
            "primary_weights_hash_unchanged",
            primary_weights_hash_after,
            primary_weights_hash_before,
            primary_weights_hash_after == primary_weights_hash_before,
        ),
    ]
    return pd.DataFrame(
        tests,
        columns=["check", "observed", "expected", "passed"],
    )


def write_manifest(
    paths: list[Path],
    hashes: dict[str, str],
) -> Path:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        with path.open("rb") as stream:
            row_count = max(sum(1 for _ in stream) - 1, 0)
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "rows": row_count,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                **hashes,
                "generated_by": (
                    "scripts/analysis/build_baseline90_sensitivity.py"
                ),
            }
        )
    manifest_path = REPORTS / "baseline90_manifest.csv"
    write_csv(pd.DataFrame(rows), manifest_path)
    return manifest_path


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = weighting.load_json_yaml(WEIGHT_CONFIG)
    robustness = load_json(ROBUSTNESS_CONFIG)
    if THRESHOLD not in [
        int(value) for value in robustness["baseline_minutes_sensitivity"]
    ]:
        raise RuntimeError(
            "O limiar de 90 minutos nao esta no protocolo congelado."
        )

    hashes = {
        "weighting_config_sha256": file_sha256(WEIGHT_CONFIG),
        "robustness_config_sha256": file_sha256(ROBUSTNESS_CONFIG),
        "input_cohort_sha256": file_sha256(COHORT_PATH),
        "input_panel_sha256": file_sha256(PANEL_PATH),
        "primary_weights_sha256": file_sha256(PRIMARY_WEIGHTS_PATH),
    }
    primary_weights_hash_before = hashes["primary_weights_sha256"]

    cohort, panel = weighting.read_inputs()
    weighting.validate_input_lineage(cohort)
    risk = build_risk90(cohort)
    risk, history_columns = weighting.add_history_covariates(
        risk, panel, config
    )
    risk = weighting.add_strata(risk, config)
    risk, missing_columns = weighting.impute_covariates(
        risk, config, history_columns
    )

    weights, candidates, balance, coefficients, _ = (
        weighting.select_overlap_weights(risk, config, missing_columns)
    )
    if "fallback_score" not in candidates:
        candidates["fallback_score"] = np.nan
    selected = candidates.loc[candidates["selected"]].iloc[0]

    weight_summary = weighting.summarize_weights(
        risk,
        pd.Series(
            weights["analysis_weight"].to_numpy(), index=risk.index
        ),
        pd.Series(
            weights["propensity_score"].to_numpy(), index=risk.index
        ),
        config,
        "calibrated_overlap_baseline90",
    )
    pretrend, pretrend_joint = weighting.pretrend_diagnostics(
        panel, weights, config
    )
    estimation = prepare_estimation_panel(panel, weights)
    results, fit = fit_primary_sensitivity(estimation, selected, risk)
    gates = scientific_gate_summary(
        selected, balance, pretrend_joint, config
    )

    primary_weights_hash_after = file_sha256(PRIMARY_WEIGHTS_PATH)
    qa = technical_qa(
        risk,
        panel,
        estimation,
        weights,
        candidates,
        pretrend,
        pretrend_joint,
        results,
        fit,
        primary_weights_hash_before,
        primary_weights_hash_after,
    )

    weights["baseline_minutes_threshold"] = THRESHOLD
    weights["weight_gate_passed"] = bool(
        selected["passes_selection_gates"]
    )
    for name, value in hashes.items():
        weights[name] = value

    outputs = {
        WEIGHTS_OUT: weights,
        REPORTS / "baseline90_candidate_weighting_diagnostics.csv": candidates,
        REPORTS / "baseline90_balance_diagnostics.csv": balance,
        REPORTS / "baseline90_weight_distribution.csv": weight_summary,
        REPORTS / "baseline90_propensity_coefficients.csv": coefficients,
        REPORTS / "baseline90_pretrend_coefficients.csv": pretrend,
        REPORTS / "baseline90_pretrend_joint_tests.csv": pretrend_joint,
        REPORTS / "baseline90_coefficients.csv": results,
        REPORTS / "baseline90_gate_summary.csv": gates,
        REPORTS / "baseline90_qa.csv": qa,
    }
    for path, frame in outputs.items():
        write_csv(frame, path)
    manifest_path = write_manifest(list(outputs), hashes)
    print(
        f"wrote {manifest_path.relative_to(ROOT)}: "
        f"{manifest_path.stat().st_size:,} bytes"
    )

    if not qa["passed"].all():
        failures = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(
            f"QA tecnico da sensibilidade de 90 minutos falhou: {failures}"
        )

    failed_scientific = gates.loc[~gates["passed"], "gate"].tolist()
    print(
        "Sensibilidade de 90 minutos concluida com pesos novos. "
        f"Gates cientificos nao atendidos: {failed_scientific}"
    )


if __name__ == "__main__":
    main()
