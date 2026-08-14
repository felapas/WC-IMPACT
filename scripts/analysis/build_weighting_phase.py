"""Fase 5: propensity score, overlap weights, balance e pré-tendências.

Todas as escolhas de pesos usam exclusivamente covariáveis anteriores à Copa.
O script não estima efeitos confirmatórios com outcomes pós-tratamento.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "python"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DESIGN_CONFIG = ROOT / "config" / "analysis.yml"
WEIGHT_CONFIG = ROOT / "config" / "weighting.yml"
OVERRIDES = ROOT / "config" / "manual_design_overrides.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "weighting"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_yaml(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(PROCESSED / "analysis_sample_baseline_only.csv", low_memory=False)
    panel = pd.read_csv(PROCESSED / "design_player_match.csv", low_memory=False)
    panel["match_date"] = pd.to_datetime(panel["match_date"])
    return cohort, panel


def validate_input_lineage(cohort: pd.DataFrame) -> None:
    expected_design = file_sha256(DESIGN_CONFIG)
    expected_overrides = file_sha256(OVERRIDES)
    design_values = set(cohort["analysis_config_sha256"].dropna().astype(str))
    override_values = set(cohort["manual_overrides_sha256"].dropna().astype(str))
    if design_values != {expected_design}:
        raise RuntimeError("Coorte da Fase 4 não corresponde ao config/analysis.yml atual")
    if override_values != {expected_overrides}:
        raise RuntimeError("Coorte da Fase 4 não corresponde aos overrides atuais")


def add_strata(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    frame = frame.copy()
    frame["weighting_stratum"] = frame[config["strata"]].astype(str).agg("|".join, axis=1)
    return frame


def add_history_covariates(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    history_columns: list[str] = []
    specification = config["history_balance"]
    selected_panel = panel[
        panel["transfermarkt_player_id"].isin(frame["transfermarkt_player_id"])
        & panel["relative_club_match"].isin(specification["event_times"])
    ]
    for outcome in specification["outcomes"]:
        wide = selected_panel.pivot_table(
            index="transfermarkt_player_id",
            columns="relative_club_match",
            values=outcome,
            aggfunc="first",
        )
        rename = {
            event_time: f"history_{outcome}_m{abs(int(event_time))}"
            for event_time in specification["event_times"]
        }
        wide = wide.rename(columns=rename).reset_index()
        history_columns.extend(rename.values())
        frame = frame.merge(wide, on="transfermarkt_player_id", how="left", validate="one_to_one")
    return frame, history_columns


def impute_covariates(
    frame: pd.DataFrame,
    config: dict[str, object],
    history_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    missing_columns: list[str] = []
    for column in config["continuous_covariates"]:
        missing_name = f"missing_{column}"
        frame[missing_name] = frame[column].isna().astype("int8")
        if frame[missing_name].any():
            missing_columns.append(missing_name)
        medians = frame.groupby("weighting_stratum")[column].transform("median")
        frame[f"imputed_{column}"] = frame[column].fillna(medians).fillna(frame[column].median())
    for column in history_columns:
        medians = frame.groupby("weighting_stratum")[column].transform("median")
        frame[f"imputed_{column}"] = frame[column].fillna(medians).fillna(frame[column].median())
    return frame, missing_columns


def model_matrix(
    group: pd.DataFrame,
    config: dict[str, object],
    missing_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = [np.ones(len(group))]
    names = ["intercept"]
    for column in config["continuous_covariates"]:
        values = group[f"imputed_{column}"].astype(float)
        standard_deviation = values.std(ddof=0)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            standard_deviation = 1.0
        arrays.append(((values - values.mean()) / standard_deviation).to_numpy())
        names.append(column)
    for column in missing_columns:
        if group[column].nunique() > 1:
            arrays.append(group[column].astype(float).to_numpy())
            names.append(column)
    return np.column_stack(arrays), names


def fit_ridge_logistic(
    x: np.ndarray,
    treatment: np.ndarray,
    ridge: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    converged = False
    for iteration in range(1, max_iterations + 1):
        linear = np.clip(x @ beta, -30, 30)
        propensity = 1 / (1 + np.exp(-linear))
        variance = np.maximum(propensity * (1 - propensity), 1e-8)
        gradient = x.T @ (treatment - propensity) - ridge * penalty @ beta
        hessian = (x.T * variance) @ x + ridge * penalty
        step = np.linalg.solve(
            hessian + 1e-10 * np.eye(hessian.shape[0]), gradient
        )
        beta += step
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
    propensity = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
    return beta, propensity, converged, iteration


def estimate_propensity(
    frame: pd.DataFrame,
    config: dict[str, object],
    missing_columns: list[str],
    ridge: float,
) -> tuple[pd.Series, pd.DataFrame, bool, int]:
    propensity = pd.Series(index=frame.index, dtype=float)
    coefficients: list[dict[str, object]] = []
    convergence: list[bool] = []
    iterations: list[int] = []
    specification = config["propensity"]
    for stratum, group in frame.groupby("weighting_stratum", sort=True):
        x, names = model_matrix(group, config, missing_columns)
        treatment = group[config["treatment"]].to_numpy(float)
        beta, fitted, converged, iteration = fit_ridge_logistic(
            x,
            treatment,
            ridge,
            int(specification["max_iterations"]),
            float(specification["tolerance"]),
        )
        fitted = np.clip(
            fitted,
            float(specification["probability_clip"]),
            1 - float(specification["probability_clip"]),
        )
        propensity.loc[group.index] = fitted
        convergence.append(converged)
        iterations.append(iteration)
        for name, value in zip(names, beta):
            coefficients.append(
                {
                    "ridge": ridge,
                    "weighting_stratum": stratum,
                    "feature": name,
                    "coefficient": value,
                    "converged": converged,
                    "iterations": iteration,
                }
            )
    return propensity, pd.DataFrame(coefficients), all(convergence), max(iterations)


def calibration_matrix(
    frame: pd.DataFrame,
    config: dict[str, object],
    covariates: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []
    calibration_covariates = covariates if covariates is not None else config["continuous_covariates"]
    for column in calibration_covariates:
        values = frame[f"imputed_{column}"].astype(float)
        standard_deviation = values.std(ddof=0)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            standard_deviation = 1.0
        arrays.append(((values - values.mean()) / standard_deviation).to_numpy())
        names.append(column)
    if covariates is None:
        for column in [name for name in frame.columns if name.startswith("history_") and not name.startswith("history_missing_")]:
            values = frame[f"imputed_{column}"].astype(float)
            standard_deviation = values.std(ddof=0)
            if not np.isfinite(standard_deviation) or standard_deviation == 0:
                standard_deviation = 1.0
            arrays.append(((values - values.mean()) / standard_deviation).to_numpy())
            names.append(column)
    strata = pd.get_dummies(
        frame["weighting_stratum"], prefix="stratum", dtype=float, drop_first=True
    )
    if not strata.empty:
        arrays.extend(strata.to_numpy().T)
        names.extend(strata.columns.tolist())
    return np.column_stack(arrays), names


def entropy_tilt(
    x: np.ndarray,
    base_weights: np.ndarray,
    target: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, bool, int, float]:
    coefficients = np.zeros(x.shape[1])
    converged = False
    maximum_error = np.inf
    for iteration in range(1, max_iterations + 1):
        linear = np.clip(x @ coefficients, -30, 30)
        weights = base_weights * np.exp(linear)
        weights /= weights.sum()
        mean = weights @ x
        gradient = mean - target
        maximum_error = float(np.max(np.abs(gradient)))
        if maximum_error < tolerance:
            converged = True
            break
        centered = x - mean
        hessian = (centered.T * weights) @ centered
        step = np.linalg.solve(
            hessian + 1e-8 * np.eye(hessian.shape[0]), gradient
        )
        coefficients -= step
    linear = np.clip(x @ coefficients, -30, 30)
    weights = base_weights * np.exp(linear)
    return weights, converged, iteration, maximum_error


def calibrate_overlap_weights(
    frame: pd.DataFrame,
    base_weights: pd.Series,
    config: dict[str, object],
) -> tuple[pd.Series, bool, float]:
    matrix, _ = calibration_matrix(frame, config)
    target = np.average(matrix, axis=0, weights=base_weights)
    calibrated = pd.Series(index=frame.index, dtype=float)
    convergence: list[bool] = []
    errors: list[float] = []
    calibration = config["calibration"]
    treatment = frame[config["treatment"]].astype(int)
    for group_value in [1, 0]:
        selector = treatment.eq(group_value)
        weights, converged, _, error = entropy_tilt(
            matrix[selector.to_numpy()],
            base_weights.loc[selector].to_numpy(float),
            target,
            float(calibration["tolerance"]),
            int(calibration["max_iterations"]),
        )
        calibrated.loc[selector] = weights
        convergence.append(converged)
        errors.append(error)
    return calibrated, all(convergence), max(errors)


def equal_group_normalization(
    weights: pd.Series, treatment: pd.Series
) -> pd.Series:
    result = weights.copy().astype(float)
    target_sum = len(weights) / 2
    for group_value in [1, 0]:
        selector = treatment.eq(group_value)
        result.loc[selector] *= target_sum / result.loc[selector].sum()
    return result


def effective_sample_size(weights: pd.Series) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()
    return float(np.interp(probability, cumulative, sorted_values))


def weighted_ks(
    values_treated: np.ndarray,
    weights_treated: np.ndarray,
    values_control: np.ndarray,
    weights_control: np.ndarray,
) -> float:
    values = np.sort(np.unique(np.concatenate([values_treated, values_control])))
    treated_order = np.argsort(values_treated)
    control_order = np.argsort(values_control)
    treated_values = values_treated[treated_order]
    control_values = values_control[control_order]
    treated_cdf = np.cumsum(weights_treated[treated_order]) / weights_treated.sum()
    control_cdf = np.cumsum(weights_control[control_order]) / weights_control.sum()
    treated_at = np.where(
        np.searchsorted(treated_values, values, side="right") > 0,
        treated_cdf[np.maximum(np.searchsorted(treated_values, values, side="right") - 1, 0)],
        0,
    )
    control_at = np.where(
        np.searchsorted(control_values, values, side="right") > 0,
        control_cdf[np.maximum(np.searchsorted(control_values, values, side="right") - 1, 0)],
        0,
    )
    return float(np.max(np.abs(treated_at - control_at)))


def balance_for_variable(
    values: pd.Series,
    treatment: pd.Series,
    weights: pd.Series,
) -> dict[str, float]:
    values = values.astype(float)
    treated = treatment.eq(1)
    control = ~treated
    treated_mean = float(np.average(values[treated], weights=weights[treated]))
    control_mean = float(np.average(values[control], weights=weights[control]))
    treated_variance = float(
        np.average(np.square(values[treated] - treated_mean), weights=weights[treated])
    )
    control_variance = float(
        np.average(np.square(values[control] - control_mean), weights=weights[control])
    )
    pooled = math.sqrt(max((treated_variance + control_variance) / 2, 1e-15))
    return {
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "smd": (treated_mean - control_mean) / pooled,
        "variance_ratio": treated_variance / control_variance if control_variance > 0 else np.nan,
        "ks": weighted_ks(
            values[treated].to_numpy(),
            weights[treated].to_numpy(),
            values[control].to_numpy(),
            weights[control].to_numpy(),
        ),
    }


def build_balance_table(
    frame: pd.DataFrame,
    weights: pd.Series,
    config: dict[str, object],
    missing_columns: list[str],
    method: str,
) -> pd.DataFrame:
    treatment = frame[config["treatment"]].astype(int)
    unweighted = pd.Series(1.0, index=frame.index)
    variables: list[tuple[str, pd.Series, str, bool]] = []
    for column in config["continuous_covariates"]:
        variables.append(
            (
                column,
                frame[f"imputed_{column}"],
                "continuous",
                column in set(config["key_covariates"]),
            )
        )
    for column in missing_columns:
        variables.append((column, frame[column], "missing_indicator", False))
    for column in [name for name in frame.columns if name.startswith("history_") and not name.startswith("history_missing_")]:
        variables.append((column, frame[f"imputed_{column}"], "prehistory", True))
    strata = pd.get_dummies(frame["weighting_stratum"], prefix="stratum", dtype=float)
    for column in strata.columns:
        variables.append((column, strata[column], "stratum_indicator", False))
    rows: list[dict[str, object]] = []
    for variable, values, variable_type, key in variables:
        before = balance_for_variable(values, treatment, unweighted)
        after = balance_for_variable(values, treatment, weights)
        rows.append(
            {
                "method": method,
                "variable": variable,
                "variable_type": variable_type,
                "key_covariate": key,
                "treated_mean_before": before["treated_mean"],
                "control_mean_before": before["control_mean"],
                "smd_before": before["smd"],
                "variance_ratio_before": before["variance_ratio"],
                "ks_before": before["ks"],
                "treated_mean_after": after["treated_mean"],
                "control_mean_after": after["control_mean"],
                "smd_after": after["smd"],
                "variance_ratio_after": after["variance_ratio"],
                "ks_after": after["ks"],
            }
        )
    return pd.DataFrame(rows)


def summarize_weights(
    frame: pd.DataFrame,
    weights: pd.Series,
    propensity: pd.Series,
    config: dict[str, object],
    method: str,
) -> pd.DataFrame:
    treatment = frame[config["treatment"]].astype(int)
    rows: list[dict[str, object]] = []
    for group_value, label in [(1, "treated"), (0, "control")]:
        selector = treatment.eq(group_value)
        group_weights = weights[selector]
        normalized = group_weights / group_weights.mean()
        group_propensity = propensity[selector]
        rows.append(
            {
                "method": method,
                "group": label,
                "n": int(selector.sum()),
                "weight_sum": group_weights.sum(),
                "ess": effective_sample_size(group_weights),
                "ess_fraction": effective_sample_size(group_weights) / selector.sum(),
                "weight_min_normalized": normalized.min(),
                "weight_median_normalized": normalized.median(),
                "weight_p95_normalized": normalized.quantile(0.95),
                "weight_p99_normalized": normalized.quantile(0.99),
                "weight_max_normalized": normalized.max(),
                "propensity_min": group_propensity.min(),
                "propensity_p05": group_propensity.quantile(0.05),
                "propensity_median": group_propensity.median(),
                "propensity_p95": group_propensity.quantile(0.95),
                "propensity_max": group_propensity.max(),
            }
        )
    return pd.DataFrame(rows)


def candidate_diagnostics(
    frame: pd.DataFrame,
    propensity: pd.Series,
    weights: pd.Series,
    balance: pd.DataFrame,
    config: dict[str, object],
    ridge: float,
    model_converged: bool,
    calibration_converged: bool,
    calibration_error: float,
    max_iterations: int,
) -> dict[str, object]:
    continuous = balance[balance["variable_type"].isin(["continuous", "prehistory"])]
    key = continuous[continuous["key_covariate"]]
    summary = summarize_weights(frame, weights, propensity, config, "candidate")
    treated_summary = summary[summary["group"].eq("treated")].iloc[0]
    control_summary = summary[summary["group"].eq("control")].iloc[0]
    lower = float(config["propensity"]["common_support_lower"])
    upper = float(config["propensity"]["common_support_upper"])
    treated = frame[config["treatment"]].eq(1)
    support_fraction = propensity[treated].between(lower, upper).mean()
    maximum_weight = max(
        treated_summary["weight_max_normalized"],
        control_summary["weight_max_normalized"],
    )
    gates = config["selection_gates"]
    maximum_smd = continuous["smd_after"].abs().max()
    maximum_key_smd = key["smd_after"].abs().max()
    passed = (
        model_converged
        and calibration_converged
        and maximum_smd <= float(gates["max_abs_smd"])
        and maximum_key_smd <= float(gates["max_abs_smd_key"])
        and treated_summary["ess_fraction"] >= float(gates["min_ess_fraction_each_group"])
        and control_summary["ess_fraction"] >= float(gates["min_ess_fraction_each_group"])
        and support_fraction >= float(gates["min_treated_support_fraction"])
        and maximum_weight <= float(gates["max_normalized_weight"])
    )
    return {
        "ridge": ridge,
        "model_converged": model_converged,
        "calibration_converged": calibration_converged,
        "calibration_max_error": calibration_error,
        "max_model_iterations": max_iterations,
        "max_abs_smd": maximum_smd,
        "max_abs_smd_key": maximum_key_smd,
        "treated_ess": treated_summary["ess"],
        "control_ess": control_summary["ess"],
        "treated_ess_fraction": treated_summary["ess_fraction"],
        "control_ess_fraction": control_summary["ess_fraction"],
        "treated_support_fraction": support_fraction,
        "max_normalized_weight": maximum_weight,
        "passes_selection_gates": passed,
    }


def select_overlap_weights(
    frame: pd.DataFrame,
    config: dict[str, object],
    missing_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    treatment = frame[config["treatment"]].astype(int)
    candidate_rows: list[dict[str, object]] = []
    candidate_objects: dict[float, dict[str, object]] = {}
    all_coefficients: list[pd.DataFrame] = []
    for ridge in [float(value) for value in config["propensity"]["ridge_grid"]]:
        propensity, coefficients, model_converged, iterations = estimate_propensity(
            frame, config, missing_columns, ridge
        )
        raw_overlap = pd.Series(
            np.where(treatment.eq(1), 1 - propensity, propensity), index=frame.index
        )
        calibrated, calibration_converged, calibration_error = calibrate_overlap_weights(
            frame, raw_overlap, config
        )
        normalized = equal_group_normalization(calibrated, treatment)
        balance = build_balance_table(
            frame, normalized, config, missing_columns, f"calibrated_overlap_ridge_{ridge:g}"
        )
        diagnostics = candidate_diagnostics(
            frame,
            propensity,
            normalized,
            balance,
            config,
            ridge,
            model_converged,
            calibration_converged,
            calibration_error,
            iterations,
        )
        candidate_rows.append(diagnostics)
        coefficients["selected"] = False
        all_coefficients.append(coefficients)
        candidate_objects[ridge] = {
            "propensity": propensity,
            "raw_overlap": raw_overlap,
            "calibrated": calibrated,
            "normalized": normalized,
            "balance": balance,
        }
    diagnostics = pd.DataFrame(candidate_rows).sort_values("ridge").reset_index(drop=True)
    passing = diagnostics[diagnostics["passes_selection_gates"]]
    if not passing.empty:
        selected_ridge = float(passing.sort_values("ridge").iloc[0]["ridge"])
    else:
        diagnostics["fallback_score"] = (
            diagnostics["max_abs_smd"]
            + diagnostics["max_abs_smd_key"]
            + np.maximum(0, 0.5 - diagnostics["treated_ess_fraction"])
            + np.maximum(0, 0.5 - diagnostics["control_ess_fraction"])
            + np.maximum(0, 0.8 - diagnostics["treated_support_fraction"])
        )
        selected_ridge = float(diagnostics.sort_values("fallback_score").iloc[0]["ridge"])
    diagnostics["selected"] = diagnostics["ridge"].eq(selected_ridge)
    selected = candidate_objects[selected_ridge]
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    coefficients.loc[coefficients["ridge"].eq(selected_ridge), "selected"] = True
    weights = frame[[
        "transfermarkt_player_id", "player_name", "baseline_club_id", "baseline_club_name",
        "league_name", "position_group", "weighting_stratum", "national_team_eligible",
        "eligible_same_nation_support", "treated_world_cup",
    ]].copy()
    weights["propensity_score"] = selected["propensity"]
    weights["overlap_weight_raw"] = selected["raw_overlap"]
    weights["overlap_weight_calibrated"] = selected["calibrated"]
    weights["analysis_weight"] = selected["normalized"]
    lower = float(config["propensity"]["common_support_lower"])
    upper = float(config["propensity"]["common_support_upper"])
    weights["common_support"] = weights["propensity_score"].between(lower, upper)
    weights["selected_ridge"] = selected_ridge
    return weights, diagnostics, selected["balance"], coefficients, selected["propensity"]


def entropy_balance_att(
    frame: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.Series, dict[str, object]]:
    matrix, _ = calibration_matrix(
        frame, config, list(config["continuous_covariates"])
    )
    treatment = frame[config["treatment"]].astype(int)
    treated = treatment.eq(1)
    control = ~treated
    target = matrix[treated.to_numpy()].mean(axis=0)
    control_weights, converged, iterations, error = entropy_tilt(
        matrix[control.to_numpy()],
        np.ones(control.sum()),
        target,
        float(config["entropy_balancing"]["tolerance"]),
        int(config["entropy_balancing"]["max_iterations"]),
    )
    control_weights *= treated.sum() / control_weights.sum()
    weights = pd.Series(1.0, index=frame.index)
    weights.loc[control] = control_weights
    normalized_control = weights.loc[control] / weights.loc[control].mean()
    diagnostics = {
        "method": "entropy_balancing_att",
        "converged": converged,
        "iterations": iterations,
        "max_moment_error": error,
        "treated_ess": effective_sample_size(weights.loc[treated]),
        "control_ess": effective_sample_size(weights.loc[control]),
        "control_ess_fraction": effective_sample_size(weights.loc[control]) / control.sum(),
        "max_normalized_control_weight": normalized_control.max(),
        "stable_for_primary": (
            converged
            and normalized_control.max()
            <= float(config["entropy_balancing"]["max_normalized_control_weight"])
            and effective_sample_size(weights.loc[control]) >= treated.sum()
        ),
    }
    return weights, diagnostics


def clustered_weighted_difference(
    values: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    treated = treatment == 1
    control = ~treated
    treated_weights = weights[treated] / weights[treated].sum()
    control_weights = weights[control] / weights[control].sum()
    treated_mean = float(np.sum(treated_weights * values[treated]))
    control_mean = float(np.sum(control_weights * values[control]))
    effect = treated_mean - control_mean
    influence = np.zeros(len(values))
    influence[treated] = treated_weights * (values[treated] - treated_mean)
    influence[control] = -control_weights * (values[control] - control_mean)
    unique_clusters = np.unique(clusters)
    cluster_scores = np.array([influence[clusters == cluster].sum() for cluster in unique_clusters])
    correction = len(unique_clusters) / (len(unique_clusters) - 1) if len(unique_clusters) > 1 else 1
    variance = correction * np.square(cluster_scores).sum()
    return effect, math.sqrt(max(variance, 0)), influence


def pretrend_diagnostics(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specification = config["pretrends"]
    windows = [
        (
            "calibration_window",
            int(specification["calibration_reference_match"]),
            list(range(int(specification["calibration_joint_window"][0]), int(specification["calibration_joint_window"][1]) + 1)),
        ),
        (
            "holdout_window",
            int(specification["holdout_reference_match"]),
            list(range(int(specification["holdout_joint_window"][0]), int(specification["holdout_joint_window"][1]) + 1)),
        ),
    ]
    earliest = min(min(times) for _, _, times in windows)
    latest = max(reference for _, reference, _ in windows)
    panel = panel[
        panel["transfermarkt_player_id"].isin(weights["transfermarkt_player_id"])
        & panel["relative_club_match"].between(earliest, latest)
    ].merge(
        weights[["transfermarkt_player_id", "analysis_weight"]],
        on="transfermarkt_player_id",
        how="inner",
        validate="many_to_one",
    )
    player_info = weights.set_index("transfermarkt_player_id")
    z = NormalDist().inv_cdf(1 - float(specification["alpha"]) / 2)
    coefficient_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(config["random_seed"]))
    for outcome in specification["outcomes"]:
        wide = panel.pivot_table(
            index="transfermarkt_player_id",
            columns="relative_club_match",
            values=outcome,
            aggfunc="first",
        )
        pooled_frame = panel[panel["relative_club_match"].between(-8, -1)].dropna(subset=[outcome])
        pooled_sd = float(pooled_frame[outcome].std(ddof=1))
        if not np.isfinite(pooled_sd) or pooled_sd == 0:
            pooled_sd = 1.0
        margin = float(specification["equivalence_margin_sd"]) * pooled_sd
        for label, reference, joint_times in windows:
            for event_time in joint_times:
                if event_time not in wide.columns or reference not in wide.columns:
                    continue
                delta = (wide[event_time] - wide[reference]).dropna()
                info = player_info.loc[delta.index]
                effect, standard_error, _ = clustered_weighted_difference(
                    delta.to_numpy(float),
                    info["treated_world_cup"].to_numpy(int),
                    info["analysis_weight"].to_numpy(float),
                    info["baseline_club_id"].to_numpy(int),
                )
                lower = effect - z * standard_error
                upper = effect + z * standard_error
                coefficient_rows.append(
                    {
                        "diagnostic_window": label,
                        "outcome": outcome,
                        "relative_club_match": event_time,
                        "reference_match": reference,
                        "effect": effect,
                        "standard_error": standard_error,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "pooled_pre_sd": pooled_sd,
                        "effect_sd": effect / pooled_sd,
                        "ci_lower_sd": lower / pooled_sd,
                        "ci_upper_sd": upper / pooled_sd,
                        "equivalence_margin": margin,
                        "equivalence_passed": lower >= -margin and upper <= margin,
                        "n_players": len(delta),
                        "n_treated": int(info["treated_world_cup"].sum()),
                        "n_controls": int(info["treated_world_cup"].eq(0).sum()),
                        "uses_post_treatment_outcomes": 0,
                    }
                )
            required = joint_times + [reference]
            if not set(required).issubset(wide.columns):
                continue
            complete = wide[required].dropna()
            info = player_info.loc[complete.index]
            deltas = complete[joint_times].to_numpy(float) - complete[[reference]].to_numpy(float)
            treatment = info["treated_world_cup"].to_numpy(int)
            analysis_weights = info["analysis_weight"].to_numpy(float)
            clusters = info["baseline_club_id"].to_numpy(int)
            treated = treatment == 1
            control = ~treated
            wt = analysis_weights[treated] / analysis_weights[treated].sum()
            wc = analysis_weights[control] / analysis_weights[control].sum()
            mean_t = wt @ deltas[treated]
            mean_c = wc @ deltas[control]
            coefficients = mean_t - mean_c
            influence = np.zeros_like(deltas)
            influence[treated] = wt[:, None] * (deltas[treated] - mean_t)
            influence[control] = -wc[:, None] * (deltas[control] - mean_c)
            unique_clusters = np.unique(clusters)
            scores = np.vstack([influence[clusters == cluster].sum(axis=0) for cluster in unique_clusters])
            correction = len(unique_clusters) / (len(unique_clusters) - 1)
            covariance = correction * scores.T @ scores
            wald = float(coefficients @ np.linalg.pinv(covariance) @ coefficients)
            simulations = int(specification["chi_square_simulations"])
            p_value = float((rng.chisquare(len(joint_times), simulations) >= wald).mean())
            outcome_rows = [
                row for row in coefficient_rows
                if row["outcome"] == outcome
                and row["diagnostic_window"] == label
                and row["relative_club_match"] in joint_times
            ]
            joint_rows.append(
                {
                    "diagnostic_window": label,
                    "outcome": outcome,
                    "joint_start": joint_times[0],
                    "joint_end": joint_times[-1],
                    "reference_match": reference,
                    "wald_statistic": wald,
                    "degrees_of_freedom": len(joint_times),
                    "simulation_p_value": p_value,
                    "all_pointwise_equivalence_passed": bool(outcome_rows and all(row["equivalence_passed"] for row in outcome_rows)),
                    "max_abs_pretrend_sd": max(abs(row["effect_sd"]) for row in outcome_rows),
                    "n_complete_players": len(complete),
                    "uses_post_treatment_outcomes": 0,
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(joint_rows)


def svg_love_plot(balance: pd.DataFrame, path: Path) -> None:
    data = balance[balance["variable_type"].eq("continuous")].copy()
    data = data.sort_values("smd_before", key=lambda x: x.abs(), ascending=True)
    width, left, right, row_height = 920, 260, 40, 34
    height = 80 + len(data) * row_height
    maximum = max(0.5, data[["smd_before", "smd_after"]].abs().to_numpy().max() * 1.1)
    plot_width = width - left - right
    def x(value: float) -> float:
        return left + abs(value) / maximum * plot_width
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        '<title>Balanceamento antes e depois dos pesos</title>',
        '<desc>Diferenças médias padronizadas absolutas; limites em 0,05 e 0,10.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for threshold, color in [(0.05, "#94a3b8"), (0.10, "#dc2626")]:
        parts.append(
            f'<line x1="{x(threshold):.1f}" y1="30" x2="{x(threshold):.1f}" y2="{height-35}" stroke="{color}" stroke-width="1" stroke-dasharray="5 4"/>'
        )
        parts.append(f'<text x="{x(threshold)+4:.1f}" y="22" font-size="12" fill="#334155">{threshold:.2f}</text>')
    for index, row in enumerate(data.itertuples(index=False)):
        y = 48 + index * row_height
        parts.append(f'<text x="10" y="{y+5}" font-size="12" fill="#111827">{html.escape(row.variable)}</text>')
        parts.append(f'<line x1="{x(row.smd_before):.1f}" y1="{y}" x2="{x(row.smd_after):.1f}" y2="{y}" stroke="#cbd5e1"/>')
        parts.append(f'<circle cx="{x(row.smd_before):.1f}" cy="{y}" r="5" fill="#f97316"><title>Antes: {abs(row.smd_before):.3f}</title></circle>')
        parts.append(f'<rect x="{x(row.smd_after)-4:.1f}" y="{y-4}" width="8" height="8" fill="#2563eb"><title>Depois: {abs(row.smd_after):.3f}</title></rect>')
    parts.append(f'<text x="{left}" y="{height-10}" font-size="12" fill="#334155">SMD absoluto (laranja: antes; azul: depois)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_propensity(weights: pd.DataFrame, path: Path) -> None:
    width, height, left, right, top, bottom = 920, 440, 65, 30, 35, 55
    plot_width, plot_height = width - left - right, height - top - bottom
    bins = np.linspace(0, 1, 21)
    treated = weights[weights["treated_world_cup"].eq(1)]
    controls = weights[weights["treated_world_cup"].eq(0)]
    ht, _ = np.histogram(treated["propensity_score"], bins=bins)
    hc, _ = np.histogram(controls["propensity_score"], bins=bins)
    ht = ht / max(ht.sum(), 1)
    hc = hc / max(hc.sum(), 1)
    maximum = max(ht.max(), hc.max())
    bar_width = plot_width / 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        '<title>Distribuição do propensity score</title>',
        '<desc>Histogramas proporcionais para tratados e controles na população qualificada.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#334155"/>',
    ]
    for index in range(20):
        x0 = left + index * bar_width
        treated_height = ht[index] / maximum * plot_height
        control_height = hc[index] / maximum * plot_height
        parts.append(f'<rect x="{x0+2:.1f}" y="{top+plot_height-treated_height:.1f}" width="{bar_width/2-3:.1f}" height="{treated_height:.1f}" fill="#2563eb" opacity="0.75"><title>Tratados {bins[index]:.2f}-{bins[index+1]:.2f}: {ht[index]:.3f}</title></rect>')
        parts.append(f'<rect x="{x0+bar_width/2:.1f}" y="{top+plot_height-control_height:.1f}" width="{bar_width/2-3:.1f}" height="{control_height:.1f}" fill="#f97316" opacity="0.65"><title>Controles {bins[index]:.2f}-{bins[index+1]:.2f}: {hc[index]:.3f}</title></rect>')
    for value in [0, 0.25, 0.5, 0.75, 1]:
        xpos = left + value * plot_width
        parts.append(f'<text x="{xpos-10:.1f}" y="{height-25}" font-size="12" fill="#334155">{value:.2f}</text>')
    parts.append(f'<text x="{left+plot_width/2-50:.1f}" y="{height-6}" font-size="12" fill="#334155">Propensity score</text>')
    parts.append(f'<rect x="{width-230}" y="18" width="12" height="12" fill="#2563eb"/><text x="{width-212}" y="29" font-size="12">Tratados</text>')
    parts.append(f'<rect x="{width-140}" y="18" width="12" height="12" fill="#f97316"/><text x="{width-122}" y="29" font-size="12">Controles</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_pretrend(coefficients: pd.DataFrame, path: Path) -> None:
    data = coefficients[
        coefficients["outcome"].eq("minutes_share")
        & coefficients["diagnostic_window"].eq("holdout_window")
    ].sort_values("relative_club_match")
    width, height, left, right, top, bottom = 920, 450, 65, 30, 35, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum = min(-0.15, data["ci_lower_sd"].min() * 1.1)
    maximum = max(0.15, data["ci_upper_sd"].max() * 1.1)
    def xpos(event: float) -> float:
        return left + (event - data["relative_club_match"].min()) / (data["relative_club_match"].max() - data["relative_club_match"].min()) * plot_width
    def ypos(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        '<title>Pré-tendências ponderadas de minutes share</title>',
        '<desc>Diferenças em diferenças pré-Copa, padronizadas, com intervalos de 95% e margem de equivalência de 0,10 desvio-padrão.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{ypos(0.10):.1f}" width="{plot_width}" height="{ypos(-0.10)-ypos(0.10):.1f}" fill="#dbeafe" opacity="0.55"/>',
        f'<line x1="{left}" y1="{ypos(0):.1f}" x2="{left+plot_width}" y2="{ypos(0):.1f}" stroke="#334155"/>',
    ]
    points: list[str] = []
    for row in data.itertuples(index=False):
        x = xpos(row.relative_club_match)
        y = ypos(row.effect_sd)
        parts.append(f'<line x1="{x:.1f}" y1="{ypos(row.ci_lower_sd):.1f}" x2="{x:.1f}" y2="{ypos(row.ci_upper_sd):.1f}" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"><title>Jogo {row.relative_club_match}: {row.effect_sd:.3f} DP</title></circle>')
        points.append(f'{x:.1f},{y:.1f}')
        parts.append(f'<text x="{x-8:.1f}" y="{height-30}" font-size="11" fill="#334155">{int(row.relative_club_match)}</text>')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="1.5"/>')
    parts.append(f'<text x="{left+plot_width/2-45:.1f}" y="{height-8}" font-size="12" fill="#334155">Jogo relativo</text>')
    parts.append(f'<text x="8" y="{top+plot_height/2:.1f}" font-size="12" fill="#334155">Efeito (DP)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def build_gate_summary(
    candidate_diagnostics: pd.DataFrame,
    pretrend_joint: pd.DataFrame,
    entropy_diagnostics: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    selected = candidate_diagnostics[candidate_diagnostics["selected"]].iloc[0]
    primary_pretrend = pretrend_joint[
        pretrend_joint["outcome"].eq("minutes_share")
        & pretrend_joint["diagnostic_window"].eq("holdout_window")
    ].iloc[0]
    gates = [
        ("overlap_candidate_passes_all_weight_gates", bool(selected["passes_selection_gates"]), "required"),
        ("treated_support_reaches_target_90pct", selected["treated_support_fraction"] >= float(config["selection_gates"]["target_treated_support_fraction"]), "target"),
        ("primary_pretrend_pointwise_equivalence", bool(primary_pretrend["all_pointwise_equivalence_passed"]), "required_for_strong_causal_language"),
        ("primary_pretrend_joint_p_above_0_05", primary_pretrend["simulation_p_value"] >= 0.05, "diagnostic_not_proof"),
        ("entropy_balancing_stable_for_primary", bool(entropy_diagnostics["stable_for_primary"]), "sensitivity_only"),
    ]
    rows = [
        {"gate": gate, "passed": passed, "role": role}
        for gate, passed, role in gates
    ]
    weighting_ready = bool(gates[0][1])
    causal_language_ready = weighting_ready and bool(gates[2][1])
    rows.extend(
        [
            {"gate": "weighting_ready_for_models", "passed": weighting_ready, "role": "decision"},
            {"gate": "strong_causal_language_ready", "passed": causal_language_ready, "role": "decision"},
        ]
    )
    return pd.DataFrame(rows)


def build_qa(
    risk: pd.DataFrame,
    weights: pd.DataFrame,
    candidates: pd.DataFrame,
    balance: pd.DataFrame,
    pretrend: pd.DataFrame,
    pretrend_joint: pd.DataFrame,
    gates: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    selected = candidates[candidates["selected"]].iloc[0]
    continuous = balance[balance["variable_type"].isin(["continuous", "prehistory"])]
    key = continuous[continuous["key_covariate"]]
    forbidden = set(config["forbidden_columns"])
    used = set(config["continuous_covariates"]) | set(config["key_covariates"])
    tests = [
        ("risk_set_unique_player", len(risk), risk["transfermarkt_player_id"].nunique(), not risk["transfermarkt_player_id"].duplicated().any()),
        ("weights_unique_player", len(weights), weights["transfermarkt_player_id"].nunique(), not weights["transfermarkt_player_id"].duplicated().any()),
        ("all_strata_have_both_groups", int((risk.groupby("weighting_stratum")["treated_world_cup"].nunique().eq(2)).sum()), risk["weighting_stratum"].nunique(), risk.groupby("weighting_stratum")["treated_world_cup"].nunique().eq(2).all()),
        ("no_forbidden_post_columns_used", len(used & forbidden), 0, not (used & forbidden)),
        ("selected_candidate_exists", int(candidates["selected"].sum()), 1, candidates["selected"].sum() == 1),
        ("selected_candidate_passes_gates", int(selected["passes_selection_gates"]), 1, bool(selected["passes_selection_gates"])),
        ("max_abs_smd", continuous["smd_after"].abs().max(), f"<={config['selection_gates']['max_abs_smd']}", continuous["smd_after"].abs().max() <= float(config["selection_gates"]["max_abs_smd"])),
        ("max_abs_smd_key", key["smd_after"].abs().max(), f"<={config['selection_gates']['max_abs_smd_key']}", key["smd_after"].abs().max() <= float(config["selection_gates"]["max_abs_smd_key"])),
        ("treated_ess_fraction", selected["treated_ess_fraction"], f">={config['selection_gates']['min_ess_fraction_each_group']}", selected["treated_ess_fraction"] >= float(config["selection_gates"]["min_ess_fraction_each_group"])),
        ("control_ess_fraction", selected["control_ess_fraction"], f">={config['selection_gates']['min_ess_fraction_each_group']}", selected["control_ess_fraction"] >= float(config["selection_gates"]["min_ess_fraction_each_group"])),
        ("treated_common_support", selected["treated_support_fraction"], f">={config['selection_gates']['min_treated_support_fraction']}", selected["treated_support_fraction"] >= float(config["selection_gates"]["min_treated_support_fraction"])),
        ("finite_positive_analysis_weights", int(np.isfinite(weights["analysis_weight"]).sum()), len(weights), np.isfinite(weights["analysis_weight"]).all() and weights["analysis_weight"].gt(0).all()),
        ("propensity_score_range", int(weights["propensity_score"].between(0, 1, inclusive="neither").sum()), len(weights), weights["propensity_score"].between(0, 1, inclusive="neither").all()),
        ("pretrend_uses_pre_only", int(pretrend["uses_post_treatment_outcomes"].eq(0).sum()), len(pretrend), pretrend["uses_post_treatment_outcomes"].eq(0).all()),
        ("pretrend_primary_present", int(pretrend["outcome"].eq("minutes_share").sum()), ">0", pretrend["outcome"].eq("minutes_share").any()),
        ("pretrend_calibration_primary_present", int((pretrend_joint["outcome"].eq("minutes_share") & pretrend_joint["diagnostic_window"].eq("calibration_window")).sum()), 1, (pretrend_joint["outcome"].eq("minutes_share") & pretrend_joint["diagnostic_window"].eq("calibration_window")).sum() == 1),
        ("pretrend_holdout_primary_present", int((pretrend_joint["outcome"].eq("minutes_share") & pretrend_joint["diagnostic_window"].eq("holdout_window")).sum()), 1, (pretrend_joint["outcome"].eq("minutes_share") & pretrend_joint["diagnostic_window"].eq("holdout_window")).sum() == 1),
        ("gate_summary_complete", len(gates), 7, len(gates) == 7),
    ]
    return pd.DataFrame(tests, columns=["check", "observed", "expected", "passed"])


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(frame):,} rows x {len(frame.columns)} cols")


def write_manifest(paths: list[Path], config_hash: str) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        with path.open("rb") as stream:
            row_count = max(sum(1 for _ in stream) - 1, 0) if path.suffix == ".csv" else np.nan
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "rows": row_count,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "weighting_config_sha256": config_hash,
                "input_cohort_sha256": file_sha256(PROCESSED / "analysis_sample_baseline_only.csv"),
                "generated_by": "scripts/analysis/build_weighting_phase.py",
            }
        )
    write_csv(pd.DataFrame(rows), REPORTS / "weighting_manifest.csv")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = load_json_yaml(WEIGHT_CONFIG)
    cohort, panel = read_inputs()
    validate_input_lineage(cohort)
    population = config["population_flag"]
    risk = cohort[cohort[population].eq(True)].copy()
    risk, history_columns = add_history_covariates(risk, panel, config)
    risk = add_strata(risk, config)
    risk, missing_columns = impute_covariates(risk, config, history_columns)

    weights, candidates, balance, coefficients, _ = select_overlap_weights(
        risk, config, missing_columns
    )
    entropy_weights, entropy_diagnostics = entropy_balance_att(risk, config)
    entropy_balance = build_balance_table(
        risk, entropy_weights, config, missing_columns, "entropy_balancing_att"
    )
    weights["entropy_att_weight"] = entropy_weights
    weight_summary = summarize_weights(
        risk,
        pd.Series(weights["analysis_weight"].to_numpy(), index=risk.index),
        pd.Series(weights["propensity_score"].to_numpy(), index=risk.index),
        config,
        "calibrated_overlap",
    )
    entropy_summary = pd.DataFrame([entropy_diagnostics])
    pretrend, pretrend_joint = pretrend_diagnostics(panel, weights, config)
    gates = build_gate_summary(candidates, pretrend_joint, entropy_diagnostics, config)
    qa = build_qa(risk, weights, candidates, balance, pretrend, pretrend_joint, gates, config)

    config_hash = file_sha256(WEIGHT_CONFIG)
    weights["weighting_config_sha256"] = config_hash
    outputs = {
        PROCESSED / "analysis_weights.csv": weights,
        REPORTS / "candidate_weighting_diagnostics.csv": candidates,
        REPORTS / "balance_diagnostics.csv": balance,
        REPORTS / "weight_distribution.csv": weight_summary,
        REPORTS / "propensity_coefficients.csv": coefficients,
        REPORTS / "entropy_balance_diagnostics.csv": entropy_summary,
        REPORTS / "entropy_balance.csv": entropy_balance,
        REPORTS / "pretrend_coefficients.csv": pretrend,
        REPORTS / "pretrend_joint_tests.csv": pretrend_joint,
        REPORTS / "phase5_gate_summary.csv": gates,
        REPORTS / "phase5_qa.csv": qa,
    }
    for path, frame in outputs.items():
        write_csv(frame, path)

    love_plot = REPORTS / "love_plot.svg"
    propensity_plot = REPORTS / "propensity_overlap.svg"
    pretrend_plot = REPORTS / "pretrend_minutes_share.svg"
    svg_love_plot(balance, love_plot)
    svg_propensity(weights, propensity_plot)
    svg_pretrend(pretrend, pretrend_plot)
    figure_paths = [love_plot, propensity_plot, pretrend_plot]
    for path in figure_paths:
        print(f"wrote {path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")

    write_manifest(list(outputs.keys()) + figure_paths, config_hash)
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(f"QA técnico da Fase 5 falhou: {failed}")
    print("Fase 5 concluída: pesos, balanceamento, suporte e pré-tendências gerados.")


if __name__ == "__main__":
    main()
