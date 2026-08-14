"""Fase 6: modelos confirmatórios ponderados, event study e sensibilidades.

Implementação sem dependências estatísticas externas: NumPy/Pandas, absorção
iterativa de efeitos fixos e variância sandwich agrupada.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "confirmatory"
CONFIG_PATH = ROOT / "config" / "confirmatory.yml"
ANALYSIS_CONFIG_PATH = ROOT / "config" / "analysis.yml"
PANEL_PATH = PROCESSED / "design_player_match.csv"
WEIGHTS_PATH = PROCESSED / "analysis_weights.csv"
COHORT_PATH = PROCESSED / "analysis_sample_baseline_only.csv"
PHASE5_GATES = ROOT / "reports" / "weighting" / "phase5_gate_summary.csv"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_p_value(statistic: float) -> float:
    return 2 * (1 - NormalDist().cdf(abs(float(statistic))))


def noninferiority_table(
    results: pd.DataFrame,
    analysis_config: dict[str, object],
    alpha: float,
) -> pd.DataFrame:
    """Interpreta o SESOI negativo como margem direcional complementar.

    O teste unilateral H0: beta <= -SESOI não substitui o teste bilateral nem
    implica ausência de efeito. Ele apenas verifica, sob cada especificação,
    se os dados excluem uma queda aguda média maior que a margem congelada.
    """
    margin = -abs(float(analysis_config["sesoi"]["absolute_effect"]))
    z_one_sided = NormalDist().inv_cdf(1 - alpha)
    selected = results[
        results["outcome"].eq("minutes_share")
        & results["coefficient"].eq("treated_x_acute")
        & results["specification"].isin(
            [
                "primary_player_club_game_fe",
                "external_player_league_position_week_fe",
                "sensitivity_differential_linear_trend",
            ]
        )
    ]
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        for cluster_scheme, se in [
            ("player", float(row.standard_error_player)),
            ("player_and_club_game", float(row.standard_error_double_cluster)),
        ]:
            estimate = float(row.estimate)
            statistic = (estimate - margin) / se
            lower = estimate - z_one_sided * se
            rows.append(
                {
                    "specification": row.specification,
                    "outcome": row.outcome,
                    "coefficient": row.coefficient,
                    "cluster_scheme": cluster_scheme,
                    "estimate": estimate,
                    "standard_error": se,
                    "noninferiority_margin": margin,
                    "one_sided_confidence_level": 1 - alpha,
                    "one_sided_lower_bound": lower,
                    "p_value_noninferiority": 1 - NormalDist().cdf(statistic),
                    "noninferiority_passed": lower > margin,
                    "estimate_minutes_per_match": estimate * 90,
                    "margin_minutes_per_match": margin * 90,
                    "lower_bound_minutes_per_match": lower * 90,
                    "interpretation": "model_conditional_excludes_decline_beyond_margin"
                    if lower > margin
                    else "material_decline_not_excluded_under_specification",
                }
            )
    return pd.DataFrame(rows)


def factor_codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values.astype(str), sort=True)[0]


def residualize(
    matrix: np.ndarray,
    weights: np.ndarray,
    groups: list[np.ndarray],
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> tuple[np.ndarray, int, float]:
    result = matrix.astype(float).copy()
    maximum_change = np.inf
    for iteration in range(1, max_iterations + 1):
        previous = result.copy()
        for codes in groups:
            count = int(codes.max()) + 1
            denominator = np.bincount(codes, weights=weights, minlength=count)
            for column in range(result.shape[1]):
                numerator = np.bincount(
                    codes, weights=weights * result[:, column], minlength=count
                )
                means = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
                result[:, column] -= means[codes]
        maximum_change = float(np.max(np.abs(result - previous)))
        if maximum_change < tolerance:
            break
    return result, iteration, maximum_change


def cluster_meat(scores: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = int(codes.max()) + 1
    aggregated = np.zeros((groups, scores.shape[1]))
    np.add.at(aggregated, codes, scores)
    return aggregated.T @ aggregated, aggregated


def fit_hdfe(
    frame: pd.DataFrame,
    outcome: str,
    regressors: dict[str, np.ndarray],
    fixed_effects: list[str],
) -> dict[str, object]:
    columns = [outcome, "analysis_weight"] + fixed_effects
    valid = frame[columns].notna().all(axis=1).to_numpy().copy()
    for values in regressors.values():
        valid &= np.isfinite(values)
    data = frame.loc[valid].reset_index(drop=True)
    y = data[outcome].to_numpy(float)
    w = data["analysis_weight"].to_numpy(float)
    x = np.column_stack([np.asarray(values)[valid] for values in regressors.values()])
    names = list(regressors)
    groups = [factor_codes(data[column]) for column in fixed_effects]
    transformed, iterations, error = residualize(
        np.column_stack([y, x]), w, groups
    )
    y_tilde = transformed[:, 0]
    x_tilde = transformed[:, 1:]
    xtwx = (x_tilde.T * w) @ x_tilde
    bread = np.linalg.pinv(xtwx)
    beta = bread @ ((x_tilde.T * w) @ y_tilde)
    residual = y_tilde - x_tilde @ beta
    scores = x_tilde * (w * residual)[:, None]
    n, k = x_tilde.shape

    player_codes = factor_codes(data["transfermarkt_player_id"])
    game_codes = factor_codes(data["club_game_id"])
    player_meat, player_scores = cluster_meat(scores, player_codes)
    game_meat, game_scores = cluster_meat(scores, game_codes)
    observation_meat = scores.T @ scores
    player_count = int(player_codes.max()) + 1
    game_count = int(game_codes.max()) + 1
    player_correction = player_count / max(player_count - 1, 1) * (n - 1) / max(n - k, 1)
    game_correction = game_count / max(game_count - 1, 1) * (n - 1) / max(n - k, 1)
    observation_correction = n / max(n - 1, 1) * (n - 1) / max(n - k, 1)
    variance_player = bread @ (player_correction * player_meat) @ bread
    variance_double = bread @ (
        player_correction * player_meat
        + game_correction * game_meat
        - observation_correction * observation_meat
    ) @ bread
    variance_double = (variance_double + variance_double.T) / 2

    return {
        "names": names,
        "beta": beta,
        "se_player": np.sqrt(np.maximum(np.diag(variance_player), 0)),
        "se_double": np.sqrt(np.maximum(np.diag(variance_double), 0)),
        "bread": bread,
        "scores": scores,
        "player_scores": player_scores,
        "game_scores": game_scores,
        "data": data,
        "n": n,
        "players": data["transfermarkt_player_id"].nunique(),
        "club_games": data["club_game_id"].nunique(),
        "fe_iterations": iterations,
        "fe_error": error,
        "r2_within": 1 - np.sum(w * residual**2) / max(np.sum(w * y_tilde**2), 1e-15),
    }


def fitted_fixed_effects(
    target: np.ndarray,
    weights: np.ndarray,
    groups: list[np.ndarray],
    tolerance: float = 1e-9,
    max_iterations: int = 500,
) -> tuple[np.ndarray, int]:
    effects = [np.zeros(int(codes.max()) + 1) for codes in groups]
    fitted = np.zeros(len(target))
    for iteration in range(1, max_iterations + 1):
        maximum_change = 0.0
        for index, codes in enumerate(groups):
            partial = target - (fitted - effects[index][codes])
            count = len(effects[index])
            denominator = np.bincount(codes, weights=weights, minlength=count)
            numerator = np.bincount(codes, weights=weights * partial, minlength=count)
            updated = np.divide(numerator, denominator, out=np.zeros(count), where=denominator > 0)
            change = updated - effects[index]
            fitted += change[codes]
            effects[index] = updated
            maximum_change = max(maximum_change, float(np.max(np.abs(change))))
        if maximum_change < tolerance:
            break
    return fitted, iteration


def fit_ppml_hdfe(
    frame: pd.DataFrame,
    outcome: str,
    regressors: dict[str, np.ndarray],
    fixed_effects: list[str],
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> dict[str, object]:
    columns = [outcome, "analysis_weight", "transfermarkt_player_id", "club_game_id"] + fixed_effects
    valid = frame[columns].notna().all(axis=1).to_numpy().copy()
    for values in regressors.values():
        valid &= np.isfinite(values)
    data = frame.loc[valid].copy()
    source_index = np.flatnonzero(valid)
    separated_rows = 0
    while True:
        player_positive = data.groupby("transfermarkt_player_id")[outcome].transform("sum").gt(0)
        game_positive = data.groupby("club_game_id")[outcome].transform("sum").gt(0)
        keep = player_positive & game_positive
        if keep.all():
            break
        separated_rows += int((~keep).sum())
        source_index = source_index[keep.to_numpy()]
        data = data.loc[keep].copy()
    data = data.reset_index(drop=True)
    y = data[outcome].to_numpy(float)
    base_weight = data["analysis_weight"].to_numpy(float)
    x = np.column_stack([np.asarray(values)[source_index] for values in regressors.values()])
    names = list(regressors)
    groups = [factor_codes(data[column]) for column in fixed_effects]
    beta = np.zeros(x.shape[1])
    eta = np.full(len(y), math.log(max(np.average(y, weights=base_weight), 1e-4)))
    converged = False
    maximum_change = np.inf
    for iteration in range(1, max_iterations + 1):
        mu = np.exp(np.clip(eta, -20, 20))
        irls_weight = base_weight * mu
        working = eta + (y - mu) / np.maximum(mu, 1e-12)
        transformed, _, _ = residualize(np.column_stack([working, x]), irls_weight, groups, tolerance=1e-9)
        y_tilde = transformed[:, 0]
        x_tilde = transformed[:, 1:]
        beta_new = np.linalg.pinv((x_tilde.T * irls_weight) @ x_tilde) @ ((x_tilde.T * irls_weight) @ y_tilde)
        fe_fit, _ = fitted_fixed_effects(working - x @ beta_new, irls_weight, groups)
        eta_new = np.clip(x @ beta_new + fe_fit, -20, 20)
        maximum_change = max(float(np.max(np.abs(beta_new - beta))), float(np.max(np.abs(eta_new - eta))))
        beta, eta = beta_new, eta_new
        if maximum_change < tolerance:
            converged = True
            break
    mu = np.exp(np.clip(eta, -20, 20))
    irls_weight = base_weight * mu
    transformed_x, _, _ = residualize(x, irls_weight, groups, tolerance=1e-9)
    bread = np.linalg.pinv((transformed_x.T * irls_weight) @ transformed_x)
    scores = transformed_x * (base_weight * (y - mu))[:, None]
    player_codes = factor_codes(data["transfermarkt_player_id"])
    meat, _ = cluster_meat(scores, player_codes)
    cluster_count = int(player_codes.max()) + 1
    correction = cluster_count / max(cluster_count - 1, 1) * (len(y) - 1) / max(len(y) - len(beta), 1)
    variance = bread @ (correction * meat) @ bread
    return {
        "names": names,
        "beta": beta,
        "standard_error": np.sqrt(np.maximum(np.diag(variance), 0)),
        "converged": converged,
        "iterations": iteration,
        "maximum_change": maximum_change,
        "n_observations": len(data),
        "n_players": data["transfermarkt_player_id"].nunique(),
        "separated_rows_dropped": separated_rows,
    }


def weighted_summary_scale(frame: pd.DataFrame, outcome: str) -> tuple[float, float]:
    pre = frame[frame["relative_club_match"].between(-8, -1)].dropna(subset=[outcome])
    values = pre[outcome].to_numpy(float)
    weights = pre["analysis_weight"].to_numpy(float)
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    treated = pre[pre["treated_world_cup"].eq(1)]
    treated_mean = float(np.average(treated[outcome], weights=treated["analysis_weight"]))
    return math.sqrt(max(variance, 1e-15)), treated_mean


def model_rows(
    fit: dict[str, object],
    outcome: str,
    specification: str,
    scale: float,
    baseline_mean: float,
    alpha: float,
) -> list[dict[str, object]]:
    z = NormalDist().inv_cdf(1 - alpha / 2)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(fit["names"]):
        estimate = float(fit["beta"][index])
        se_player = float(fit["se_player"][index])
        se_double = float(fit["se_double"][index])
        rows.append(
            {
                "specification": specification,
                "outcome": outcome,
                "coefficient": name,
                "estimate": estimate,
                "standard_error_player": se_player,
                "ci_lower_player": estimate - z * se_player,
                "ci_upper_player": estimate + z * se_player,
                "p_value_player": normal_p_value(estimate / se_player) if se_player > 0 else np.nan,
                "standard_error_double_cluster": se_double,
                "ci_lower_double_cluster": estimate - z * se_double,
                "ci_upper_double_cluster": estimate + z * se_double,
                "p_value_double_cluster": normal_p_value(estimate / se_double) if se_double > 0 else np.nan,
                "effect_sd": estimate / scale,
                "percent_of_treated_baseline": 100 * estimate / baseline_mean if baseline_mean != 0 else np.nan,
                "baseline_weighted_sd": scale,
                "treated_baseline_mean": baseline_mean,
                "n_observations": fit["n"],
                "n_players": fit["players"],
                "n_club_games": fit["club_games"],
                "within_r_squared": fit["r2_within"],
                "fe_iterations": fit["fe_iterations"],
                "fe_max_error": fit["fe_error"],
            }
        )
    return rows


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(float))
    adjusted = np.empty(len(p_values))
    running = 0.0
    m = len(p_values)
    for rank, original_index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values.iloc[original_index]))
        running = max(running, value)
        adjusted[original_index] = running
    return pd.Series(adjusted, index=p_values.index)


def multiplier_p_value(
    fit: dict[str, object], coefficient: str, cluster_column: str, draws: int, seed: int
) -> float:
    index = fit["names"].index(coefficient)
    data = fit["data"]
    codes = factor_codes(data[cluster_column])
    _, aggregated = cluster_meat(fit["scores"], codes)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, len(aggregated)))
    perturbations = (signs @ aggregated) @ fit["bread"].T
    se = float(np.std(perturbations[:, index], ddof=1))
    observed = abs(float(fit["beta"][index]) / se) if se > 0 else np.inf
    simulated = np.abs(perturbations[:, index] / se) if se > 0 else np.zeros(draws)
    return float((1 + np.sum(simulated >= observed)) / (draws + 1))


def event_study(
    frame: pd.DataFrame, outcome: str, config: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    event_times = [int(value) for value in config["event_study"]["event_times"]]
    treatment = frame["treated_world_cup"].to_numpy(float)
    regressors = {
        f"event_{event:+d}": treatment * frame["relative_club_match"].eq(event).to_numpy(float)
        for event in event_times
    }
    fit = fit_hdfe(frame, outcome, regressors, ["transfermarkt_player_id", "club_game_id"])
    scale, baseline = weighted_summary_scale(frame, outcome)
    rows = model_rows(fit, outcome, "event_study_player_club_game_fe", scale, baseline, float(config["inference"]["alpha"]))
    table = pd.DataFrame(rows)
    table["relative_club_match"] = event_times
    draws = int(config["event_study"]["simultaneous_band_draws"])
    rng = np.random.default_rng(int(config["random_seed"]) + sum(ord(c) for c in outcome))
    signs = rng.choice([-1.0, 1.0], size=(draws, len(fit["player_scores"])))
    perturbations = (signs @ fit["player_scores"]) @ fit["bread"].T
    standardized = np.abs(perturbations / np.maximum(fit["se_player"], 1e-15))
    critical = float(np.quantile(np.max(standardized, axis=1), 1 - float(config["inference"]["alpha"])))
    table["simultaneous_critical_value"] = critical
    table["simultaneous_lower"] = table["estimate"] - critical * table["standard_error_player"]
    table["simultaneous_upper"] = table["estimate"] + critical * table["standard_error_player"]
    table["simultaneous_lower_sd"] = table["simultaneous_lower"] / scale
    table["simultaneous_upper_sd"] = table["simultaneous_upper"] / scale
    return table, fit


def dr_att_did(frame: pd.DataFrame, weights: pd.DataFrame, cohort: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    covariates = load_json(ROOT / "config" / "weighting.yml")["continuous_covariates"]
    player = weights[["transfermarkt_player_id", "treated_world_cup", "propensity_score", "weighting_stratum"]].copy()
    player = player.merge(cohort[["transfermarkt_player_id"] + covariates], on="transfermarkt_player_id", how="left", validate="one_to_one")
    for column in covariates:
        player[column] = player.groupby("weighting_stratum")[column].transform(lambda values: values.fillna(values.median()))
        player[column] = player[column].fillna(player[column].median())
    x = player[covariates].to_numpy(float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1)
    x = np.column_stack([np.ones(len(x)), x])
    treatment = player["treated_world_cup"].to_numpy(int)
    propensity = np.clip(player["propensity_score"].to_numpy(float), 0.01, 0.99)
    rows: list[dict[str, object]] = []
    z = NormalDist().inv_cdf(1 - float(config["inference"]["alpha"]) / 2)
    for outcome in config["confirmatory_outcomes"]:
        wide = frame.pivot_table(index="transfermarkt_player_id", columns="relative_club_match", values=outcome, aggfunc="first")
        pre = wide[[event for event in range(-8, 0) if event in wide]].mean(axis=1)
        acute = wide[[event for event in range(1, 5) if event in wide]].mean(axis=1)
        delta = (acute - pre).reindex(player["transfermarkt_player_id"]).to_numpy(float)
        valid = np.isfinite(delta)
        controls = valid & (treatment == 0)
        ridge = 1e-6 * np.eye(x.shape[1])
        ridge[0, 0] = 0
        beta0 = np.linalg.pinv(x[controls].T @ x[controls] + ridge) @ (x[controls].T @ delta[controls])
        residualized = delta - x @ beta0
        treated_valid = valid & (treatment == 1)
        control_odds = propensity[controls] / (1 - propensity[controls])
        control_odds /= control_odds.sum()
        estimate = float(residualized[treated_valid].mean() - np.sum(control_odds * residualized[controls]))
        influence_t = (residualized[treated_valid] - residualized[treated_valid].mean()) / treated_valid.sum()
        influence_c = -control_odds * (residualized[controls] - np.sum(control_odds * residualized[controls]))
        standard_error = float(math.sqrt(np.sum(influence_t**2) + np.sum(influence_c**2)))
        rows.append({
            "outcome": outcome,
            "estimand": "ATT",
            "estimate": estimate,
            "standard_error": standard_error,
            "ci_lower": estimate - z * standard_error,
            "ci_upper": estimate + z * standard_error,
            "p_value": normal_p_value(estimate / standard_error) if standard_error > 0 else np.nan,
            "n_treated": int(treated_valid.sum()),
            "n_controls": int(controls.sum()),
            "note": "different_estimand_from_primary_ATO",
        })
    return pd.DataFrame(rows)


def svg_event_study(table: pd.DataFrame, path: Path, label: str) -> None:
    data = table.sort_values("relative_club_match")
    width, height, left, right, top, bottom = 920, 470, 65, 35, 35, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    lower = min(-0.35, float(data["simultaneous_lower_sd"].min()) * 1.05)
    upper = max(0.35, float(data["simultaneous_upper_sd"].max()) * 1.05)
    events = data["relative_club_match"].tolist()
    positions = {event: left + index / (len(events) - 1) * plot_width for index, event in enumerate(events)}
    def y(value: float) -> float:
        return top + (upper - value) / (upper - lower) * plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
        f'<title>Event study de {html.escape(label)}</title>',
        '<desc>Coeficientes em desvios-padrão com intervalos pontuais de 95% e bandas simultâneas.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{left+plot_width}" y2="{y(0):.1f}" stroke="#334155"/>',
    ]
    for row in data.itertuples(index=False):
        xpos = positions[row.relative_club_match]
        parts.append(f'<line x1="{xpos:.1f}" y1="{y(row.simultaneous_lower_sd):.1f}" x2="{xpos:.1f}" y2="{y(row.simultaneous_upper_sd):.1f}" stroke="#94a3b8" stroke-width="5"/>')
        parts.append(f'<line x1="{xpos:.1f}" y1="{y(row.ci_lower_player/row.baseline_weighted_sd):.1f}" x2="{xpos:.1f}" y2="{y(row.ci_upper_player/row.baseline_weighted_sd):.1f}" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<circle cx="{xpos:.1f}" cy="{y(row.effect_sd):.1f}" r="4" fill="#2563eb"><title>{row.relative_club_match}: {row.effect_sd:.3f} DP</title></circle>')
        parts.append(f'<text x="{xpos-7:.1f}" y="{height-30}" font-size="11" fill="#334155">{int(row.relative_club_match)}</text>')
    parts.append(f'<text x="{left+plot_width/2-45:.1f}" y="{height-8}" font-size="12" fill="#334155">Jogo relativo</text>')
    parts.append(f'<text x="8" y="{top+plot_height/2:.1f}" font-size="12" fill="#334155">Efeito (DP)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_forest(table: pd.DataFrame, path: Path) -> None:
    data = table[(table["specification"] == "primary_player_club_game_fe") & (table["coefficient"] == "treated_x_acute")].copy()
    labels = {"minutes_share":"Minutes share", "in_match_squad":"Na súmula", "played":"Jogou", "started":"Titular", "minutes_played":"Minutos", "npxg_xa_per_club_match":"npxG+xA"}
    width, left, right, row_height = 920, 220, 45, 54
    height = 70 + len(data) * row_height
    minimum = min(-0.35, float((data["ci_lower_player"] / data["baseline_weighted_sd"]).min()) * 1.1)
    maximum = max(0.35, float((data["ci_upper_player"] / data["baseline_weighted_sd"]).max()) * 1.1)
    plot_width = width - left - right
    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">', '<title>Efeitos agudos confirmatórios</title>', '<rect width="100%" height="100%" fill="white"/>', f'<line x1="{x(0):.1f}" y1="25" x2="{x(0):.1f}" y2="{height-35}" stroke="#334155"/>']
    for index, row in enumerate(data.itertuples(index=False)):
        y = 50 + index * row_height
        estimate = row.effect_sd
        lower = row.ci_lower_player / row.baseline_weighted_sd
        upper = row.ci_upper_player / row.baseline_weighted_sd
        parts.append(f'<text x="10" y="{y+4}" font-size="12" fill="#111827">{html.escape(labels[row.outcome])}</text>')
        parts.append(f'<line x1="{x(lower):.1f}" y1="{y}" x2="{x(upper):.1f}" y2="{y}" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<circle cx="{x(estimate):.1f}" cy="{y}" r="5" fill="#2563eb"><title>{estimate:.3f} DP</title></circle>')
    parts.append(f'<text x="{left+plot_width/2-55:.1f}" y="{height-8}" font-size="12" fill="#334155">Efeito agudo (DP)</text></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(frame):,} rows x {len(frame.columns)} cols")


def main() -> None:
    config = load_json(CONFIG_PATH)
    analysis_config = load_json(ANALYSIS_CONFIG_PATH)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    weights = pd.read_csv(WEIGHTS_PATH)
    cohort = pd.read_csv(COHORT_PATH, low_memory=False)
    phase5_gates = pd.read_csv(PHASE5_GATES)
    if not bool(phase5_gates.loc[phase5_gates["gate"].eq("weighting_ready_for_models"), "passed"].iloc[0]):
        raise RuntimeError("Pesos da Fase 5 não estão liberados para modelos.")
    panel = panel[panel["transfermarkt_player_id"].isin(weights["transfermarkt_player_id"])].merge(
        weights[["transfermarkt_player_id", "analysis_weight", "propensity_score", "weighting_stratum"]],
        on="transfermarkt_player_id", how="inner", validate="many_to_one"
    )
    panel["club_game_id"] = panel["club_id"].astype(str) + "|" + panel["game_id"].astype(str)
    dates = pd.to_datetime(panel["match_date"])
    iso = dates.dt.isocalendar()
    panel["league_position_week"] = panel["league_name_y"].astype(str) + "|" + panel["position_group"].astype(str) + "|" + iso.year.astype(str) + "-" + iso.week.astype(str)
    panel["baseline_club_cluster"] = panel["baseline_club_id"].astype(str)
    panel["national_cluster"] = panel["national_team_eligible"].astype(str)
    for column in config["external_controls"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        panel[column] = panel[column].fillna(panel.groupby("league_position_week")[column].transform("median")).fillna(panel[column].median())

    estimation = panel[panel["relative_club_match"].isin(list(range(-8, 0)) + list(range(1, 9)))].copy()
    treatment = estimation["treated_world_cup"].to_numpy(float)
    acute = estimation["relative_club_match"].between(1, 4).to_numpy(float)
    consolidated = estimation["relative_club_match"].between(5, 8).to_numpy(float)
    alpha = float(config["inference"]["alpha"])
    all_rows: list[dict[str, object]] = []
    primary_fits: dict[str, dict[str, object]] = {}
    for outcome in config["confirmatory_outcomes"]:
        scale, baseline = weighted_summary_scale(estimation, outcome)
        primary = fit_hdfe(
            estimation, outcome,
            {"treated_x_acute": treatment * acute, "treated_x_consolidated": treatment * consolidated},
            ["transfermarkt_player_id", "club_game_id"],
        )
        primary_fits[outcome] = primary
        all_rows.extend(model_rows(primary, outcome, "primary_player_club_game_fe", scale, baseline, alpha))
        external_regressors = {
            "treated_x_acute": treatment * acute,
            "treated_x_consolidated": treatment * consolidated,
            **{f"control_{column}": estimation[column].to_numpy(float) for column in config["external_controls"]},
        }
        external = fit_hdfe(estimation, outcome, external_regressors, ["transfermarkt_player_id", "league_position_week"])
        all_rows.extend(model_rows(external, outcome, "external_player_league_position_week_fe", scale, baseline, alpha))
        trend = fit_hdfe(
            estimation, outcome,
            {"treated_x_acute": treatment * acute, "treated_x_consolidated": treatment * consolidated, "treated_x_linear_time": treatment * estimation["relative_club_match"].to_numpy(float)},
            ["transfermarkt_player_id", "club_game_id"],
        )
        all_rows.extend(model_rows(trend, outcome, "sensitivity_differential_linear_trend", scale, baseline, alpha))

    results = pd.DataFrame(all_rows)
    family = (results["specification"].eq("primary_player_club_game_fe") & results["coefficient"].eq("treated_x_acute"))
    results["p_value_holm_acute_family"] = np.nan
    results.loc[family, "p_value_holm_acute_family"] = holm_adjust(results.loc[family, "p_value_player"]).to_numpy()
    noninferiority = noninferiority_table(results, analysis_config, alpha)

    ppml_rows: list[dict[str, object]] = []
    for outcome in ["minutes_played", "npxg_xa_per_club_match"]:
        fit = fit_ppml_hdfe(
            estimation,
            outcome,
            {"treated_x_acute": treatment * acute, "treated_x_consolidated": treatment * consolidated},
            ["transfermarkt_player_id", "club_game_id"],
        )
        for index, name in enumerate(fit["names"]):
            estimate = float(fit["beta"][index])
            standard_error = float(fit["standard_error"][index])
            ppml_rows.append({
                "outcome": outcome,
                "coefficient": name,
                "log_rate_ratio": estimate,
                "standard_error_player": standard_error,
                "rate_ratio": math.exp(estimate),
                "percent_change": 100 * (math.exp(estimate) - 1),
                "ci_lower_rate_ratio": math.exp(estimate - 1.959963984540054 * standard_error),
                "ci_upper_rate_ratio": math.exp(estimate + 1.959963984540054 * standard_error),
                "p_value": normal_p_value(estimate / standard_error) if standard_error > 0 else np.nan,
                "converged": fit["converged"],
                "iterations": fit["iterations"],
                "maximum_change": fit["maximum_change"],
                "n_observations": fit["n_observations"],
                "n_players": fit["n_players"],
                "separated_rows_dropped": fit["separated_rows_dropped"],
            })
    ppml = pd.DataFrame(ppml_rows)

    primary_fit = primary_fits[config["primary_outcome"]]
    robustness = pd.DataFrame([
        {"outcome": config["primary_outcome"], "coefficient": config["primary_coefficient"], "cluster_scheme": "baseline_club_score_multiplier", "p_value": multiplier_p_value(primary_fit, config["primary_coefficient"], "baseline_club_cluster", int(config["inference"]["score_multiplier_draws"]), int(config["random_seed"]))},
        {"outcome": config["primary_outcome"], "coefficient": config["primary_coefficient"], "cluster_scheme": "national_eligibility_score_multiplier", "p_value": multiplier_p_value(primary_fit, config["primary_coefficient"], "national_cluster", int(config["inference"]["score_multiplier_draws"]), int(config["random_seed"]) + 1)},
    ])

    event_tables = []
    for outcome in config["confirmatory_outcomes"]:
        table, _ = event_study(estimation, outcome, config)
        event_tables.append(table)
    events = pd.concat(event_tables, ignore_index=True)

    placebo_frame = panel[panel["relative_club_match"].between(-14, -9)].copy()
    placebo_treatment = placebo_frame["treated_world_cup"].to_numpy(float)
    pseudo_post = placebo_frame["relative_club_match"].between(-11, -9).to_numpy(float)
    placebo_rows = []
    for outcome in config["confirmatory_outcomes"]:
        scale, baseline = weighted_summary_scale(panel, outcome)
        fit = fit_hdfe(placebo_frame, outcome, {"treated_x_holdout_placebo": placebo_treatment * pseudo_post}, ["transfermarkt_player_id", "club_game_id"])
        placebo_rows.extend(model_rows(fit, outcome, "holdout_temporal_placebo", scale, baseline, alpha))
    placebos = pd.DataFrame(placebo_rows)
    dr_results = dr_att_did(estimation, weights, cohort, config)

    strong_phase5 = bool(phase5_gates.loc[phase5_gates["gate"].eq("strong_causal_language_ready"), "passed"].iloc[0])
    primary_noninferiority = bool(
        noninferiority.loc[
            noninferiority["specification"].eq("primary_player_club_game_fe")
            & noninferiority["cluster_scheme"].eq("player"),
            "noninferiority_passed",
        ].iloc[0]
    )
    gates = pd.DataFrame([
        {"gate": "phase5_weighting_ready", "passed": True, "role": "required"},
        {"gate": "all_primary_models_converged", "passed": bool((results.loc[results["specification"].eq("primary_player_club_game_fe"), "fe_max_error"] < 1e-8).all()), "role": "required"},
        {"gate": "holdout_placebo_primary_p_above_0_05", "passed": bool(placebos.loc[placebos["outcome"].eq(config["primary_outcome"]), "p_value_player"].iloc[0] >= 0.05), "role": "diagnostic_not_proof"},
        {"gate": "phase5_strong_causal_language_ready", "passed": strong_phase5, "role": "required_for_strong_causal_language"},
        {"gate": "models_ready_for_reporting", "passed": True, "role": "decision"},
        {"gate": "primary_model_excludes_material_acute_decline", "passed": primary_noninferiority, "role": "model_conditional_not_causal"},
        {"gate": "strong_causal_language_ready", "passed": strong_phase5, "role": "decision"},
    ])

    qa_checks = [
        ("weights_unique_player", weights["transfermarkt_player_id"].nunique(), len(weights), weights["transfermarkt_player_id"].is_unique),
        ("panel_players_match_weights", panel["transfermarkt_player_id"].nunique(), len(weights), panel["transfermarkt_player_id"].nunique() == len(weights)),
        ("analysis_weight_positive_finite", int(np.isfinite(panel["analysis_weight"]).sum()), len(panel), np.isfinite(panel["analysis_weight"]).all() and panel["analysis_weight"].gt(0).all()),
        ("primary_results_complete", len(results[results["specification"].eq("primary_player_club_game_fe")]), 12, len(results[results["specification"].eq("primary_player_club_game_fe")]) == 12),
        ("acute_holm_family_complete", int(results["p_value_holm_acute_family"].notna().sum()), 6, results["p_value_holm_acute_family"].notna().sum() == 6),
        ("noninferiority_sensitivity_complete", len(noninferiority), 6, len(noninferiority) == 6 and np.isfinite(noninferiority[["estimate", "standard_error", "one_sided_lower_bound", "p_value_noninferiority"]]).all().all()),
        ("event_study_complete", len(events), 90, len(events) == 90),
        ("event_study_reference_omitted", int(events["relative_club_match"].eq(-1).sum()), 0, not events["relative_club_match"].eq(-1).any()),
        ("holdout_placebo_pre_only", int(placebo_frame["relative_club_match"].max()), -9, placebo_frame["relative_club_match"].max() == -9),
        ("placebo_complete", len(placebos), 6, len(placebos) == 6),
        ("dr_att_complete", len(dr_results), 6, len(dr_results) == 6),
        ("ppml_complete", len(ppml), 4, len(ppml) == 4),
        ("ppml_converged", int(ppml["converged"].sum()), len(ppml), ppml["converged"].all()),
        ("finite_primary_estimates", int(np.isfinite(results.loc[results["specification"].eq("primary_player_club_game_fe"), "estimate"]).sum()), 12, np.isfinite(results.loc[results["specification"].eq("primary_player_club_game_fe"), "estimate"]).all()),
        ("strong_language_respects_phase5_gate", int(gates.loc[gates["gate"].eq("strong_causal_language_ready"), "passed"].iloc[0]), int(strong_phase5), bool(gates.loc[gates["gate"].eq("strong_causal_language_ready"), "passed"].iloc[0]) == strong_phase5),
    ]
    qa = pd.DataFrame(qa_checks, columns=["check", "observed", "expected", "passed"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    outputs = {
        REPORTS / "confirmatory_coefficients.csv": results,
        REPORTS / "event_study_coefficients.csv": events,
        REPORTS / "holdout_placebo.csv": placebos,
        REPORTS / "dr_att_did.csv": dr_results,
        REPORTS / "ppml_robustness.csv": ppml,
        REPORTS / "cluster_robustness.csv": robustness,
        REPORTS / "noninferiority_primary.csv": noninferiority,
        REPORTS / "phase6_gate_summary.csv": gates,
        REPORTS / "phase6_qa.csv": qa,
    }
    for path, table in outputs.items():
        write_csv(table, path)
    event_minutes = events[events["outcome"].eq("minutes_share")]
    event_squad = events[events["outcome"].eq("in_match_squad")]
    figures = [REPORTS / "event_study_minutes_share.svg", REPORTS / "event_study_squad.svg", REPORTS / "forest_acute_outcomes.svg"]
    svg_event_study(event_minutes, figures[0], "minutes share")
    svg_event_study(event_squad, figures[1], "participação na súmula")
    svg_forest(results, figures[2])
    for path in figures:
        print(f"wrote {path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")

    config_hash = file_sha256(CONFIG_PATH)
    manifest_rows = []
    for path in list(outputs) + figures:
        manifest_rows.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "confirmatory_config_sha256": config_hash,
            "weights_sha256": file_sha256(WEIGHTS_PATH),
            "panel_sha256": file_sha256(PANEL_PATH),
            "generated_by": "scripts/analysis/build_confirmatory_phase.py",
        })
    write_csv(pd.DataFrame(manifest_rows), REPORTS / "confirmatory_manifest.csv")
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(f"QA técnico da Fase 6 falhou: {failed}")
    print("Fase 6 concluída: modelos, event studies, placebos e sensibilidades gerados.")


if __name__ == "__main__":
    main()
