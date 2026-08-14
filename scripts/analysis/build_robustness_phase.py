"""Fase 7: robustez, falsificações, dose e heterogeneidade limitada."""

from __future__ import annotations

import hashlib
import html
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from build_confirmatory_phase import fit_hdfe, normal_p_value  # noqa: E402


PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "robustness"
CONFIG_PATH = ROOT / "config" / "robustness.yml"
PANEL_PATH = PROCESSED / "design_player_match.csv"
WEIGHTS_PATH = PROCESSED / "analysis_weights.csv"
COHORT_PATH = PROCESSED / "analysis_sample_baseline_only.csv"
MATCHING_PATH = PROCESSED / "matching_assignments.csv"
PHASE6_RESULTS = ROOT / "reports" / "confirmatory" / "confirmatory_coefficients.csv"
PHASE6_EVENTS = ROOT / "reports" / "confirmatory" / "event_study_coefficients.csv"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    weights = pd.read_csv(WEIGHTS_PATH)
    cohort = pd.read_csv(COHORT_PATH, low_memory=False)
    panel = panel[panel["transfermarkt_player_id"].isin(weights["transfermarkt_player_id"])].merge(
        weights[["transfermarkt_player_id", "analysis_weight", "entropy_att_weight", "weighting_stratum", "common_support"]],
        on="transfermarkt_player_id", how="inner", validate="many_to_one"
    )
    panel["club_game_id"] = panel["club_id"].astype(str) + "|" + panel["game_id"].astype(str)
    panel["baseline_club_cluster"] = panel["baseline_club_id"].astype(str)
    panel["national_cluster"] = panel["national_team_eligible"].astype(str)
    return panel, weights, cohort


def coefficient_row(
    fit: dict[str, object], coefficient: str, specification: str, outcome: str, extra: dict[str, object]
) -> dict[str, object]:
    index = fit["names"].index(coefficient)
    estimate = float(fit["beta"][index])
    standard_error = float(fit["se_player"][index])
    z = 1.959963984540054
    return {
        "specification": specification,
        "outcome": outcome,
        "coefficient": coefficient,
        "estimate": estimate,
        "standard_error_player": standard_error,
        "ci_lower": estimate - z * standard_error,
        "ci_upper": estimate + z * standard_error,
        "p_value": normal_p_value(estimate / standard_error) if standard_error > 0 else np.nan,
        "n_observations": fit["n"],
        "n_players": fit["players"],
        "n_club_games": fit["club_games"],
        "fe_max_error": fit["fe_error"],
        **extra,
    }


def fit_window(
    panel: pd.DataFrame,
    outcome: str,
    pre_start: int,
    post_end: int,
    weight_column: str,
    specification: str,
    subset: pd.Series | None = None,
) -> dict[str, object]:
    selector = panel["relative_club_match"].between(pre_start, -1) | panel["relative_club_match"].between(1, post_end)
    if subset is not None:
        selector &= subset
    frame = panel.loc[selector].copy()
    frame["analysis_weight"] = frame[weight_column].astype(float)
    treatment_post = frame["treated_world_cup"].to_numpy(float) * frame["relative_club_match"].between(1, post_end).to_numpy(float)
    fit = fit_hdfe(frame, outcome, {"treated_x_post": treatment_post}, ["transfermarkt_player_id", "club_game_id"])
    return coefficient_row(fit, "treated_x_post", specification, outcome, {"pre_start": pre_start, "post_end": post_end, "weight_method": weight_column})


def fit_acute_consolidated(
    panel: pd.DataFrame,
    outcome: str,
    weight_column: str,
    specification: str,
    subset: pd.Series | None = None,
    add_trend: bool = False,
    pre_start: int = -8,
) -> list[dict[str, object]]:
    selector = panel["relative_club_match"].between(pre_start, -1) | panel["relative_club_match"].between(1, 8)
    if subset is not None:
        selector &= subset
    frame = panel.loc[selector].copy()
    frame["analysis_weight"] = frame[weight_column].astype(float)
    treatment = frame["treated_world_cup"].to_numpy(float)
    regressors = {
        "treated_x_acute": treatment * frame["relative_club_match"].between(1, 4).to_numpy(float),
        "treated_x_consolidated": treatment * frame["relative_club_match"].between(5, 8).to_numpy(float),
    }
    if add_trend:
        regressors["treated_x_linear_time"] = treatment * frame["relative_club_match"].to_numpy(float)
    fit = fit_hdfe(frame, outcome, regressors, ["transfermarkt_player_id", "club_game_id"])
    return [
        coefficient_row(fit, name, specification, outcome, {"pre_start": pre_start, "weight_method": weight_column})
        for name in ["treated_x_acute", "treated_x_consolidated"]
    ]


def matching_weights(weights: pd.DataFrame) -> pd.Series:
    assignments = pd.read_csv(MATCHING_PATH)
    treated = assignments[["treated_player_id", "treated_weight"]].drop_duplicates("treated_player_id").rename(columns={"treated_player_id":"transfermarkt_player_id", "treated_weight":"weight"})
    controls = assignments.groupby("control_player_id", as_index=False)["control_weight"].sum().rename(columns={"control_player_id":"transfermarkt_player_id", "control_weight":"weight"})
    combined = pd.concat([treated, controls], ignore_index=True).groupby("transfermarkt_player_id", as_index=False)["weight"].sum()
    return weights["transfermarkt_player_id"].map(combined.set_index("transfermarkt_player_id")["weight"]).fillna(0.0)


def weight_and_sample_sensitivity(panel: pd.DataFrame, weights: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    player = weights[["transfermarkt_player_id"]].copy()
    player["matching_weight"] = matching_weights(weights)
    panel = panel.merge(player, on="transfermarkt_player_id", how="left", validate="many_to_one")
    panel["unweighted"] = 1.0
    panel["overlap_truncated_95"] = panel["analysis_weight"].clip(upper=panel["analysis_weight"].quantile(0.95))
    panel["overlap_truncated_99"] = panel["analysis_weight"].clip(upper=panel["analysis_weight"].quantile(0.99))
    rows: list[dict[str, object]] = []
    for column, label in [
        ("analysis_weight", "overlap"), ("entropy_att_weight", "entropy_att"),
        ("matching_weight", "matching_1_to_3"), ("unweighted", "unweighted"),
        ("overlap_truncated_95", "overlap_truncated_95"), ("overlap_truncated_99", "overlap_truncated_99"),
    ]:
        subset = panel[column].gt(0)
        rows.extend(fit_acute_consolidated(panel, "minutes_share", column, label, subset=subset))
    same_nation_ids = set(cohort.loc[cohort["eligible_same_nation_support"].eq(True), "transfermarkt_player_id"])
    same_nation = panel["transfermarkt_player_id"].isin(same_nation_ids)
    rows.extend(fit_acute_consolidated(panel, "minutes_share", "analysis_weight", "same_nation_support", subset=same_nation))
    cohort_minutes = cohort.set_index("transfermarkt_player_id")["pre_minutes_played"]
    threshold360 = panel["transfermarkt_player_id"].map(cohort_minutes).ge(360)
    rows.extend(fit_acute_consolidated(panel, "minutes_share", "analysis_weight", "baseline_minutes_ge_360", subset=threshold360))
    return pd.DataFrame(rows)


def window_sensitivity(panel: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    rows = []
    for window in [int(value) for value in config["post_window_sensitivity"]]:
        rows.append(fit_window(panel, "minutes_share", -window, window, "analysis_weight", f"symmetric_window_{window}"))
        rows.append(fit_window(panel, "npxg_xa_per_club_match", -window, window, "analysis_weight", f"symmetric_window_{window}"))
    return pd.DataFrame(rows)


def pretrend_length_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for games in [8, 12, 14]:
        rows.extend(fit_acute_consolidated(panel, "minutes_share", "analysis_weight", f"differential_trend_pre_{games}", add_trend=True, pre_start=-games))
    return pd.DataFrame(rows)


def leave_one_out(panel: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in config["leave_one_out"]:
        values = sorted(panel[dimension].dropna().astype(str).unique())
        for value in values:
            subset = panel[dimension].astype(str).ne(value)
            estimates = fit_acute_consolidated(panel, "minutes_share", "analysis_weight", f"leave_one_{dimension}_out", subset=subset)
            for row in estimates:
                row["omitted_dimension"] = dimension
                row["omitted_value"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def trend_violation_sensitivity(config: dict[str, object]) -> pd.DataFrame:
    events = pd.read_csv(PHASE6_EVENTS)
    results = pd.read_csv(PHASE6_RESULTS)
    pre = events[(events["outcome"].eq("minutes_share")) & events["relative_club_match"].between(-8, -2)].sort_values("relative_club_match")
    coefficients = np.append(pre["estimate"].to_numpy(float), 0.0)
    max_level = float(np.max(np.abs(coefficients)))
    max_first_difference = float(np.max(np.abs(np.diff(coefficients))))
    primary = results[(results["specification"].eq("primary_player_club_game_fe")) & results["outcome"].eq("minutes_share")]
    rows = []
    for coefficient, horizon in [("treated_x_acute", 2.5), ("treated_x_consolidated", 6.5)]:
        source = primary[primary["coefficient"].eq(coefficient)].iloc[0]
        for method, scale in [("relative_magnitude", max_level), ("smoothness_first_difference", max_first_difference * horizon)]:
            for multiplier in config["trend_sensitivity_multipliers"]:
                bias = float(multiplier) * scale
                rows.append({
                    "coefficient": coefficient,
                    "method": method,
                    "multiplier": multiplier,
                    "observed_pre_bound": scale,
                    "allowed_bias": bias,
                    "estimate": source["estimate"],
                    "original_ci_lower": source["ci_lower_player"],
                    "original_ci_upper": source["ci_upper_player"],
                    "bias_adjusted_ci_lower": float(source["ci_lower_player"]) - bias,
                    "bias_adjusted_ci_upper": float(source["ci_upper_player"]) + bias,
                    "sign_robust_positive": float(source["ci_lower_player"]) - bias > 0,
                    "note": "transparent_relative_violation_analysis_inspired_by_HonestDiD_not_package_identical",
                })
    return pd.DataFrame(rows)


def confounding_sensitivity() -> pd.DataFrame:
    results = pd.read_csv(PHASE6_RESULTS)
    source = results[(results["specification"].eq("primary_player_club_game_fe")) & results["outcome"].eq("minutes_share")]
    rows = []
    for coefficient in ["treated_x_acute", "treated_x_consolidated"]:
        row = source[source["coefficient"].eq(coefficient)].iloc[0]
        degrees = max(int(row["n_players"]) - 2, 1)
        t_value = abs(float(row["estimate"]) / float(row["standard_error_player"]))
        f_value = t_value / math.sqrt(degrees)
        f2 = f_value**2
        robustness_value = 0.5 * (math.sqrt(f2**2 + 4 * f2) - f2)
        rows.append({
            "coefficient": coefficient,
            "t_value_cluster_robust": t_value,
            "approximate_degrees_of_freedom": degrees,
            "approximate_robustness_value": robustness_value,
            "interpretation": "approximate_partial_R2_strength_needed_to_reduce_point_estimate_to_zero",
            "caution": "diagnostic_approximation_for_clustered_HDFE_not_sensemakr_exact",
        })
    return pd.DataFrame(rows)


def weighted_ols(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bread = np.linalg.pinv((x.T * weights) @ x)
    beta = bread @ ((x.T * weights) @ y)
    residual = y - x @ beta
    scores = x * (weights * residual)[:, None]
    correction = len(y) / max(len(y) - x.shape[1], 1)
    variance = bread @ (correction * scores.T @ scores) @ bread
    return beta, np.sqrt(np.maximum(np.diag(variance), 0))


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def dose_response(panel: pd.DataFrame, weights: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    treated_ids = weights.loc[weights["treated_world_cup"].eq(1), ["transfermarkt_player_id", "analysis_weight"]]
    wide = panel.pivot_table(index="transfermarkt_player_id", columns="relative_club_match", values="minutes_share", aggfunc="first")
    delta = wide[[1,2,3,4]].mean(axis=1) - wide[[-8,-7,-6,-5,-4,-3,-2,-1]].mean(axis=1)
    columns = ["transfermarkt_player_id", "wc_minutes_played", "age_at_world_cup", "market_value_log", "pre_minutes_share", "pre_start_rate", "pre_injury_days", "pre_club_ppg"]
    data = treated_ids.merge(cohort[columns], on="transfermarkt_player_id", how="left", validate="one_to_one")
    data["delta"] = data["transfermarkt_player_id"].map(delta)
    data = data.dropna(subset=["delta", "wc_minutes_played"]).copy()
    controls = ["age_at_world_cup", "market_value_log", "pre_minutes_share", "pre_start_rate", "pre_injury_days", "pre_club_ppg"]
    for column in controls:
        data[column] = data[column].fillna(data[column].median())
        deviation = data[column].std(ddof=0)
        data[column] = (data[column] - data[column].mean()) / (deviation if deviation > 0 else 1)
    minutes = data["wc_minutes_played"].to_numpy(float)
    category = pd.cut(minutes, bins=[-0.1,0,90,270,np.inf], labels=["0","1-90","91-270",">270"])
    dummies = pd.get_dummies(category, dtype=float)
    base = np.column_stack([np.ones(len(data)), data[controls].to_numpy(float)])
    x_category = np.column_stack([base, dummies[["1-90","91-270",">270"]].to_numpy(float)])
    beta, se = weighted_ols(data["delta"].to_numpy(float), x_category, data["analysis_weight"].to_numpy(float))
    rows = []
    for offset, name in enumerate(["1-90","91-270",">270"]):
        index = base.shape[1] + offset
        rows.append({"model":"categories_adjusted", "term":name, "reference":"0", "estimate":beta[index], "standard_error":se[index], "p_value":normal_p_value(beta[index]/se[index]), "n_players":len(data)})
    spline = np.column_stack([np.minimum(minutes,90), np.minimum(np.maximum(minutes-90,0),180), np.maximum(minutes-270,0)]) / 90
    x_spline = np.column_stack([base, spline])
    beta_s, se_s = weighted_ols(data["delta"].to_numpy(float), x_spline, data["analysis_weight"].to_numpy(float))
    for offset, name in enumerate(["slope_0_90_per_90", "slope_91_270_per_90", "slope_above_270_per_90"]):
        index = base.shape[1] + offset
        rows.append({"model":"restricted_linear_spline_adjusted", "term":name, "reference":"none", "estimate":beta_s[index], "standard_error":se_s[index], "p_value":normal_p_value(beta_s[index]/se_s[index]), "n_players":len(data)})
    result = pd.DataFrame(rows)
    result["p_value_bh"] = bh_adjust(result["p_value"].tolist())
    counts = category.value_counts().to_dict()
    result["category_counts"] = json.dumps({str(key):int(value) for key,value in counts.items()}, sort_keys=True)
    result["interpretation"] = "associational_among_called_players"
    return result


def heterogeneity(panel: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    player = cohort.set_index("transfermarkt_player_id")
    panel = panel.copy()
    panel["age_group"] = pd.cut(panel["transfermarkt_player_id"].map(player["age_at_world_cup"]), bins=[-np.inf,24.999,29.999,np.inf], labels=["under25","25to29","30plus"]).astype(str)
    panel["baseline_starter"] = np.where(panel["transfermarkt_player_id"].map(player["pre_start_rate"]).ge(0.5), "starter_high", "starter_low")
    panel["prior_injury"] = np.where(panel["transfermarkt_player_id"].map(player["pre_injury_days"]).gt(0), "injury_yes", "injury_no")
    definitions = {
        "position_group": ("Attack", ["Defender", "Midfield"]),
        "age_group": ("25to29", ["under25", "30plus"]),
        "baseline_starter": ("starter_low", ["starter_high"]),
        "prior_injury": ("injury_no", ["injury_yes"]),
    }
    frame = panel[panel["relative_club_match"].isin(list(range(-8,0))+list(range(1,9)))].copy()
    treatment = frame["treated_world_cup"].to_numpy(float)
    acute = frame["relative_club_match"].between(1,4).to_numpy(float)
    consolidated = frame["relative_club_match"].between(5,8).to_numpy(float)
    rows = []
    for modifier, (reference, levels) in definitions.items():
        regressors = {"treated_x_acute":treatment*acute, "treated_x_consolidated":treatment*consolidated}
        for level in levels:
            regressors[f"acute_difference_{level}_vs_{reference}"] = treatment*acute*frame[modifier].eq(level).to_numpy(float)
        fit = fit_hdfe(frame, "minutes_share", regressors, ["transfermarkt_player_id","club_game_id"])
        for name in regressors:
            if not name.startswith("acute_difference_"):
                continue
            row = coefficient_row(fit, name, f"heterogeneity_{modifier}", "minutes_share", {"modifier":modifier,"reference":reference})
            rows.append(row)
    result = pd.DataFrame(rows)
    result["p_value_bh"] = bh_adjust(result["p_value"].tolist())
    return result


def coverage_matrix() -> pd.DataFrame:
    rows = [
        ("post_window", "4_games", "implemented", "complete local coverage"),
        ("post_window", "6_games", "implemented", "complete local coverage"),
        ("post_window", "8_games", "implemented", "complete local coverage"),
        ("post_window", "10_games", "implemented", "complete local coverage"),
        ("baseline_minutes", "90", "implemented_fresh_overlap_diagnostic_gate_not_met", "separate fresh-weight output preserves the frozen-grid ESS gate result"),
        ("baseline_minutes", "180", "implemented_primary", "frozen design"),
        ("baseline_minutes", "360", "implemented_subset", "restriction within weighted risk set"),
        ("pretrend_history", "8_games", "implemented", "complete"),
        ("pretrend_history", "12_games", "implemented", "complete"),
        ("pretrend_history", "14_games", "implemented", "complete"),
        ("pretrend_history", "16_games", "unavailable", "game -15 incomplete and game -16 absent"),
        ("season_placebo", "2021_22_or_2023_24", "unavailable", "no adjacent-season local panel"),
        ("population", "same_nation_support", "implemented_subset", "uses frozen overlap weights"),
        ("population", "near_miss_or_big_five_broad", "deferred_requires_reweighting", "estimand and support must be rebuilt"),
        ("weights", "overlap_entropy_matching_unweighted", "implemented", "entropy and matching remain sensitivity-only"),
        ("leave_one_out", "league_club_selection", "implemented", "primary outcome"),
        ("dose", "world_cup_minutes", "implemented_associational", "called players only"),
        ("heterogeneity", "four_frozen_modifiers", "implemented_exploratory", "formal interaction differences with BH"),
    ]
    return pd.DataFrame(rows, columns=["domain","specification","status","reason"])


def svg_specification_curve(data: pd.DataFrame, path: Path) -> None:
    selected = data[data["coefficient"].eq("treated_x_acute")].copy().reset_index(drop=True)
    width, height, left, right, top, bottom = 980, 500, 225, 35, 30, 45
    row_height = (height-top-bottom)/max(len(selected),1)
    minimum = min(-0.15, float(selected["ci_lower"].min())*1.1)
    maximum = max(0.15, float(selected["ci_upper"].max())*1.1)
    def x(value: float) -> float:
        return left + (value-minimum)/(maximum-minimum)*(width-left-right)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">','<title>Curva de especificações do efeito agudo</title>','<rect width="100%" height="100%" fill="white"/>',f'<line x1="{x(0):.1f}" y1="{top}" x2="{x(0):.1f}" y2="{height-bottom}" stroke="#334155"/>']
    for index,row in enumerate(selected.itertuples(index=False)):
        y=top+(index+0.5)*row_height
        parts.append(f'<text x="8" y="{y+4:.1f}" font-size="11" fill="#111827">{html.escape(str(row.specification))}</text>')
        parts.append(f'<line x1="{x(row.ci_lower):.1f}" y1="{y:.1f}" x2="{x(row.ci_upper):.1f}" y2="{y:.1f}" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<circle cx="{x(row.estimate):.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>')
    parts.append(f'<text x="{left+250}" y="{height-8}" font-size="12" fill="#334155">Efeito em minutes share</text></svg>')
    path.write_text("\n".join(parts),encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path,index=False,encoding="utf-8",lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(frame):,} rows x {len(frame.columns)} cols")


def main() -> None:
    config=load_json(CONFIG_PATH)
    panel,weights,cohort=prepare_panel()
    windows=window_sensitivity(panel,config)
    sample_weights=weight_and_sample_sensitivity(panel,weights,cohort)
    pretrend=pretrend_length_sensitivity(panel)
    loo=leave_one_out(panel,config)
    trends=trend_violation_sensitivity(config)
    confounding=confounding_sensitivity()
    dose=dose_response(panel,weights,cohort)
    hetero=heterogeneity(panel,cohort)
    coverage=coverage_matrix()

    consolidated_trend=trends[(trends["coefficient"].eq("treated_x_consolidated")) & trends["multiplier"].eq(0.25)]
    robust_consolidated=bool(consolidated_trend["sign_robust_positive"].all())
    gates=pd.DataFrame([
        {"gate":"core_robustness_matrix_executed","passed":True,"role":"required"},
        {"gate":"post_10_game_window_available","passed":True,"role":"robustness"},
        {"gate":"adjacent_season_placebo_available","passed":False,"role":"data_limitation"},
        {"gate":"consolidated_sign_robust_to_small_trend_violation","passed":robust_consolidated,"role":"required_for_causal_interpretation"},
        {"gate":"strong_causal_language_ready","passed":False,"role":"decision"},
        {"gate":"adjusted_association_language_ready","passed":True,"role":"decision"},
    ])
    qa_items=[
        ("window_sensitivity_complete",len(windows),8,len(windows)==8),
        ("sample_weight_sensitivity_complete",len(sample_weights),16,len(sample_weights)==16),
        ("pretrend_length_complete",len(pretrend),6,len(pretrend)==6),
        ("leave_one_out_nonempty",len(loo),">0",len(loo)>0),
        ("trend_sensitivity_complete",len(trends),24,len(trends)==24),
        ("confounding_sensitivity_complete",len(confounding),2,len(confounding)==2),
        ("dose_response_complete",len(dose),6,len(dose)==6),
        ("heterogeneity_complete",len(hetero),6,len(hetero)==6),
        ("coverage_records_unavailable_specs",int(coverage["status"].str.contains("unavailable").sum()),">=2",coverage["status"].str.contains("unavailable").sum()>=2),
        ("all_core_estimates_finite",int(np.isfinite(pd.concat([windows["estimate"],sample_weights["estimate"],pretrend["estimate"],loo["estimate"]])).sum()),len(windows)+len(sample_weights)+len(pretrend)+len(loo),np.isfinite(pd.concat([windows["estimate"],sample_weights["estimate"],pretrend["estimate"],loo["estimate"]])).all()),
        ("strong_language_closed",int(gates.loc[gates["gate"].eq("strong_causal_language_ready"),"passed"].iloc[0]),0,not bool(gates.loc[gates["gate"].eq("strong_causal_language_ready"),"passed"].iloc[0])),
    ]
    qa=pd.DataFrame(qa_items,columns=["check","observed","expected","passed"])
    outputs={
        REPORTS/"coverage_matrix.csv":coverage,
        REPORTS/"window_sensitivity.csv":windows,
        REPORTS/"sample_weight_sensitivity.csv":sample_weights,
        REPORTS/"pretrend_length_sensitivity.csv":pretrend,
        REPORTS/"leave_one_out.csv":loo,
        REPORTS/"parallel_trends_sensitivity.csv":trends,
        REPORTS/"unobserved_confounding_sensitivity.csv":confounding,
        REPORTS/"dose_response_associational.csv":dose,
        REPORTS/"heterogeneity_interactions.csv":hetero,
        REPORTS/"phase7_gate_summary.csv":gates,
        REPORTS/"phase7_qa.csv":qa,
    }
    for path,frame in outputs.items(): write_csv(frame,path)
    figure=REPORTS/"specification_curve.svg"
    svg_specification_curve(sample_weights,figure)
    print(f"wrote {figure.relative_to(ROOT)}: {figure.stat().st_size:,} bytes")
    manifest=[]
    config_hash=sha256(CONFIG_PATH)
    for path in list(outputs)+[figure]:
        manifest.append({"relative_path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path),"robustness_config_sha256":config_hash,"weights_sha256":sha256(WEIGHTS_PATH),"panel_sha256":sha256(PANEL_PATH),"generated_by":"scripts/analysis/build_robustness_phase.py"})
    write_csv(pd.DataFrame(manifest),REPORTS/"robustness_manifest.csv")
    if not qa["passed"].all():
        raise RuntimeError(f"QA técnico da Fase 7 falhou: {qa.loc[~qa['passed'],'check'].tolist()}")
    print("Fase 7 concluída: robustez, sensibilidades, dose e heterogeneidade geradas.")


if __name__=="__main__": main()
