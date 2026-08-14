"""Fase 4: protocolo, coorte baseline-only e diagnósticos de desenho.

O script usa apenas informação disponível até 13/11/2022 para definir a coorte,
o grupo em risco e o diagnóstico inicial de poder. Outcomes posteriores são
mantidos no painel para as fases seguintes, mas não participam da elegibilidade.
"""

from __future__ import annotations

import hashlib
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


CONFIG_PATH = ROOT / "config" / "analysis.yml"
OVERRIDES_PATH = ROOT / "config" / "manual_design_overrides.csv"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "design"

ELIGIBILITY_INPUT_COLUMNS = {
    "is_goalkeeper",
    "position_group",
    "pre_club_games",
    "pre_minutes_played",
    "baseline_club_id",
    "baseline_identity_valid",
    "national_team_eligible",
}

TEAM_ALIASES = {
    "United States": "USA",
    "USMNT": "USA",
    "South Korea": "Korea Republic",
    "Korea, South": "Korea Republic",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
}


def load_config() -> dict[str, object]:
    # JSON é um subconjunto válido de YAML 1.2 e evita dependência de parser.
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def canonical_team(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return TEAM_ALIASES.get(text, text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_inputs() -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(INTERIM / "big5_player_match_2022_23.csv", low_memory=False)
    panel["match_date"] = pd.to_datetime(panel["match_date"])
    panel["date_of_birth"] = pd.to_datetime(panel["date_of_birth"], errors="coerce")

    spells = pd.read_csv(PROCESSED / "roster_spells.csv", low_memory=False)
    for column in ["first_observed", "last_observed", "spell_start", "spell_end"]:
        spells[column] = pd.to_datetime(spells[column], errors="coerce")

    analytic = pd.read_csv(PROCESSED / "analytic_player_match.csv", low_memory=False)
    analytic["match_date"] = pd.to_datetime(analytic["match_date"])
    analytic["valuation_date"] = pd.to_datetime(analytic["valuation_date"], errors="coerce")

    understat_pm = pd.read_csv(INTERIM / "understat_player_match_2022_23.csv", low_memory=False)
    game_map = pd.read_csv(INTERIM / "understat_transfermarkt_game_crosswalk.csv", low_memory=False)
    player_map = pd.read_csv(INTERIM / "understat_transfermarkt_player_crosswalk.csv", low_memory=False)
    overrides = pd.read_csv(OVERRIDES_PATH, low_memory=False)
    return {
        "panel": panel,
        "spells": spells,
        "analytic": analytic,
        "understat_pm": understat_pm,
        "game_map": game_map,
        "player_map": player_map,
        "overrides": overrides,
    }


def build_schedule(panel: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    design = config["design"]
    cutoff = pd.Timestamp(design["baseline_cutoff"])
    post_start = pd.Timestamp(design["post_start"])
    n_pre = int(design["pre_diagnostic_games"])
    n_pre_min = int(design["pre_estimation_games"])
    n_post = int(design["post_games"])
    n_primary_post = int(design.get("primary_post_games", n_post))
    acute = int(design["acute_games"])
    columns = [
        "game_id", "match_date", "competition_id", "league_name", "club_id", "club_name",
        "opponent_club_id", "opponent_club_name", "is_home", "club_goals", "opponent_goals",
        "round", "club_table_position", "opponent_table_position", "club_formation",
        "days_since_club_game",
    ]
    full = panel[columns].drop_duplicates(["game_id", "club_id"]).copy()
    strength = full[full["match_date"].le(cutoff)].copy()
    strength["points"] = np.select(
        [
            strength["club_goals"].gt(strength["opponent_goals"]),
            strength["club_goals"].eq(strength["opponent_goals"]),
        ],
        [3, 1],
        default=0,
    )
    club_strength = strength.groupby("club_id", as_index=False).agg(
        pre_schedule_club_ppg=("points", "mean")
    )
    selected: list[pd.DataFrame] = []
    for club_id, games in full.sort_values(["match_date", "game_id"]).groupby("club_id"):
        pre = games[games["match_date"].le(cutoff)].tail(n_pre).copy()
        post = games[games["match_date"].ge(post_start)].head(n_post).copy()
        if len(pre) < n_pre_min or len(post) != n_post:
            raise RuntimeError(
                f"Clube {club_id} sem cobertura mínima: pre={len(pre)}, post={len(post)}"
            )
        pre["relative_club_match"] = np.arange(-len(pre), 0)
        post["relative_club_match"] = np.arange(1, len(post) + 1)
        selected.extend([pre, post])
    schedule = pd.concat(selected, ignore_index=True)
    schedule["window"] = np.select(
        [
            schedule["relative_club_match"].le(-9),
            schedule["relative_club_match"].between(-8, -1),
            schedule["relative_club_match"].between(1, acute),
            schedule["relative_club_match"].between(acute + 1, n_primary_post),
            schedule["relative_club_match"].between(n_primary_post + 1, n_post),
        ],
        ["diagnostic_pre", "estimation_pre", "post_acute", "post_consolidated", "post_extension"],
        default="invalid",
    )
    opponent_strength = club_strength.rename(
        columns={
            "club_id": "opponent_club_id",
            "pre_schedule_club_ppg": "pre_opponent_ppg",
        }
    )
    schedule = schedule.merge(opponent_strength, on="opponent_club_id", how="left")
    return schedule.sort_values(["club_id", "match_date", "game_id"]).reset_index(drop=True)


def assign_baseline_club(
    panel: pd.DataFrame,
    spells: pd.DataFrame,
    overrides: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    cutoff = pd.Timestamp(config["design"]["baseline_cutoff"])
    pre = panel[panel["match_date"].le(cutoff)].copy()
    observed = pre.groupby(["transfermarkt_player_id", "club_id"], as_index=False).agg(
        last_pre_observed=("match_date", "max"),
        first_pre_observed=("match_date", "min"),
        pre_source_minutes=("minutes_played", "sum"),
        pre_source_games=("game_id", "nunique"),
    )
    candidates = observed.merge(
        spells,
        on=["transfermarkt_player_id", "club_id"],
        how="left",
        validate="one_to_one",
    )
    candidates["active_by_spell_at_cutoff"] = (
        candidates["spell_start"].le(cutoff) & candidates["spell_end"].ge(cutoff)
    ).astype("int8")
    counts = candidates.groupby("transfermarkt_player_id", as_index=False).agg(
        baseline_club_candidate_count=("club_id", "size"),
        active_spell_candidate_count=("active_by_spell_at_cutoff", "sum"),
    )
    candidates = candidates.merge(counts, on="transfermarkt_player_id", how="left")
    candidates = candidates.sort_values(
        [
            "transfermarkt_player_id", "active_by_spell_at_cutoff", "last_pre_observed",
            "pre_source_minutes", "club_id",
        ],
        ascending=[True, False, False, False, True],
    )
    chosen = candidates.drop_duplicates("transfermarkt_player_id", keep="first").copy()
    chosen["baseline_club_assignment_source"] = np.where(
        chosen["active_by_spell_at_cutoff"].eq(1),
        "active_spell_then_last_pre_observation",
        "last_pre_observation_fallback",
    )
    chosen["baseline_club_assignment_review"] = (
        chosen["active_spell_candidate_count"].ne(1)
    ).astype("int8")
    chosen = chosen.rename(
        columns={"club_id": "baseline_club_id", "club_name": "baseline_club_name"}
    )
    manual = overrides[
        overrides["override_type"].eq("baseline_club")
        & overrides["decision"].eq("confirmed")
    ].copy()
    manual["baseline_club_id"] = pd.to_numeric(manual["baseline_club_id"], errors="coerce")
    manual_map = manual.set_index("transfermarkt_player_id")["baseline_club_id"]
    selected = chosen["transfermarkt_player_id"].isin(manual_map.index)
    expected = chosen.loc[selected, "transfermarkt_player_id"].map(manual_map)
    if not chosen.loc[selected, "baseline_club_id"].astype(float).eq(expected.astype(float)).all():
        raise RuntimeError("Override de clube-base diverge da atribuição observada")
    chosen["baseline_club_override_applied"] = selected.astype("int8")
    chosen.loc[selected, "baseline_club_assignment_review"] = 0
    chosen.loc[selected, "baseline_club_assignment_source"] = "confirmed_manual_override"
    return chosen.reset_index(drop=True)


def player_covariates(analytic: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transfermarkt_player_id", "treated_world_cup", "fifa_squad_id", "country_name",
        "statsbomb_player_id", "sb_link_status", "wc_minutes_played", "wc_played",
        "wc_started", "wc_extra_time_game", "wc_max_stage_order", "wc_npxg_p90",
        "wc_xa_p90", "wc_progressive_passes_proxy_p90", "wc_pressures_p90",
        "valuation_date", "market_value_in_eur", "market_value_log", "pre_injury_events",
        "pre_injury_days", "long_injury_crossing", "january_transfer", "age_at_world_cup",
        "pre_club_ppg", "pre_club_gd_per_game", "position_group",
    ]
    available = [column for column in columns if column in analytic.columns]
    return (
        analytic.sort_values("match_date")
        .groupby("transfermarkt_player_id", as_index=False)[available[1:]]
        .first()
    )


def add_understat(grid: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pm = inputs["understat_pm"].copy()
    game_map = inputs["game_map"]
    player_map = inputs["player_map"]
    accepted = player_map[player_map["link_status"].eq("accepted")].copy()
    accepted["understat_player_id"] = pd.to_numeric(
        accepted["understat_player_id"], errors="coerce"
    )
    accepted = accepted.drop_duplicates(
        ["competition_id", "understat_player_id", "transfermarkt_player_id"]
    )
    pm["understat_player_id"] = pd.to_numeric(pm["understat_player_id"], errors="coerce")
    pm = pm.merge(
        game_map[["competition_id", "understat_match_id", "transfermarkt_game_id"]],
        on=["competition_id", "understat_match_id"],
        how="left",
    ).merge(
        accepted[["competition_id", "understat_player_id", "transfermarkt_player_id"]],
        on=["competition_id", "understat_player_id"],
        how="inner",
    )
    metric_columns = ["shots", "goals", "shots_on_target", "penalties", "xg", "npxg", "xa", "key_passes"]
    metrics = pm.groupby(
        ["transfermarkt_game_id", "transfermarkt_player_id"], as_index=False
    )[metric_columns].sum()
    metrics = metrics.rename(
        columns={"transfermarkt_game_id": "game_id", "goals": "understat_goals"}
    )
    linked_ids = set(accepted["transfermarkt_player_id"].dropna().astype(int))
    grid = grid.merge(metrics, on=["game_id", "transfermarkt_player_id"], how="left")
    grid["understat_linked"] = grid["transfermarkt_player_id"].isin(linked_ids).astype("int8")
    metric_columns = ["shots", "understat_goals", "shots_on_target", "penalties", "xg", "npxg", "xa", "key_passes"]
    for column in metric_columns:
        grid.loc[grid["understat_linked"].eq(1), column] = grid.loc[
            grid["understat_linked"].eq(1), column
        ].fillna(0)
    grid["npxg_xa_per_club_match"] = grid["npxg"] + grid["xa"]
    grid["performance_source_inconsistency"] = (
        grid["played"].eq(0)
        & grid[["shots", "understat_goals", "xg", "npxg", "xa", "key_passes"]]
        .fillna(0)
        .ne(0)
        .any(axis=1)
    ).astype("int8")
    return grid


def build_design_grid(
    baseline: pd.DataFrame,
    schedule: pd.DataFrame,
    panel: pd.DataFrame,
    covariates: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    baseline = baseline.merge(covariates, on="transfermarkt_player_id", how="left")
    grid = baseline.merge(
        schedule,
        left_on="baseline_club_id",
        right_on="club_id",
        how="inner",
        validate="many_to_many",
    )
    lineup_columns = [
        "game_id", "transfermarkt_player_id", "club_id", "lineup_position", "started",
        "team_captain", "minutes_played", "goals", "assists", "yellow_cards",
        "red_cards", "played",
    ]
    lineup = panel[lineup_columns].drop_duplicates(
        ["game_id", "transfermarkt_player_id", "club_id"]
    )
    grid = grid.merge(
        lineup,
        on=["game_id", "transfermarkt_player_id", "club_id"],
        how="left",
        indicator="lineup_merge",
    )
    grid["in_match_squad"] = grid["lineup_merge"].eq("both").astype("int8")
    grid = grid.drop(columns="lineup_merge")
    zeros = ["started", "team_captain", "minutes_played", "goals", "assists", "yellow_cards", "red_cards", "played"]
    grid[zeros] = grid[zeros].fillna(0)
    grid["possible_minutes"] = 90
    grid["minutes_share"] = (grid["minutes_played"] / 90).clip(0, 1)
    grid["lineup_source_inconsistency"] = (
        grid["started"].eq(1) & (grid["played"].ne(1) | grid["minutes_played"].le(0))
    ).astype("int8")
    grid["effective_spell_end"] = grid[["spell_end", "last_observed"]].max(axis=1)
    grid["spell_end_adjusted_for_observed_appearance"] = (
        grid["last_observed"].gt(grid["spell_end"])
    ).astype("int8")
    grid["active_at_baseline_club"] = (
        grid["match_date"].between(grid["spell_start"], grid["effective_spell_end"])
    ).astype("int8")
    grid["zero_after_recorded_baseline_club_exit"] = (
        grid["match_date"].gt(grid["effective_spell_end"])
    ).astype("int8")
    grid = add_understat(grid, inputs)
    return grid.sort_values(
        ["transfermarkt_player_id", "match_date", "game_id"]
    ).reset_index(drop=True)


def aggregate_block(grid: pd.DataFrame, low: int, high: int, prefix: str) -> pd.DataFrame:
    block = grid[grid["relative_club_match"].between(low, high)].copy()
    result = block.groupby("transfermarkt_player_id", as_index=False).agg(
        **{
            f"{prefix}_club_games": ("game_id", "nunique"),
            f"{prefix}_minutes_played": ("minutes_played", "sum"),
            f"{prefix}_in_match_squad": ("in_match_squad", "sum"),
            f"{prefix}_played": ("played", "sum"),
            f"{prefix}_started": ("started", "sum"),
            f"{prefix}_npxg": ("npxg", lambda x: x.sum(min_count=1)),
            f"{prefix}_xa": ("xa", lambda x: x.sum(min_count=1)),
            f"{prefix}_npxg_xa": ("npxg_xa_per_club_match", lambda x: x.sum(min_count=1)),
            f"{prefix}_opponent_ppg": ("pre_opponent_ppg", "mean"),
        }
    )
    games = result[f"{prefix}_club_games"].replace(0, np.nan)
    result[f"{prefix}_minutes_share"] = result[f"{prefix}_minutes_played"] / (games * 90)
    result[f"{prefix}_squad_rate"] = result[f"{prefix}_in_match_squad"] / games
    result[f"{prefix}_played_rate"] = result[f"{prefix}_played"] / games
    result[f"{prefix}_start_rate"] = result[f"{prefix}_started"] / games
    return result


def build_cohort(
    baseline: pd.DataFrame, grid: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    design = config["design"]
    cohort = baseline.copy()
    first = grid.sort_values("match_date").groupby("transfermarkt_player_id", as_index=False).first()
    covariate_columns = [
        "transfermarkt_player_id", "treated_world_cup", "fifa_squad_id", "country_name",
        "statsbomb_player_id", "valuation_date", "market_value_in_eur", "market_value_log",
        "pre_injury_events", "pre_injury_days", "long_injury_crossing", "january_transfer",
        "age_at_world_cup", "pre_club_ppg", "pre_club_gd_per_game", "position_group",
        "wc_minutes_played", "wc_played", "wc_started", "wc_extra_time_game",
        "wc_max_stage_order", "wc_npxg_p90", "wc_xa_p90",
        "wc_progressive_passes_proxy_p90", "wc_pressures_p90", "understat_linked",
    ]
    existing = [column for column in covariate_columns if column in first.columns]
    cohort = cohort.merge(first[existing], on="transfermarkt_player_id", how="left")
    pre = aggregate_block(grid, -8, -1, "pre")
    early = aggregate_block(grid, -8, -5, "pre_early")
    late = aggregate_block(grid, -4, -1, "pre_late")
    diagnostic = aggregate_block(grid, -16, -9, "diagnostic_pre")
    for frame in [pre, early, late, diagnostic]:
        cohort = cohort.merge(frame, on="transfermarkt_player_id", how="left")
    for metric in ["minutes_share", "squad_rate", "played_rate", "start_rate", "npxg_xa"]:
        cohort[f"pre_trend_{metric}"] = (
            cohort.get(f"pre_late_{metric}") - cohort.get(f"pre_early_{metric}")
        )
    cohort["outfield"] = (
        cohort["is_goalkeeper"].fillna(0).eq(0)
        & cohort["position_group"].fillna("").ne("Goalkeeper")
    )
    cohort["pre_games_ok"] = cohort["pre_club_games"].fillna(0).ge(
        int(design["baseline_games_threshold"])
    )
    cohort["baseline_minutes_ok"] = cohort["pre_minutes_played"].fillna(0).ge(
        int(design["baseline_minutes_threshold"])
    )
    cohort["baseline_identity_valid"] = (
        cohort["treated_world_cup"].fillna(0).eq(0) | cohort["fifa_squad_id"].notna()
    )
    cohort["eligible_baseline_only"] = (
        cohort["outfield"]
        & cohort["pre_games_ok"]
        & cohort["baseline_minutes_ok"]
        & cohort["baseline_identity_valid"]
    )
    return cohort


def add_national_eligibility(
    cohort: pd.DataFrame, overrides: pd.DataFrame, config: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qualified = {canonical_team(value) for value in config["qualified_teams"]}
    cohort = cohort.copy()
    cohort["citizenship_canonical"] = cohort["country_of_citizenship"].map(canonical_team)
    cohort["official_team_canonical"] = cohort["country_name"].map(canonical_team)
    cohort["national_team_eligible"] = np.where(
        cohort["treated_world_cup"].fillna(0).eq(1),
        cohort["official_team_canonical"],
        cohort["citizenship_canonical"],
    )
    cohort["national_eligibility_source"] = np.where(
        cohort["treated_world_cup"].fillna(0).eq(1),
        "official_final_squad",
        "transfermarkt_primary_citizenship_proxy",
    )
    manual = overrides[
        overrides["override_type"].eq("national_team")
        & overrides["decision"].eq("confirmed")
    ].copy()
    manual["national_team"] = manual["national_team"].map(canonical_team)
    manual_map = manual.set_index("transfermarkt_player_id")["national_team"]
    national_override = cohort["transfermarkt_player_id"].isin(manual_map.index)
    cohort.loc[national_override, "national_team_eligible"] = cohort.loc[
        national_override, "transfermarkt_player_id"
    ].map(manual_map)
    cohort.loc[national_override, "national_eligibility_source"] = "confirmed_manual_override"
    cohort["national_eligibility_confidence"] = np.select(
        [
            cohort["treated_world_cup"].fillna(0).eq(1),
            cohort["citizenship_canonical"].notna(),
        ],
        ["high", "provisional"],
        default="missing",
    )
    cohort.loc[national_override, "national_eligibility_confidence"] = "high"
    cohort["national_team_override_applied"] = national_override.astype("int8")
    cohort["treated_country_citizenship_mismatch"] = (
        cohort["treated_world_cup"].fillna(0).eq(1)
        & cohort["citizenship_canonical"].notna()
        & cohort["official_team_canonical"].ne(cohort["citizenship_canonical"])
    ).astype("int8")
    cohort["dual_nationality_not_observed"] = (
        cohort["treated_world_cup"].fillna(0).eq(0)
    ).astype("int8")
    cohort["national_eligibility_review_required"] = (
        cohort["national_team_eligible"].isna()
        | cohort["treated_country_citizenship_mismatch"].eq(1)
    ).astype("int8")
    cohort.loc[national_override, "national_eligibility_review_required"] = 0
    cohort["qualified_national_team"] = cohort["national_team_eligible"].isin(qualified)

    support_base = cohort[cohort["eligible_baseline_only"] & cohort["qualified_national_team"]]
    support = support_base.groupby("national_team_eligible", dropna=False).agg(
        nation_treated_count=("treated_world_cup", lambda x: int(x.fillna(0).eq(1).sum())),
        nation_control_count=("treated_world_cup", lambda x: int(x.fillna(0).eq(0).sum())),
    ).reset_index()
    support["nation_control_treated_ratio"] = np.where(
        support["nation_treated_count"].gt(0),
        support["nation_control_count"] / support["nation_treated_count"],
        np.nan,
    )
    cohort = cohort.merge(support, on="national_team_eligible", how="left")
    cohort["nation_support_any"] = (
        cohort["nation_treated_count"].fillna(0).gt(0)
        & cohort["nation_control_count"].fillna(0).gt(0)
    )
    cohort["nation_support_ratio_ge_1"] = cohort[
        "nation_control_treated_ratio"
    ].fillna(0).ge(1)
    cohort["eligible_qualified_risk_set"] = (
        cohort["eligible_baseline_only"] & cohort["qualified_national_team"]
    )
    cohort["eligible_same_nation_support"] = (
        cohort["eligible_qualified_risk_set"] & cohort["nation_support_any"]
    )
    cohort["eligible_same_nation_ratio_ge_1"] = (
        cohort["eligible_qualified_risk_set"] & cohort["nation_support_ratio_ge_1"]
    )

    eligibility_columns = [
        "transfermarkt_player_id", "player_name", "treated_world_cup",
        "country_of_citizenship", "citizenship_canonical", "country_name",
        "official_team_canonical", "national_team_eligible", "qualified_national_team",
        "national_eligibility_source", "national_eligibility_confidence",
        "national_team_override_applied",
        "dual_nationality_not_observed", "treated_country_citizenship_mismatch",
        "national_eligibility_review_required", "nation_treated_count",
        "nation_control_count", "nation_control_treated_ratio", "nation_support_any",
        "nation_support_ratio_ge_1", "eligible_baseline_only",
        "eligible_qualified_risk_set", "eligible_same_nation_support",
        "eligible_same_nation_ratio_ge_1",
    ]
    eligibility = cohort[eligibility_columns].copy()
    return cohort, eligibility


def build_sample_flow(cohort: pd.DataFrame) -> pd.DataFrame:
    masks: list[tuple[str, pd.Series]] = []
    current = pd.Series(True, index=cohort.index)
    masks.append(("01_baseline_club_assigned", current.copy()))
    current &= cohort["outfield"]
    masks.append(("02_outfield", current.copy()))
    current &= cohort["pre_games_ok"]
    masks.append(("03_minimum_pre_games", current.copy()))
    current &= cohort["baseline_minutes_ok"]
    masks.append(("04_minimum_pre_minutes", current.copy()))
    current &= cohort["baseline_identity_valid"]
    masks.append(("05_baseline_only_eligible", current.copy()))
    current &= cohort["qualified_national_team"]
    masks.append(("06_qualified_nation_risk_set", current.copy()))
    current &= cohort["nation_support_any"]
    masks.append(("07_same_nation_any_support", current.copy()))
    rows: list[dict[str, object]] = []
    treated = cohort["treated_world_cup"].fillna(0).eq(1)
    for step, mask in masks:
        rows.append(
            {
                "step": step,
                "total": int(mask.sum()),
                "treated": int((mask & treated).sum()),
                "controls": int((mask & ~treated).sum()),
                "share_of_initial": float(mask.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_missingness(cohort: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "age_at_world_cup", "market_value_in_eur", "pre_minutes_played",
        "pre_start_rate", "pre_npxg", "pre_xa", "pre_injury_days",
        "country_of_citizenship", "national_team_eligible", "pre_trend_minutes_share",
        "pre_trend_npxg_xa",
    ]
    populations = {
        "baseline_only": cohort["eligible_baseline_only"],
        "qualified_risk_set": cohort["eligible_qualified_risk_set"],
    }
    rows: list[dict[str, object]] = []
    for population, mask in populations.items():
        frame = cohort[mask]
        for treated_value, group in frame.groupby("treated_world_cup", dropna=False):
            for variable in variables:
                missing = int(group[variable].isna().sum())
                rows.append(
                    {
                        "population": population,
                        "treated_world_cup": int(treated_value),
                        "variable": variable,
                        "n": len(group),
                        "missing": missing,
                        "missing_rate": missing / len(group) if len(group) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def regression_cluster_se(
    outcome: np.ndarray, treated: np.ndarray, clusters: np.ndarray
) -> tuple[float, float, np.ndarray]:
    x = np.column_stack([np.ones(len(outcome)), treated.astype(float)])
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ outcome
    residual = outcome - x @ beta
    meat = np.zeros((2, 2))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        selector = clusters == cluster
        score = x[selector].T @ residual[selector]
        meat += np.outer(score, score)
    n, k = x.shape
    g = len(unique_clusters)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1
    covariance = correction * bread @ meat @ bread
    return float(beta[1]), float(math.sqrt(max(covariance[1, 1], 0))), residual


def build_power_diagnostics(
    cohort: pd.DataFrame, grid: pd.DataFrame, config: dict[str, object]
) -> pd.DataFrame:
    risk = cohort[cohort["eligible_qualified_risk_set"]].copy()
    player_info = risk.set_index("transfermarkt_player_id")[[
        "treated_world_cup", "baseline_club_id", "league_name", "position_group"
    ]]
    pre = grid[
        grid["transfermarkt_player_id"].isin(player_info.index)
        & grid["relative_club_match"].between(-8, -1)
    ].copy()
    outcomes = [
        "minutes_share", "in_match_squad", "played", "started",
        "npxg_xa_per_club_match",
    ]
    alpha = float(config["power"]["alpha"])
    target_power = float(config["power"]["target_power"])
    simulations = int(config["power"]["simulations"])
    multipliers = [float(value) for value in config["power"]["effect_multipliers"]]
    rng = np.random.default_rng(int(config["random_seed"]))
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(target_power)
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        early = pre[pre["relative_club_match"].between(-8, -5)].groupby(
            "transfermarkt_player_id"
        )[outcome].mean()
        late = pre[pre["relative_club_match"].between(-4, -1)].groupby(
            "transfermarkt_player_id"
        )[outcome].mean()
        delta = (late - early).rename("delta").dropna().to_frame().join(player_info, how="inner")
        if delta.empty or delta["treated_world_cup"].nunique() < 2:
            continue
        strata_mean = delta.groupby(["league_name", "position_group"])["delta"].transform("mean")
        delta["residualized_delta"] = delta["delta"] - strata_mean
        y = delta["residualized_delta"].to_numpy(float)
        t = delta["treated_world_cup"].to_numpy(int)
        clusters = delta["baseline_club_id"].to_numpy(int)
        _, se0, null_residual = regression_cluster_se(y, t, clusters)
        mde = (z_alpha + z_power) * se0
        baseline_treated = early.reindex(delta.index[t == 1]).mean()
        unique_clusters = np.unique(clusters)
        cluster_lookup = {cluster: index for index, cluster in enumerate(unique_clusters)}
        cluster_index = np.array([cluster_lookup[cluster] for cluster in clusters])
        membership = np.zeros((len(y), len(unique_clusters)))
        membership[np.arange(len(y)), cluster_index] = 1.0
        x = np.column_stack([np.ones(len(y)), t.astype(float)])
        bread = np.linalg.inv(x.T @ x)
        n, k = x.shape
        g = len(unique_clusters)
        correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1
        variance_vector = bread[1, :]
        effect_specs = [(f"{multiplier:g}x_mde", multiplier, multiplier * mde) for multiplier in multipliers]
        if outcome == config["sesoi"]["primary_outcome"]:
            effect_specs.append(("primary_sesoi", np.nan, float(config["sesoi"]["absolute_effect"])))
        for effect_label, multiplier, effect in effect_specs:
            signs = rng.choice([-1.0, 1.0], size=(simulations, len(unique_clusters)))
            simulated = null_residual[None, :] * signs[:, cluster_index] + effect * t[None, :]
            control_mean = simulated[:, t == 0].mean(axis=1)
            treated_mean = simulated[:, t == 1].mean(axis=1)
            estimates = treated_mean - control_mean
            residuals = simulated - control_mean[:, None] - estimates[:, None] * t[None, :]
            score_intercept = residuals @ membership
            score_treatment = (residuals * t[None, :]) @ membership
            meat_00 = np.square(score_intercept).sum(axis=1)
            meat_01 = (score_intercept * score_treatment).sum(axis=1)
            meat_11 = np.square(score_treatment).sum(axis=1)
            variances = correction * (
                variance_vector[0] ** 2 * meat_00
                + 2 * variance_vector[0] * variance_vector[1] * meat_01
                + variance_vector[1] ** 2 * meat_11
            )
            standard_errors = np.sqrt(np.maximum(variances, 0))
            rejected = int(
                (np.abs(estimates / np.where(standard_errors > 0, standard_errors, np.nan)) > z_alpha)
                .sum()
            )
            rows.append(
                {
                    "outcome": outcome,
                    "method": "pre_period_wild_cluster_simulation",
                    "n_players": len(delta),
                    "n_treated": int(t.sum()),
                    "n_controls": int((1 - t).sum()),
                    "n_clubs": len(unique_clusters),
                    "alpha": alpha,
                    "target_power": target_power,
                    "cluster_se_under_null": se0,
                    "analytic_mde_80": mde,
                    "baseline_treated_mean": baseline_treated,
                    "mde_share_of_baseline": mde / baseline_treated if baseline_treated not in [0, np.nan] and pd.notna(baseline_treated) else np.nan,
                    "effect_label": effect_label,
                    "effect_multiplier": multiplier,
                    "simulated_effect": effect,
                    "simulated_power": rejected / simulations,
                    "simulations": simulations,
                    "uses_post_treatment_outcomes": 0,
                }
            )
    return pd.DataFrame(rows)


def build_review_outputs(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_columns = [
        "transfermarkt_player_id", "player_name", "baseline_club_id",
        "baseline_club_name", "baseline_club_assignment_source",
        "baseline_club_candidate_count", "active_spell_candidate_count",
        "last_pre_observed", "spell_start", "spell_end", "eligible_baseline_only",
        "treated_world_cup",
    ]
    baseline_review = cohort.loc[
        cohort["baseline_club_assignment_review"].eq(1), baseline_columns
    ].copy()
    baseline_review["review_reason"] = np.where(
        baseline_review["active_spell_candidate_count"].eq(0),
        "no_active_spell_at_cutoff_used_last_pre_observation",
        "multiple_active_spells_at_cutoff",
    )
    baseline_review = baseline_review.sort_values(
        ["eligible_baseline_only", "treated_world_cup", "player_name"],
        ascending=[False, False, True],
    )

    national_columns = [
        "transfermarkt_player_id", "player_name", "treated_world_cup",
        "country_of_citizenship", "country_name", "citizenship_canonical",
        "official_team_canonical", "national_team_eligible",
        "national_eligibility_source", "national_eligibility_confidence",
        "treated_country_citizenship_mismatch", "eligible_baseline_only",
        "eligible_qualified_risk_set",
    ]
    national_review = cohort.loc[
        cohort["national_eligibility_review_required"].eq(1), national_columns
    ].copy()
    national_review["review_reason"] = np.select(
        [
            national_review["national_team_eligible"].isna(),
            national_review["treated_country_citizenship_mismatch"].eq(1),
        ],
        ["missing_primary_citizenship", "official_team_citizenship_mismatch"],
        default="other",
    )
    national_review = national_review.sort_values(
        ["eligible_baseline_only", "treated_world_cup", "player_name"],
        ascending=[False, False, True],
    )

    support = cohort.groupby("national_team_eligible", dropna=False, as_index=False).agg(
        all_players=("transfermarkt_player_id", "size"),
        baseline_eligible=("eligible_baseline_only", "sum"),
        qualified_risk_set=("eligible_qualified_risk_set", "sum"),
        treated=("treated_world_cup", lambda x: int(x.fillna(0).eq(1).sum())),
        controls=("treated_world_cup", lambda x: int(x.fillna(0).eq(0).sum())),
        qualified_national_team=("qualified_national_team", "max"),
        eligible_treated=(
            "treated_world_cup",
            lambda x: int(
                (
                    x.fillna(0).eq(1)
                    & cohort.loc[x.index, "eligible_qualified_risk_set"]
                ).sum()
            ),
        ),
        eligible_controls=(
            "treated_world_cup",
            lambda x: int(
                (
                    x.fillna(0).eq(0)
                    & cohort.loc[x.index, "eligible_qualified_risk_set"]
                ).sum()
            ),
        ),
    )
    support["eligible_control_treated_ratio"] = np.where(
        support["eligible_treated"].gt(0),
        support["eligible_controls"] / support["eligible_treated"],
        np.nan,
    )
    support["same_nation_overlap"] = (
        support["eligible_treated"].gt(0) & support["eligible_controls"].gt(0)
    )
    return baseline_review, national_review, support


def build_qa(
    cohort: pd.DataFrame,
    grid: pd.DataFrame,
    eligibility: pd.DataFrame,
    power: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    cutoff = pd.Timestamp(config["design"]["baseline_cutoff"])
    expected_post_games = int(config["design"]["post_games"])
    forbidden = set(config["post_treatment_variables_forbidden_in_eligibility"])
    pre_counts = grid[grid["relative_club_match"].lt(0)].groupby(
        "transfermarkt_player_id"
    )["game_id"].nunique()
    post_counts = grid[grid["relative_club_match"].gt(0)].groupby(
        "transfermarkt_player_id"
    )["game_id"].nunique()
    treated = cohort["treated_world_cup"].fillna(0).eq(1)
    eligible = cohort["eligible_baseline_only"]
    risk = cohort["eligible_qualified_risk_set"]
    exited = grid["zero_after_recorded_baseline_club_exit"].eq(1)
    tests = [
        ("cohort_unique_player", len(cohort), cohort["transfermarkt_player_id"].nunique(), not cohort["transfermarkt_player_id"].duplicated().any()),
        ("grid_unique_player_game", len(grid), len(grid.drop_duplicates(["transfermarkt_player_id", "game_id"])), not grid.duplicated(["transfermarkt_player_id", "game_id"]).any()),
        ("minimum_eight_pre_games", int(pre_counts.ge(8).sum()), len(pre_counts), pre_counts.ge(8).all()),
        ("exactly_configured_post_games", int(post_counts.eq(expected_post_games).sum()), len(post_counts), post_counts.eq(expected_post_games).all()),
        ("eligibility_uses_no_forbidden_post_columns", len(ELIGIBILITY_INPUT_COLUMNS & forbidden), 0, not (ELIGIBILITY_INPUT_COLUMNS & forbidden)),
        ("treated_have_official_identity", int(cohort.loc[treated, "fifa_squad_id"].notna().sum()), int(treated.sum()), cohort.loc[treated, "fifa_squad_id"].notna().all()),
        ("treated_official_teams_are_qualified", int(cohort.loc[treated, "qualified_national_team"].sum()), int(treated.sum()), cohort.loc[treated, "qualified_national_team"].all()),
        ("valuations_not_after_cutoff", int(cohort["valuation_date"].isna().sum() + cohort["valuation_date"].le(cutoff).sum()), len(cohort), (cohort["valuation_date"].isna() | cohort["valuation_date"].le(cutoff)).all()),
        ("baseline_eligible_players", int(eligible.sum()), ">=1000", eligible.sum() >= 1000),
        ("qualified_risk_set_treated", int((risk & treated).sum()), ">=200", (risk & treated).sum() >= 200),
        ("qualified_risk_set_controls", int((risk & ~treated).sum()), ">=300", (risk & ~treated).sum() >= 300),
        ("post_treatment_injuries_retained", int(cohort.loc[eligible, "long_injury_crossing"].fillna(0).sum()), ">0 informational", cohort.loc[eligible, "long_injury_crossing"].fillna(0).sum() > 0),
        ("post_treatment_transfers_retained", int(cohort.loc[eligible, "january_transfer"].fillna(0).sum()), ">0 informational", cohort.loc[eligible, "january_transfer"].fillna(0).sum() > 0),
        ("recorded_exit_rows_zero_minutes", int(grid.loc[exited, "minutes_played"].eq(0).sum()), int(exited.sum()), grid.loc[exited, "minutes_played"].eq(0).all()),
        ("npxg_not_above_xg", int((grid["npxg"].isna() | grid["xg"].isna() | grid["npxg"].le(grid["xg"] + 1e-10)).sum()), len(grid), (grid["npxg"].isna() | grid["xg"].isna() | grid["npxg"].le(grid["xg"] + 1e-10)).all()),
        ("eligibility_table_unique_player", len(eligibility), eligibility["transfermarkt_player_id"].nunique(), not eligibility["transfermarkt_player_id"].duplicated().any()),
        ("power_uses_pre_only", int(power["uses_post_treatment_outcomes"].eq(0).sum()), len(power), not power.empty and power["uses_post_treatment_outcomes"].eq(0).all()),
        ("power_primary_outcome_present", int(power["outcome"].eq("minutes_share").sum()), ">0", power["outcome"].eq("minutes_share").any()),
        ("power_primary_sesoi_present", int(power["effect_label"].eq("primary_sesoi").sum()), 1, power["effect_label"].eq("primary_sesoi").sum() == 1),
        ("lineup_source_inconsistencies_flagged", int(grid["lineup_source_inconsistency"].sum()), "informational", True),
        ("performance_source_inconsistencies_flagged", int(grid["performance_source_inconsistency"].sum()), "informational", True),
        ("baseline_assignment_reviews", int(cohort["baseline_club_assignment_review"].sum()), "informational", True),
        ("national_eligibility_reviews", int(cohort["national_eligibility_review_required"].sum()), "informational", True),
        ("baseline_manual_overrides_applied", int(cohort["baseline_club_override_applied"].sum()), 4, cohort["baseline_club_override_applied"].sum() == 4),
        ("national_manual_overrides_applied", int(cohort["national_team_override_applied"].sum()), 8, cohort["national_team_override_applied"].sum() == 8),
    ]
    return pd.DataFrame(tests, columns=["check", "observed", "expected", "passed"])


def build_dictionary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    units = {
        "analysis_sample_baseline_only.csv": "jogador",
        "design_player_match.csv": "jogador + partida do clube-base",
        "national_team_eligibility.csv": "jogador",
        "sample_flow.csv": "etapa de seleção",
        "missingness.csv": "população + tratamento + variável",
        "mde_power.csv": "outcome + tamanho de efeito",
        "design_qa.csv": "controle de qualidade",
        "baseline_assignment_review.csv": "jogador sinalizado para revisão de clube-base",
        "national_eligibility_review.csv": "jogador sinalizado para revisão nacional",
        "national_team_support.csv": "seleção nacional elegível",
    }
    special = {
        "eligible_baseline_only": "Elegibilidade definida somente com informação até 13/11/2022.",
        "eligible_qualified_risk_set": "Elegível baseline e ligado a seleção classificada.",
        "national_eligibility_confidence": "Confiança da atribuição de seleção elegível.",
        "dual_nationality_not_observed": "Indica que a fonte usada não representa dupla nacionalidade.",
        "zero_after_recorded_baseline_club_exit": "Partida posterior ao fim registrado do vínculo com o clube-base.",
        "analytic_mde_80": "Menor efeito detectável aproximado com poder de 80%.",
    }
    for filename, frame in outputs.items():
        for column in frame.columns:
            rows.append(
                {
                    "file": filename,
                    "unit_of_observation": units[filename],
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "description": special.get(
                        column,
                        "Campo de fonte ou derivado; regra reproduzível em build_design_phase.py.",
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(frame):,} rows x {len(frame.columns)} cols")


def write_manifest(paths: list[Path], config_hash: str, overrides_hash: str) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        with path.open("rb") as stream:
            row_count = max(sum(1 for _ in stream) - 1, 0)
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "rows": row_count,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "config_sha256": config_hash,
                "manual_overrides_sha256": overrides_hash,
                "generated_by": "scripts/analysis/build_design_phase.py",
            }
        )
    write_csv(pd.DataFrame(rows), REPORTS / "design_manifest.csv")


def main() -> None:
    config = load_config()
    config_hash = sha256(CONFIG_PATH)
    overrides_hash = sha256(OVERRIDES_PATH)
    inputs = read_inputs()
    schedule = build_schedule(inputs["panel"], config)
    baseline = assign_baseline_club(
        inputs["panel"], inputs["spells"], inputs["overrides"], config
    )
    covariates = player_covariates(inputs["analytic"])
    grid = build_design_grid(baseline, schedule, inputs["panel"], covariates, inputs)
    cohort = build_cohort(baseline, grid, config)
    cohort, eligibility = add_national_eligibility(cohort, inputs["overrides"], config)
    cohort["analysis_config_sha256"] = config_hash
    cohort["manual_overrides_sha256"] = overrides_hash
    grid = grid.merge(
        cohort[[
            "transfermarkt_player_id", "eligible_baseline_only",
            "eligible_qualified_risk_set", "national_team_eligible",
        ]],
        on="transfermarkt_player_id",
        how="left",
    )
    grid["analysis_config_sha256"] = config_hash
    grid["manual_overrides_sha256"] = overrides_hash
    sample_flow = build_sample_flow(cohort)
    missingness = build_missingness(cohort)
    power = build_power_diagnostics(cohort, grid, config)
    baseline_review, national_review, national_support = build_review_outputs(cohort)
    qa = build_qa(cohort, grid, eligibility, power, config)

    outputs = {
        "analysis_sample_baseline_only.csv": cohort,
        "design_player_match.csv": grid,
        "national_team_eligibility.csv": eligibility,
        "sample_flow.csv": sample_flow,
        "missingness.csv": missingness,
        "mde_power.csv": power,
        "design_qa.csv": qa,
        "baseline_assignment_review.csv": baseline_review,
        "national_eligibility_review.csv": national_review,
        "national_team_support.csv": national_support,
    }
    dictionary = build_dictionary(outputs)
    output_paths = [
        PROCESSED / "analysis_sample_baseline_only.csv",
        PROCESSED / "design_player_match.csv",
        PROCESSED / "national_team_eligibility.csv",
        REPORTS / "sample_flow.csv",
        REPORTS / "missingness.csv",
        REPORTS / "mde_power.csv",
        REPORTS / "design_qa.csv",
        REPORTS / "baseline_assignment_review.csv",
        REPORTS / "national_eligibility_review.csv",
        REPORTS / "national_team_support.csv",
        REPORTS / "design_data_dictionary.csv",
    ]
    for (filename, frame), path in zip(outputs.items(), output_paths[:-1]):
        if path.name != filename:
            raise RuntimeError(f"Mapeamento de saída incorreto: {filename} != {path.name}")
        write_csv(frame, path)
    write_csv(dictionary, output_paths[-1])
    write_manifest(output_paths, config_hash, overrides_hash)
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(f"QA da Fase 4 falhou: {failed}")
    print("Fase 4 concluída: protocolo, coorte baseline-only e diagnósticos aprovados.")


if __name__ == "__main__":
    main()
