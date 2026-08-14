"""Placebo sazonal com dados reais de 2023/24 na pausa FIFA de novembro.

O indicador de tratamento é a convocação oficial para Qatar 2022. Portanto, esta
não é uma estimativa de efeito da Copa: é uma falsificação para verificar se a
coorte que foi convocada em 2022 apresentava uma mudança diferencial de
utilização em uma temporada posterior sem Copa no meio do calendário.

O script não altera painéis, pesos ou relatórios congelados de 2022/23.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import build_weighting_phase as weighting  # noqa: E402
from build_confirmatory_phase import fit_hdfe, model_rows, weighted_summary_scale  # noqa: E402

RAW = ROOT / "data" / "raw" / "transfermarkt" / "dcaribou_2023_24"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "placebo_2023_24"
CONFIG_PATH = ROOT / "config" / "placebo_2023_24.yml"

BIG5 = {"GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}
QUALIFIED_TEAMS = {
    "Argentina", "Australia", "Belgium", "Brazil", "Cameroon", "Canada",
    "Costa Rica", "Croatia", "Denmark", "Ecuador", "England", "France",
    "Germany", "Ghana", "Iran", "Japan", "Korea Republic", "Mexico",
    "Morocco", "Netherlands", "Poland", "Portugal", "Qatar", "Saudi Arabia",
    "Senegal", "Serbia", "Spain", "Switzerland", "Tunisia", "USA", "Uruguay", "Wales",
}
TEAM_ALIASES = {
    "United States": "USA", "US": "USA", "South Korea": "Korea Republic",
    "Korea, South": "Korea Republic", "IR Iran": "Iran", "Türkiye": "Turkey",
}
SOURCE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
SOURCE_FILES = [
    "games.csv.gz", "appearances.csv.gz", "game_lineups.csv.gz", "players.csv.gz",
    "player_valuations.csv.gz", "transfers.csv.gz",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / filename
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(frame):,} rows x {len(frame.columns)} cols")


def canonical_team(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    return TEAM_ALIASES.get(value, value)


def assert_sources() -> None:
    missing = [name for name in SOURCE_FILES if not (RAW / name).exists()]
    if missing:
        raise RuntimeError(
            "Dados brutos ausentes: " + ", ".join(missing) + ". "
            "Baixe o snapshot público do transfermarkt-datasets antes de executar."
        )


def read_games() -> pd.DataFrame:
    games = pd.read_csv(RAW / "games.csv.gz", parse_dates=["date"], low_memory=False)
    games = games.loc[
        games["season"].eq(2023) & games["competition_id"].isin(BIG5)
    ].copy()
    if len(games) != 1752:
        raise RuntimeError(f"Cobertura Big Five 2023/24 inválida: {len(games)} jogos, esperado 1752")
    games["game_id"] = pd.to_numeric(games["game_id"], errors="raise").astype("int64")
    return games


def read_filtered(path: Path, game_ids: set[int], columns: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000, low_memory=False):
        chunk["game_id"] = pd.to_numeric(chunk["game_id"], errors="coerce")
        selected = chunk[chunk["game_id"].isin(game_ids)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def club_schedule(games: pd.DataFrame, cutoff: pd.Timestamp, post_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    home = games.rename(columns={
        "home_club_id": "club_id", "home_club_name": "club_name", "away_club_id": "opponent_club_id",
        "away_club_name": "opponent_club_name", "home_club_goals": "club_goals",
        "away_club_goals": "opponent_goals", "home_club_position": "club_table_position",
        "away_club_position": "opponent_table_position", "home_club_formation": "club_formation",
    }).copy()
    home["is_home"] = 1
    away = games.rename(columns={
        "away_club_id": "club_id", "away_club_name": "club_name", "home_club_id": "opponent_club_id",
        "home_club_name": "opponent_club_name", "away_club_goals": "club_goals",
        "home_club_goals": "opponent_goals", "away_club_position": "club_table_position",
        "home_club_position": "opponent_table_position", "away_club_formation": "club_formation",
    }).copy()
    away["is_home"] = 0
    columns = [
        "game_id", "date", "competition_id", "club_id", "club_name", "opponent_club_id", "opponent_club_name",
        "is_home", "club_goals", "opponent_goals", "round", "club_table_position", "opponent_table_position", "club_formation",
    ]
    full = pd.concat([home[columns], away[columns]], ignore_index=True).rename(columns={"date": "match_date"})
    full["league_name"] = full["competition_id"].map(BIG5)
    full = full.sort_values(["club_id", "match_date", "game_id"]).reset_index(drop=True)
    full["days_since_club_game"] = full.groupby("club_id")["match_date"].diff().dt.days

    selected: list[pd.DataFrame] = []
    for club_id, group in full.groupby("club_id", sort=True):
        pre = group[group["match_date"].le(cutoff)].tail(8).copy()
        post = group[group["match_date"].ge(post_start)].head(8).copy()
        if len(pre) != 8 or len(post) != 8:
            raise RuntimeError(f"Clube {club_id} sem oito jogos na janela placebo: pre={len(pre)} post={len(post)}")
        pre["relative_club_match"] = np.arange(-8, 0)
        post["relative_club_match"] = np.arange(1, 9)
        selected.extend([pre, post])
    schedule = pd.concat(selected, ignore_index=True)
    schedule["window"] = np.select(
        [schedule["relative_club_match"].lt(0), schedule["relative_club_match"].between(1, 4), schedule["relative_club_match"].between(5, 8)],
        ["pre", "post_acute", "post_consolidated"],
        default="invalid",
    )
    return full, schedule.sort_values(["club_id", "match_date", "game_id"]).reset_index(drop=True)


def treatment_and_eligibility(players: pd.DataFrame) -> pd.DataFrame:
    crosswalk = pd.read_csv(INTERIM / "fifa_player_crosswalk.csv", low_memory=False)
    treated = crosswalk[
        crosswalk["tm_link_status"].eq("accepted") & crosswalk["transfermarkt_player_id"].notna()
    ][["transfermarkt_player_id", "fifa_squad_id", "country_name"]].copy()
    treated["transfermarkt_player_id"] = pd.to_numeric(treated["transfermarkt_player_id"], errors="raise").astype("int64")
    treated = treated.drop_duplicates("transfermarkt_player_id")

    eligibility = pd.read_csv(PROCESSED / "national_team_eligibility.csv", low_memory=False)
    eligibility = eligibility[["transfermarkt_player_id", "national_team_eligible"]].drop_duplicates("transfermarkt_player_id")
    eligibility["transfermarkt_player_id"] = pd.to_numeric(eligibility["transfermarkt_player_id"], errors="coerce").astype("Int64")

    result = players.merge(treated, on="transfermarkt_player_id", how="left", validate="one_to_one")
    result = result.merge(eligibility, on="transfermarkt_player_id", how="left", validate="one_to_one")
    result["citizenship_canonical"] = result["country_of_citizenship"].map(canonical_team)
    result["national_team_eligible"] = result["national_team_eligible"].fillna(result["citizenship_canonical"])
    result["treated_world_cup"] = result["fifa_squad_id"].notna().astype("int8")
    result["qualified_national_team"] = result["national_team_eligible"].isin(QUALIFIED_TEAMS)
    return result


def latest_valuation(player_ids: set[int], cutoff: pd.Timestamp) -> pd.DataFrame:
    values = pd.read_csv(RAW / "player_valuations.csv.gz", parse_dates=["date"], low_memory=False)
    values["player_id"] = pd.to_numeric(values["player_id"], errors="coerce").astype("Int64")
    values = values[values["player_id"].isin(player_ids) & values["date"].le(cutoff)].copy()
    values = values.sort_values(["player_id", "date"]).drop_duplicates("player_id", keep="last")
    values = values.rename(columns={"player_id": "transfermarkt_player_id", "date": "valuation_date"})
    return values[["transfermarkt_player_id", "valuation_date", "market_value_in_eur"]]


def transfer_end_dates(player_ids: set[int], cutoff: pd.Timestamp, baseline: pd.DataFrame) -> pd.DataFrame:
    transfers = pd.read_csv(RAW / "transfers.csv.gz", parse_dates=["transfer_date"], low_memory=False)
    transfers["player_id"] = pd.to_numeric(transfers["player_id"], errors="coerce").astype("Int64")
    transfers["from_club_id"] = pd.to_numeric(transfers["from_club_id"], errors="coerce").astype("Int64")
    transfers = transfers[transfers["player_id"].isin(player_ids) & transfers["transfer_date"].gt(cutoff)].copy()
    key = baseline[["transfermarkt_player_id", "baseline_club_id"]].rename(columns={"baseline_club_id": "from_club_id"})
    transfers = transfers.merge(key, left_on=["player_id", "from_club_id"], right_on=["transfermarkt_player_id", "from_club_id"], how="inner")
    end = transfers.groupby("transfermarkt_player_id", as_index=False)["transfer_date"].min().rename(columns={"transfer_date": "spell_end"})
    return end


def slope(values: pd.Series, times: pd.Series) -> float:
    valid = values.notna() & times.notna()
    if valid.sum() < 2:
        return 0.0
    x = times[valid].to_numpy(float)
    y = values[valid].to_numpy(float)
    denominator = float(np.sum((x - x.mean()) ** 2))
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denominator) if denominator else 0.0


def prepare_panel(config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(config["baseline_cutoff"])
    post_start = pd.Timestamp(config["post_start"])
    games = read_games()
    full_schedule, schedule = club_schedule(games, cutoff, post_start)
    selected_ids = set(schedule["game_id"].astype(int))

    lineups = read_filtered(
        RAW / "game_lineups.csv.gz", selected_ids,
        ["game_id", "player_id", "club_id", "player_name", "type", "position", "team_captain"],
    )
    lineups["player_id"] = pd.to_numeric(lineups["player_id"], errors="coerce").astype("Int64")
    lineups["club_id"] = pd.to_numeric(lineups["club_id"], errors="coerce").astype("Int64")
    lineups = lineups.dropna(subset=["player_id", "club_id"]).copy()
    lineups["player_id"] = lineups["player_id"].astype("int64")
    lineups["club_id"] = lineups["club_id"].astype("int64")
    lineups = lineups.drop_duplicates(["game_id", "player_id", "club_id"], keep="first")

    appearances = read_filtered(
        RAW / "appearances.csv.gz", selected_ids,
        ["game_id", "player_id", "player_club_id", "minutes_played", "goals", "assists", "yellow_cards", "red_cards"],
    )
    appearances["player_id"] = pd.to_numeric(appearances["player_id"], errors="coerce").astype("Int64")
    appearances["player_club_id"] = pd.to_numeric(appearances["player_club_id"], errors="coerce").astype("Int64")
    appearances = appearances.dropna(subset=["player_id", "player_club_id"]).copy()
    appearances["player_id"] = appearances["player_id"].astype("int64")
    appearances["player_club_id"] = appearances["player_club_id"].astype("int64")
    appearances = appearances.groupby(["game_id", "player_id", "player_club_id"], as_index=False).agg(
        minutes_played=("minutes_played", "max"), goals=("goals", "max"), assists=("assists", "max"),
        yellow_cards=("yellow_cards", "max"), red_cards=("red_cards", "max"),
    )

    players = pd.read_csv(RAW / "players.csv.gz", low_memory=False)
    players = players.rename(columns={"player_id": "transfermarkt_player_id", "name": "player_name", "position": "broad_position", "sub_position": "sub_position"})
    players["transfermarkt_player_id"] = pd.to_numeric(players["transfermarkt_player_id"], errors="coerce").astype("Int64")
    players = players.dropna(subset=["transfermarkt_player_id"]).copy()
    players["transfermarkt_player_id"] = players["transfermarkt_player_id"].astype("int64")
    players = treatment_and_eligibility(players)

    pre = schedule[schedule["relative_club_match"].lt(0)][["game_id", "match_date", "club_id"]]
    observed = lineups.merge(pre, on=["game_id", "club_id"], how="inner")
    baseline = observed.sort_values(["player_id", "match_date", "game_id"]).drop_duplicates("player_id", keep="last")
    baseline = baseline.rename(columns={"player_id": "transfermarkt_player_id", "club_id": "baseline_club_id", "player_name": "lineup_player_name"})
    baseline = baseline.merge(
        players[["transfermarkt_player_id", "player_name", "date_of_birth", "broad_position", "sub_position", "country_of_citizenship", "treated_world_cup", "fifa_squad_id", "country_name", "national_team_eligible", "qualified_national_team"]],
        on="transfermarkt_player_id", how="left", validate="one_to_one",
    )
    baseline["player_name"] = baseline["player_name"].fillna(baseline["lineup_player_name"])
    baseline["position_group"] = baseline["broad_position"].fillna("").replace({"Missing": "Unknown"})
    baseline["is_goalkeeper"] = baseline["position_group"].eq("Goalkeeper") | baseline["position"].fillna("").str.contains("Goalkeeper", case=False)
    baseline = baseline.merge(
        schedule[["club_id", "club_name", "competition_id", "league_name"]].drop_duplicates().rename(columns={"club_id": "baseline_club_id", "club_name": "baseline_club_name"}),
        on="baseline_club_id", how="left", validate="many_to_one",
    )
    # A aparição usada para atribuir o clube basal não é uma observação da janela.
    # Removê-la evita colisão com game_id e match_date do calendário analítico.
    baseline = baseline.drop(
        columns=["game_id", "match_date", "type", "team_captain", "competition_id", "league_name"],
        errors="ignore",
    )

    grid = baseline.merge(schedule, left_on="baseline_club_id", right_on="club_id", how="inner", validate="many_to_many")
    grid = grid.merge(
        lineups.rename(columns={"player_id": "transfermarkt_player_id", "position": "lineup_position"})[["game_id", "transfermarkt_player_id", "club_id", "type", "lineup_position", "team_captain"]],
        on=["game_id", "transfermarkt_player_id", "club_id"], how="left",
    )
    grid = grid.merge(
        appearances.rename(columns={"player_id": "transfermarkt_player_id", "player_club_id": "club_id"}),
        on=["game_id", "transfermarkt_player_id", "club_id"], how="left",
    )
    for column in ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"]:
        grid[column] = pd.to_numeric(grid[column], errors="coerce").fillna(0.0)
    grid["in_match_squad"] = grid["type"].notna().astype("int8")
    grid["started"] = grid["type"].eq("starting_lineup").astype("int8")
    grid["played"] = grid["minutes_played"].gt(0).astype("int8")
    grid["possible_minutes"] = 90.0
    grid["minutes_share"] = grid["minutes_played"] / grid["possible_minutes"]

    end_dates = transfer_end_dates(set(baseline["transfermarkt_player_id"]), cutoff, baseline)
    grid = grid.merge(end_dates, on="transfermarkt_player_id", how="left")
    grid["active_at_baseline_club"] = grid["spell_end"].isna() | grid["match_date"].lt(grid["spell_end"])
    after_exit = ~grid["active_at_baseline_club"]
    grid.loc[after_exit, ["in_match_squad", "started", "played"]] = 0
    grid.loc[after_exit, ["minutes_played", "goals", "assists", "yellow_cards", "red_cards", "minutes_share"]] = 0.0

    pre_strength = full_schedule[full_schedule["match_date"].le(cutoff)].copy()
    pre_strength["points"] = np.select(
        [pre_strength["club_goals"].gt(pre_strength["opponent_goals"]), pre_strength["club_goals"].eq(pre_strength["opponent_goals"])],
        [3, 1], default=0,
    )
    club_strength = pre_strength.groupby("club_id", as_index=False).agg(pre_club_ppg=("points", "mean"))
    grid = grid.merge(club_strength, on="club_id", how="left")
    opponent_strength = club_strength.rename(columns={"club_id": "opponent_club_id", "pre_club_ppg": "pre_opponent_ppg"})
    grid = grid.merge(opponent_strength, on="opponent_club_id", how="left")

    values = latest_valuation(set(baseline["transfermarkt_player_id"]), cutoff)
    grid = grid.merge(values, on="transfermarkt_player_id", how="left")
    grid["market_value_in_eur"] = pd.to_numeric(grid["market_value_in_eur"], errors="coerce")
    grid["market_value_log"] = np.log1p(grid["market_value_in_eur"])
    grid["age_at_pseudo_cutoff"] = (cutoff - pd.to_datetime(grid["date_of_birth"], errors="coerce")).dt.days / 365.2425
    return grid, schedule, baseline, full_schedule


def build_risk_and_weights(grid: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre = grid[grid["relative_club_match"].between(-8, -1)].copy()
    summaries = pre.groupby("transfermarkt_player_id", as_index=False).agg(
        player_name=("player_name", "first"), baseline_club_id=("baseline_club_id", "first"), baseline_club_name=("baseline_club_name", "first"),
        league_name=("league_name", "first"), position_group=("position_group", "first"), treated_world_cup=("treated_world_cup", "first"),
        national_team_eligible=("national_team_eligible", "first"), qualified_national_team=("qualified_national_team", "first"), is_goalkeeper=("is_goalkeeper", "first"),
        age_at_pseudo_cutoff=("age_at_pseudo_cutoff", "first"), market_value_log=("market_value_log", "first"),
        pre_club_ppg=("pre_club_ppg", "first"), pre_minutes_share=("minutes_share", "mean"), pre_squad_rate=("in_match_squad", "mean"),
        pre_start_rate=("started", "mean"), pre_minutes_played=("minutes_played", "sum"), pre_opponent_ppg=("pre_opponent_ppg", "mean"),
    )
    trends = pre.groupby("transfermarkt_player_id").apply(lambda group: slope(group["minutes_share"], group["relative_club_match"]), include_groups=False).rename("pre_trend_minutes_share").reset_index()
    summaries = summaries.merge(trends, on="transfermarkt_player_id", how="left")
    history = pre.pivot_table(index="transfermarkt_player_id", columns="relative_club_match", values="minutes_share", aggfunc="first")
    history = history.rename(columns={event: f"history_minutes_share_m{abs(int(event))}" for event in history.columns}).reset_index()
    summaries = summaries.merge(history, on="transfermarkt_player_id", how="left")
    summaries["outfield"] = ~summaries["is_goalkeeper"].fillna(True)
    summaries["eligible_baseline_only"] = summaries["outfield"] & summaries["pre_minutes_played"].ge(180)
    summaries["eligible_qualified_risk_set"] = summaries["eligible_baseline_only"] & summaries["qualified_national_team"].eq(True)
    risk = summaries[summaries["eligible_qualified_risk_set"]].copy()

    stratum_counts = risk.groupby(["league_name", "position_group"])["treated_world_cup"].agg(["sum", "count"])
    viable = stratum_counts[(stratum_counts["sum"] > 0) & (stratum_counts["sum"] < stratum_counts["count"])].index
    risk = risk.set_index(["league_name", "position_group"]).loc[viable].reset_index()
    nation_counts = risk.groupby("national_team_eligible")["treated_world_cup"].agg(["sum", "count"])
    risk["eligible_same_nation_support"] = risk["national_team_eligible"].map(nation_counts["sum"]).gt(0) & risk["national_team_eligible"].map(nation_counts["sum"]).lt(risk["national_team_eligible"].map(nation_counts["count"]))
    risk = weighting.add_strata(risk, config)
    history_columns = [column for column in risk.columns if column.startswith("history_minutes_share_")]
    risk, missing_columns = weighting.impute_covariates(risk, config, history_columns)
    weights, diagnostics, balance, _, _ = weighting.select_overlap_weights(risk, config, missing_columns)
    return summaries, risk, weights, pd.concat([diagnostics.assign(table="candidate"), balance.assign(table="balance")], ignore_index=True, sort=False)


def estimate(grid: pd.DataFrame, weights: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimation = grid.merge(weights[["transfermarkt_player_id", "analysis_weight"]], on="transfermarkt_player_id", how="inner", validate="many_to_one")
    estimation = estimation[estimation["relative_club_match"].isin(list(range(-8, 0)) + list(range(1, 9)))].copy()
    estimation["club_game_id"] = estimation["club_id"].astype(str) + "|" + estimation["game_id"].astype(str)
    treatment = estimation["treated_world_cup"].to_numpy(float)
    acute = estimation["relative_club_match"].between(1, 4).to_numpy(float)
    consolidated = estimation["relative_club_match"].between(5, 8).to_numpy(float)
    scale, baseline = weighted_summary_scale(estimation, "minutes_share")
    alpha = float(config["analysis"]["alpha"])
    rows: list[dict[str, object]] = []
    primary = fit_hdfe(estimation, "minutes_share", {"treated_x_acute": treatment * acute, "treated_x_consolidated": treatment * consolidated}, ["transfermarkt_player_id", "club_game_id"])
    rows.extend(model_rows(primary, "minutes_share", "placebo_primary_player_club_game_fe", scale, baseline, alpha))
    trend = fit_hdfe(estimation, "minutes_share", {"treated_x_acute": treatment * acute, "treated_x_consolidated": treatment * consolidated, "treated_x_linear_time": treatment * estimation["relative_club_match"].to_numpy(float)}, ["transfermarkt_player_id", "club_game_id"])
    rows.extend(model_rows(trend, "minutes_share", "placebo_differential_linear_trend", scale, baseline, alpha))
    unweighted = estimation.copy()
    unweighted["analysis_weight"] = 1.0
    unweighted_fit = fit_hdfe(unweighted, "minutes_share", {"treated_x_acute": treatment * acute, "treated_x_consolidated": treatment * consolidated}, ["transfermarkt_player_id", "club_game_id"])
    rows.extend(model_rows(unweighted_fit, "minutes_share", "placebo_unweighted_player_club_game_fe", scale, baseline, alpha))
    return pd.DataFrame(rows), estimation


def sample_flow(summaries: pd.DataFrame, risk: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    steps = [
        ("players_with_pre_pause_lineup", summaries),
        ("outfield_with_180_pre_minutes", summaries[summaries["eligible_baseline_only"]]),
        ("qualified_national_team_risk_set", summaries[summaries["eligible_qualified_risk_set"]]),
        ("analysis_strata_with_treated_and_control", risk),
        ("weighted_placebo_population", weights),
    ]
    rows = []
    for name, frame in steps:
        rows.append({"step": name, "total": len(frame), "future_qatar_2022_selected": int(frame["treated_world_cup"].eq(1).sum()), "controls": int(frame["treated_world_cup"].eq(0).sum())})
    return pd.DataFrame(rows)


def quality_table(grid: pd.DataFrame, schedule: pd.DataFrame, risk: pd.DataFrame, weights: pd.DataFrame, results: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    selected = results[results["specification"].eq("placebo_primary_player_club_game_fe")]
    candidate = pd.read_csv(REPORTS / "weighting_diagnostics.csv")
    selected_candidate = candidate[(candidate["table"].eq("candidate")) & candidate["selected"].eq(True)].iloc[0]
    rows = [
        ("all_big5_clubs_have_16_window_games", schedule.groupby("club_id")["game_id"].nunique().eq(16).all(), True),
        ("window_uses_2023_november_pause", pd.Timestamp(config["baseline_cutoff"]).date() == pd.Timestamp("2023-11-12").date() and pd.Timestamp(config["post_start"]).date() == pd.Timestamp("2023-11-24").date(), True),
        ("unique_player_club_game", not grid.duplicated(["transfermarkt_player_id", "club_id", "game_id"]).any(), True),
        ("minutes_share_in_unit_interval", grid["minutes_share"].between(0, 1).all(), True),
        ("both_treatment_groups_in_risk_set", risk["treated_world_cup"].nunique() == 2, True),
        ("weighting_candidate_selected", len(candidate[(candidate["table"].eq("candidate")) & candidate["selected"].eq(True)]) == 1, True),
        ("primary_models_converged", selected["fe_max_error"].lt(1e-8).all(), True),
        ("placebo_results_are_not_world_cup_effects", True, True),
        ("weighting_selection_gates_passed", bool(selected_candidate["passes_selection_gates"]), True),
        ("results_reportable_as_diagnostic_only", not bool(selected_candidate["passes_selection_gates"]), True),
    ]
    return pd.DataFrame(rows, columns=["check", "observed", "expected"])


def write_manifest() -> None:
    records: list[dict[str, object]] = []
    for path in sorted(REPORTS.glob("*.csv")):
        if path.name == "manifest.csv":
            continue
        with path.open("rb") as stream:
            rows = max(sum(1 for _ in stream) - 1, 0)
        records.append({"relative_path": path.name, "rows": rows, "bytes": path.stat().st_size, "sha256": sha256(path), "generated_by": "scripts/analysis/run_placebo_2023_24.py"})
    for path in [CONFIG_PATH, *[RAW / name for name in SOURCE_FILES]]:
        source_url = SOURCE_URL + path.name if path.parent == RAW else ""
        records.append({"relative_path": str(path.relative_to(ROOT)), "rows": pd.NA, "bytes": path.stat().st_size, "sha256": sha256(path), "generated_by": "input_or_config", "source_url": source_url})
    pd.DataFrame(records).to_csv(REPORTS / "manifest.csv", index=False, encoding="utf-8", lineterminator="\n")


def write_readme(flow: pd.DataFrame, diagnostics: pd.DataFrame, results: pd.DataFrame) -> None:
    selected = diagnostics[(diagnostics["table"].eq("candidate")) & diagnostics["selected"].eq(True)].iloc[0]
    balance = diagnostics[diagnostics["table"].eq("balance")]
    continuous = balance[balance["variable_type"].isin(["continuous", "prehistory"])]
    primary = results[(results["specification"].eq("placebo_primary_player_club_game_fe"))].set_index("coefficient")
    acute = primary.loc["treated_x_acute"]
    consolidated = primary.loc["treated_x_consolidated"]
    text = f"""# Placebo sazonal 2023/24 — pausa FIFA de novembro

Este relatório usa partidas **reais** de 2023/24. O marcador `treated_world_cup`
indica quem integrou a lista oficial de Qatar 2022; como a Copa já havia
terminado, os coeficientes são uma falsificação de trajetória, não efeitos da Copa.

- Corte pré-pausa: 12/11/2023.
- Retorno da janela internacional: 24/11/2023.
- Janelas: oito jogos antes, quatro agudos depois e quatro consolidados depois.
- Reconstrução dos dados brutos: `python scripts/data/download_transfermarkt_placebo_2023_24.py`.
- Reconstrução da análise: `python scripts/analysis/run_placebo_2023_24.py`.
- População ponderada: {int(flow.iloc[-1]['total'])} jogadores ({int(flow.iloc[-1]['future_qatar_2022_selected'])} futuros convocados e {int(flow.iloc[-1]['controls'])} controles).
- Ridge selecionado: {float(selected['ridge']):.3g}; ESS tratado/controle: {float(selected['treated_ess']):.2f}/{float(selected['control_ess']):.2f}.
- Após calibração, máximo |SMD|: {float(continuous['smd_after'].abs().max()):.4g}; máximo KS: {float(continuous['ks_after'].max()):.4g}; razão de variância: {float(continuous['variance_ratio_after'].min()):.3f}–{float(continuous['variance_ratio_after'].max()):.3f}.
- Gates de ponderação: {'aprovados' if bool(selected['passes_selection_gates']) else 'não aprovados; resultados são somente diagnósticos'}. O ESS dos controles foi {float(selected['control_ess_fraction']):.1%} (gate: 50%) e o KS máximo foi {float(continuous['ks_after'].max()):.4g} (gate: 0,10).
- Efeito placebo agudo em `minutes_share`: {float(acute['estimate']):.4f} (IC95% {float(acute['ci_lower_player']):.4f} a {float(acute['ci_upper_player']):.4f}; p={float(acute['p_value_player']):.4f}).
- Efeito placebo consolidado: {float(consolidated['estimate']):.4f} (IC95% {float(consolidated['ci_lower_player']):.4f} a {float(consolidated['ci_upper_player']):.4f}; p={float(consolidated['p_value_player']):.4f}).

## Limites

Esta falsificação cobre `minutes_share`, não npxG+xA, lesões ou dados
físicos: os snapshots locais de Understat e lesões para 2023/24 não estão
incluídos no pipeline. Os pesos foram recalculados com covariáveis pré-pausa disponíveis
(idade, valor, utilização, titularidade, força de clube/adversário e trajetória
de minutos), sem reutilizar pesos de 2022/23. A elegibilidade nacional prioriza
a tabela auditada de 2022 quando o atleta está presente nela; para demais atletas,
usa cidadania primária do Transfermarkt e deve ser lida como proxy.

Não há holdout pré-pausa homogêneo nesta temporada: em 12/11/2023, a
Bundesliga tinha somente 11 rodadas concluídas. Assim, todos os oito jogos
pré-pausa disponíveis para a especificação foram usados na calibração; o
placebo não acrescenta um teste independente de pré-tendência. Além disso, a
coorte observável em 2023/24 não é idêntica à de 2022/23 (314 convocados), e o
snapshot público baixado pode diferir do snapshot histórico usado na análise
principal.
"""
    (REPORTS / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    assert_sources()
    config = load_config()
    grid, schedule, _, _ = prepare_panel(config)
    summaries, risk, weights, diagnostics = build_risk_and_weights(grid, config)
    write_csv(diagnostics, "weighting_diagnostics.csv")
    results, estimation = estimate(grid, weights, config)
    flow = sample_flow(summaries, risk, weights)
    write_csv(flow, "sample_flow.csv")
    write_csv(weights, "analysis_weights.csv")
    write_csv(results, "placebo_coefficients.csv")
    qa = quality_table(grid, schedule, risk, weights, results, config)
    write_csv(qa, "qa.csv")
    write_readme(flow, diagnostics, results)
    write_manifest()
    print("Placebo 2023/24 concluído. Consulte reports/placebo_2023_24/README.md.")


if __name__ == "__main__":
    main()
