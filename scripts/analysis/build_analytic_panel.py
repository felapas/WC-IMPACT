"""Fase 3: painel analítico, amostras e matching para o estudo FAME 2026.

O script lê apenas data/raw e data/interim e grava saídas reproduzíveis em
data/processed. Nenhuma elegibilidade usa informação de desempenho pós-Copa.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "python"
if TOOLS.exists():
    sys.path.insert(0, str(TOOLS))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

PRE_END = pd.Timestamp("2022-11-13")
POST_START = pd.Timestamp("2022-12-19")
INTERVENTION = pd.Timestamp("2022-11-20")
SEASON_START = pd.Timestamp("2022-08-01")
SEASON_END = pd.Timestamp("2023-06-30")
JAN_START = pd.Timestamp("2023-01-01")
JAN_END = pd.Timestamp("2023-01-31")
LONG_INJURY_DAYS = 28
BASELINE_MINUTES = 180
MIN_GAMES_EACH_SIDE = 4
MATCH_RATIO = 3
MATCH_CALIPER = 2.0

BIG5 = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}


def open_transfermarkt() -> duckdb.DuckDBPyConnection:
    path = RAW / "transfermarkt" / "dcaribou" / "transfermarkt-datasets.duckdb"
    return duckdb.connect(str(path), read_only=True)


def read_inputs() -> dict[str, pd.DataFrame]:
    names = [
        "big5_player_match_2022_23.csv",
        "fifa_player_crosswalk.csv",
        "wc_player_match_2022.csv",
        "wc_player_tournament_2022.csv",
        "understat_player_match_2022_23.csv",
        "understat_transfermarkt_game_crosswalk.csv",
        "understat_transfermarkt_player_crosswalk.csv",
    ]
    data = {name.removesuffix(".csv"): pd.read_csv(INTERIM / name, low_memory=False) for name in names}
    panel = data["big5_player_match_2022_23"]
    panel["match_date"] = pd.to_datetime(panel["match_date"])
    panel["date_of_birth"] = pd.to_datetime(panel["date_of_birth"], errors="coerce")
    return data


def load_reference_tables(conn: duckdb.DuckDBPyConnection, player_ids: set[int], club_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    players_sql = ",".join(map(str, sorted(player_ids))) or "-1"
    clubs_sql = ",".join(map(str, sorted(club_ids))) or "-1"
    transfers = conn.sql(
        f"""
        SELECT player_id, transfer_date, from_club_id, to_club_id,
               from_club_name, to_club_name, transfer_fee, market_value_in_eur
        FROM transfers
        WHERE player_id IN ({players_sql})
          AND transfer_date BETWEEN DATE '2022-07-01' AND DATE '2023-06-30'
          AND (from_club_id IN ({clubs_sql}) OR to_club_id IN ({clubs_sql}))
        """
    ).df()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])

    valuations = conn.sql(
        f"""
        WITH ranked AS (
            SELECT player_id, date valuation_date, market_value_in_eur,
                   row_number() OVER (PARTITION BY player_id ORDER BY date DESC) rn
            FROM player_valuations
            WHERE player_id IN ({players_sql}) AND date <= DATE '2022-11-14'
        )
        SELECT player_id, valuation_date, market_value_in_eur
        FROM ranked WHERE rn=1
        """
    ).df()
    valuations["valuation_date"] = pd.to_datetime(valuations["valuation_date"])
    return transfers, valuations


def build_club_schedule(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "game_id", "match_date", "competition_id", "league_name", "club_id", "club_name",
        "opponent_club_id", "opponent_club_name", "is_home", "club_goals", "opponent_goals",
        "round", "club_table_position", "opponent_table_position", "club_formation",
        "days_since_club_game",
    ]
    full = panel[columns].drop_duplicates(["game_id", "club_id"]).sort_values(["club_id", "match_date", "game_id"])
    if full.duplicated(["game_id", "club_id"]).any():
        raise RuntimeError("Duplicatas de clube-jogo na agenda")

    selected: list[pd.DataFrame] = []
    for club_id, games in full.groupby("club_id"):
        pre = games[games["match_date"] <= PRE_END].tail(8).copy()
        post = games[games["match_date"] >= POST_START].head(8).copy()
        if len(pre) != 8 or len(post) != 8:
            raise RuntimeError(f"Clube {club_id} sem oito jogos em cada janela: pre={len(pre)}, post={len(post)}")
        pre["relative_club_match"] = np.arange(-len(pre), 0)
        post["relative_club_match"] = np.arange(1, len(post) + 1)
        selected.extend([pre, post])
    window = pd.concat(selected, ignore_index=True)
    window["window"] = np.select(
        [
            window["relative_club_match"] < 0,
            window["relative_club_match"].between(1, 4),
            window["relative_club_match"].between(5, 8),
        ],
        ["pre", "post_acute", "post_consolidated"],
        default="invalid",
    )
    return full.reset_index(drop=True), window.sort_values(["club_id", "match_date"]).reset_index(drop=True)


def build_roster_spells(panel: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        "transfermarkt_player_id", "player_name", "club_id", "club_name", "competition_id",
        "league_name", "date_of_birth", "broad_position", "sub_position",
        "country_of_citizenship", "foot", "height_in_cm", "is_goalkeeper",
    ]
    base = panel.sort_values("match_date").groupby(["transfermarkt_player_id", "club_id"], as_index=False).agg(
        player_name=("player_name", "last"),
        club_name=("club_name", "last"),
        competition_id=("competition_id", "last"),
        league_name=("league_name", "last"),
        date_of_birth=("date_of_birth", "last"),
        broad_position=("broad_position", "last"),
        sub_position=("sub_position", "last"),
        country_of_citizenship=("country_of_citizenship", "last"),
        foot=("foot", "last"),
        height_in_cm=("height_in_cm", "last"),
        is_goalkeeper=("is_goalkeeper", "max"),
        first_observed=("match_date", "min"),
        last_observed=("match_date", "max"),
    )
    transfer_by_player = {pid: group for pid, group in transfers.groupby("player_id")}
    records: list[dict[str, object]] = []
    big5_club_ids = set(base["club_id"].astype(int))
    for row in base.itertuples(index=False):
        player_transfers = transfer_by_player.get(row.transfermarkt_player_id, pd.DataFrame())
        inbound = player_transfers[
            player_transfers.get("to_club_id", pd.Series(dtype=float)).eq(row.club_id)
            & player_transfers["transfer_date"].between(SEASON_START, SEASON_END)
        ] if not player_transfers.empty else pd.DataFrame()
        outbound = player_transfers[
            player_transfers.get("from_club_id", pd.Series(dtype=float)).eq(row.club_id)
            & player_transfers["transfer_date"].between(SEASON_START, SEASON_END)
        ] if not player_transfers.empty else pd.DataFrame()
        spell_start = SEASON_START
        spell_end = SEASON_END
        start_source = "season_boundary"
        end_source = "season_boundary"
        if not inbound.empty:
            candidates = inbound[inbound["transfer_date"] <= row.first_observed + pd.Timedelta(days=14)]
            if not candidates.empty:
                spell_start = max(SEASON_START, candidates["transfer_date"].max())
                start_source = "transfer"
        if not outbound.empty:
            candidates = outbound[outbound["transfer_date"] >= row.last_observed - pd.Timedelta(days=14)]
            if not candidates.empty:
                spell_end = min(SEASON_END, candidates["transfer_date"].min() - pd.Timedelta(days=1))
                end_source = "transfer"
        jan = False
        if not player_transfers.empty:
            jan_rows = player_transfers[player_transfers["transfer_date"].between(JAN_START, JAN_END)]
            jan = any(
                int(value) in big5_club_ids
                for value in pd.concat([jan_rows["from_club_id"], jan_rows["to_club_id"]]).dropna()
            )
        record = row._asdict()
        record.update(
            {
                "spell_start": spell_start,
                "spell_end": spell_end,
                "start_source": start_source,
                "end_source": end_source,
                "january_transfer": int(jan),
                "spell_valid": int(spell_start <= spell_end),
            }
        )
        records.append(record)
    spells = pd.DataFrame(records)
    return spells.sort_values(["transfermarkt_player_id", "spell_start", "club_id"]).reset_index(drop=True)


def complete_player_game_grid(spells: pd.DataFrame, schedule: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    grid = spells.merge(schedule, on=["club_id", "club_name", "competition_id", "league_name"], how="inner")
    grid = grid[grid["match_date"].between(grid["spell_start"], grid["spell_end"])].copy()
    lineup_columns = [
        "game_id", "transfermarkt_player_id", "club_id", "lineup_position", "started", "team_captain",
        "minutes_played", "goals", "assists", "yellow_cards", "red_cards", "played",
    ]
    lineup = panel[lineup_columns].drop_duplicates(["game_id", "transfermarkt_player_id", "club_id"])
    grid = grid.merge(
        lineup,
        on=["game_id", "transfermarkt_player_id", "club_id"],
        how="left",
        indicator="lineup_merge",
    )
    grid["in_match_squad"] = grid["lineup_merge"].eq("both").astype("int8")
    grid = grid.drop(columns="lineup_merge")

    # Alguns jogadores trocaram de clube sem um registro de transferência completo na
    # fonte. Nesses casos, dois vínculos podem cobrir o mesmo jogo. Mantemos uma única
    # atribuição, priorizando a escalação observada e depois o intervalo efetivamente
    # observado do jogador no clube; os casos ficam explicitamente marcados para auditoria.
    assignment_key = ["transfermarkt_player_id", "game_id"]
    conflict = grid.duplicated(assignment_key, keep=False)
    grid["roster_assignment_conflict"] = conflict.astype("int8")
    grid["observed_club_membership"] = grid["match_date"].between(
        grid["first_observed"], grid["last_observed"]
    ).astype("int8")
    before_observed = (grid["first_observed"] - grid["match_date"]).dt.days.clip(lower=0)
    after_observed = (grid["match_date"] - grid["last_observed"]).dt.days.clip(lower=0)
    grid["_observed_interval_distance"] = before_observed + after_observed
    grid = grid.sort_values(
        assignment_key
        + ["in_match_squad", "observed_club_membership", "_observed_interval_distance", "first_observed", "club_id"],
        ascending=[True, True, False, False, True, False, True],
    ).drop_duplicates(assignment_key, keep="first")
    grid["roster_assignment_rule"] = np.where(
        grid["roster_assignment_conflict"].eq(1),
        "lineup_then_observed_club_interval",
        "single_active_spell",
    )
    grid = grid.drop(columns="_observed_interval_distance")
    zero_columns = ["started", "team_captain", "minutes_played", "goals", "assists", "yellow_cards", "red_cards", "played"]
    grid[zero_columns] = grid[zero_columns].fillna(0)
    grid["lineup_source_inconsistency"] = (
        grid["started"].eq(1) & (grid["played"].ne(1) | grid["minutes_played"].le(0))
    ).astype("int8")
    grid["lineup_source_inconsistency_reason"] = np.where(
        grid["lineup_source_inconsistency"].eq(1),
        "starter_without_played_or_minutes_in_source",
        "",
    )
    grid["possible_minutes"] = 90
    grid["minutes_share"] = (grid["minutes_played"] / grid["possible_minutes"]).clip(0, 1)
    grid["active_roster"] = 1
    grid["squad_status"] = np.select(
        [grid["started"].eq(1), grid["played"].eq(1), grid["in_match_squad"].eq(1)],
        ["starter", "sub_used", "sub_unused"],
        default="not_in_squad",
    )

    appearance_dates = {
        int(pid): np.sort(group.loc[group["played"].eq(1), "match_date"].drop_duplicates().values.astype("datetime64[D]"))
        for pid, group in panel.groupby("transfermarkt_player_id")
    }
    days_since: list[float] = []
    for row in grid.itertuples(index=False):
        dates = appearance_dates.get(int(row.transfermarkt_player_id), np.array([], dtype="datetime64[D]"))
        current = np.datetime64(pd.Timestamp(row.match_date).date())
        index = np.searchsorted(dates, current, side="left") - 1
        days_since.append(float((current - dates[index]).astype(int)) if index >= 0 else np.nan)
    grid["days_since_player_appearance"] = days_since
    return grid.sort_values(["match_date", "game_id", "club_id", "transfermarkt_player_id"]).reset_index(drop=True)


def add_injuries(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    injuries = pd.read_csv(RAW / "transfermarkt" / "salimt" / "player_injuries.csv")
    injuries["from_date"] = pd.to_datetime(injuries["from_date"], errors="coerce")
    injuries["end_date"] = pd.to_datetime(injuries["end_date"], errors="coerce")
    injuries["days_missed"] = pd.to_numeric(injuries["days_missed"], errors="coerce")
    invalid = injuries["from_date"].isna() | injuries["end_date"].isna() | (injuries["end_date"] < injuries["from_date"])
    relevant = injuries[
        ~invalid
        & injuries["from_date"].le(SEASON_END)
        & injuries["end_date"].ge(SEASON_START)
    ].copy()
    by_player = {int(pid): group for pid, group in relevant.groupby("player_id")}
    injured_flags: list[int] = []
    injury_reason: list[str] = []
    for row in grid.itertuples(index=False):
        events = by_player.get(int(row.transfermarkt_player_id), pd.DataFrame())
        if events.empty:
            injured_flags.append(0)
            injury_reason.append("")
            continue
        active = events[(events["from_date"] <= row.match_date) & (events["end_date"] >= row.match_date)]
        injured_flags.append(int(not active.empty))
        injury_reason.append(" | ".join(sorted(active["injury_reason"].dropna().astype(str).unique())))
    grid["injured_on_match_date"] = injured_flags
    grid["injury_reason_on_match_date"] = injury_reason

    pre_start = PRE_END - pd.Timedelta(days=179)
    summaries: list[dict[str, object]] = []
    for pid in grid["transfermarkt_player_id"].drop_duplicates().astype(int):
        events = by_player.get(pid, pd.DataFrame())
        if events.empty:
            summaries.append({"transfermarkt_player_id": pid, "pre_injury_events": 0, "pre_injury_days": 0, "long_injury_crossing": 0})
            continue
        pre = events[(events["from_date"] <= PRE_END) & (events["end_date"] >= pre_start)]
        clipped_days = sum(
            max(0, (min(end, PRE_END) - max(start, pre_start)).days + 1)
            for start, end in zip(pre["from_date"], pre["end_date"])
        )
        crossing = events[
            events["days_missed"].ge(LONG_INJURY_DAYS)
            & events["from_date"].le(JAN_END)
            & events["end_date"].ge(pd.Timestamp("2022-11-14"))
        ]
        summaries.append(
            {
                "transfermarkt_player_id": pid,
                "pre_injury_events": len(pre),
                "pre_injury_days": clipped_days,
                "long_injury_crossing": int(not crossing.empty),
            }
        )
    audit = pd.DataFrame(
        [
            {"injury_check": "source_rows", "value": len(injuries)},
            {"injury_check": "invalid_date_rows", "value": int(invalid.sum())},
            {"injury_check": "relevant_date_rows", "value": len(relevant)},
            {"injury_check": "season_label_22_23_outside_dates", "value": int((injuries["season_name"].eq("22/23") & ~injuries.index.isin(relevant.index)).sum())},
        ]
    )
    return grid, pd.DataFrame(summaries), audit


def add_understat(grid: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pm = inputs["understat_player_match_2022_23"].copy()
    game_map = inputs["understat_transfermarkt_game_crosswalk"]
    player_map = inputs["understat_transfermarkt_player_crosswalk"]
    accepted = player_map[player_map["link_status"].eq("accepted")].copy()
    accepted["understat_player_id"] = pd.to_numeric(accepted["understat_player_id"], errors="coerce")
    pm["understat_player_id"] = pd.to_numeric(pm["understat_player_id"], errors="coerce")
    pm = pm.merge(
        game_map[["competition_id", "understat_match_id", "transfermarkt_game_id"]],
        on=["competition_id", "understat_match_id"], how="left",
    ).merge(
        accepted[["competition_id", "understat_player_id", "transfermarkt_player_id"]],
        on=["competition_id", "understat_player_id"], how="inner",
    )
    metric_columns = ["shots", "goals", "shots_on_target", "penalties", "xg", "npxg", "xa", "key_passes"]
    metrics = pm.groupby(["transfermarkt_game_id", "transfermarkt_player_id"], as_index=False)[metric_columns].sum()
    metrics = metrics.rename(columns={"transfermarkt_game_id": "game_id", "goals": "understat_goals"})
    linked_ids = set(accepted["transfermarkt_player_id"].dropna().astype(int))
    grid = grid.merge(metrics, on=["game_id", "transfermarkt_player_id"], how="left")
    grid["understat_linked"] = grid["transfermarkt_player_id"].isin(linked_ids).astype("int8")
    renamed_metrics = ["shots", "understat_goals", "shots_on_target", "penalties", "xg", "npxg", "xa", "key_passes"]
    for column in renamed_metrics:
        grid.loc[grid["understat_linked"].eq(1), column] = grid.loc[grid["understat_linked"].eq(1), column].fillna(0)
    offensive_columns = ["shots", "understat_goals", "xg", "npxg", "xa", "key_passes"]
    grid["performance_source_inconsistency"] = (
        grid["played"].eq(0) & grid[offensive_columns].fillna(0).ne(0).any(axis=1)
    ).astype("int8")
    grid["npxg_xa"] = grid["npxg"] + grid["xa"]
    for column in ["shots", "xg", "npxg", "xa", "key_passes", "npxg_xa"]:
        grid[f"{column}_p90"] = np.where(grid["minutes_played"] > 0, grid[column] * 90 / grid["minutes_played"], np.nan)
    return grid


def add_treatment_and_covariates(
    grid: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    valuations: pd.DataFrame,
    injury_summary: pd.DataFrame,
    transfers: pd.DataFrame,
    full_schedule: pd.DataFrame,
) -> pd.DataFrame:
    fifa = inputs["fifa_player_crosswalk"].copy()
    fifa = fifa[fifa["big5_at_world_cup"].eq(True) & fifa["tm_link_status"].eq("accepted")]
    fifa["transfermarkt_player_id"] = pd.to_numeric(fifa["transfermarkt_player_id"], errors="coerce")
    # Um candidato de linkage em revisão não é uma identidade operacional.
    # Mantemos o atleta como tratado pela lista oficial, mas sem anexar métricas
    # StatsBomb de outro jogador.
    fifa.loc[~fifa["sb_link_status"].eq("accepted"), "statsbomb_player_id"] = pd.NA
    treatment_columns = [
        "transfermarkt_player_id", "fifa_squad_id", "country_name", "statsbomb_player_id",
        "sb_link_status", "club_at_world_cup",
    ]
    treatment = fifa[treatment_columns].drop_duplicates("transfermarkt_player_id")
    treatment["treated_world_cup"] = 1
    grid = grid.merge(treatment, on="transfermarkt_player_id", how="left")
    grid["treated_world_cup"] = grid["treated_world_cup"].fillna(0).astype("int8")

    wc = inputs["wc_player_tournament_2022"].copy()
    # O StatsBomb alterou a grafia de quatro nomes durante o torneio, gerando mais
    # de uma linha para o mesmo ID. Consolida-se por ID antes da junção para manter
    # uma linha por jogador e preservar a dose total da Copa.
    wc_sum_columns = [
        "started", "played", "minutes_played", "extra_time_game",
        "extra_time_minutes", "match_duration_minutes", "actions",
        "under_pressure_actions", "shots", "goals", "xg", "npxg", "passes",
        "completed_passes", "xa", "progressive_passes_proxy", "carries",
        "progressive_carries_proxy", "pressures", "ball_recoveries", "dribbles",
        "completed_dribbles", "duels", "duels_won", "squad_matchdays",
    ]
    wc_aggregation = {column: "sum" for column in wc_sum_columns}
    wc_aggregation.update({"player_name_statsbomb": "last", "team_name": "last", "max_stage_order": "max"})
    wc = wc.groupby("statsbomb_player_id", as_index=False).agg(wc_aggregation)
    for column in ["xg", "npxg", "xa", "shots", "passes", "progressive_passes_proxy", "progressive_carries_proxy", "pressures"]:
        wc[f"{column}_p90"] = np.where(
            wc["minutes_played"].gt(0), wc[column] * 90 / wc["minutes_played"], np.nan
        )
    wc = wc.add_prefix("wc_")
    grid = grid.merge(wc, left_on="statsbomb_player_id", right_on="wc_statsbomb_player_id", how="left")
    dose_columns = [
        "wc_minutes_played", "wc_played", "wc_started", "wc_extra_time_game",
        "wc_extra_time_minutes", "wc_match_duration_minutes", "wc_max_stage_order",
        "wc_xg_p90", "wc_npxg_p90", "wc_xa_p90", "wc_progressive_passes_proxy_p90",
        "wc_progressive_carries_proxy_p90", "wc_pressures_p90",
    ]
    for column in dose_columns:
        if column in grid:
            grid.loc[grid["treated_world_cup"].eq(1), column] = grid.loc[grid["treated_world_cup"].eq(1), column].fillna(0)

    grid = grid.merge(valuations, left_on="transfermarkt_player_id", right_on="player_id", how="left").drop(columns="player_id")
    grid["market_value_log"] = np.log1p(grid["market_value_in_eur"])
    grid = grid.merge(injury_summary, on="transfermarkt_player_id", how="left")
    for column in ["pre_injury_events", "pre_injury_days", "long_injury_crossing"]:
        grid[column] = grid[column].fillna(0)

    jan_ids = set(
        transfers.loc[transfers["transfer_date"].between(JAN_START, JAN_END), "player_id"].astype(int)
    )
    grid["january_transfer"] = (
        grid["january_transfer"].fillna(0).astype(bool)
        | grid["transfermarkt_player_id"].isin(jan_ids)
    ).astype("int8")
    grid["age_at_world_cup"] = (INTERVENTION - pd.to_datetime(grid["date_of_birth"])).dt.days / 365.2425

    strength = full_schedule[full_schedule["match_date"] <= PRE_END].copy()
    strength["points"] = np.select(
        [strength["club_goals"] > strength["opponent_goals"], strength["club_goals"] == strength["opponent_goals"]],
        [3, 1], default=0,
    )
    strength["goal_difference"] = strength["club_goals"] - strength["opponent_goals"]
    club_strength = strength.groupby("club_id", as_index=False).agg(
        pre_club_ppg=("points", "mean"),
        pre_club_gd_per_game=("goal_difference", "mean"),
        pre_club_matches=("game_id", "nunique"),
    )
    grid = grid.merge(club_strength, on="club_id", how="left")
    opponent_strength = club_strength.rename(
        columns={
            "club_id": "opponent_club_id",
            "pre_club_ppg": "pre_opponent_ppg",
            "pre_club_gd_per_game": "pre_opponent_gd_per_game",
            "pre_club_matches": "pre_opponent_matches",
        }
    )
    grid = grid.merge(opponent_strength, on="opponent_club_id", how="left")
    return grid


def position_group(frame: pd.DataFrame) -> pd.Series:
    broad = frame["broad_position"].fillna("").str.lower()
    lineup = frame["lineup_position"].fillna("").str.lower()
    return np.select(
        [
            broad.str.contains("goal") | lineup.str.contains("goal"),
            broad.str.contains("defend") | lineup.str.contains("back"),
            broad.str.contains("mid") | lineup.str.contains("mid"),
            broad.str.contains("attack") | lineup.str.contains("wing|forward|striker"),
        ],
        ["Goalkeeper", "Defender", "Midfield", "Attack"],
        default="Unknown",
    )


def add_normalizations(grid: pd.DataFrame) -> pd.DataFrame:
    grid["position_group"] = position_group(grid)
    for metric in ["minutes_share", "npxg_xa_p90"]:
        group = grid.groupby(["competition_id", "position_group"])[metric]
        mean = group.transform("mean")
        std = group.transform("std").replace(0, np.nan)
        median = group.transform("median")
        mad = grid.assign(_abs=(grid[metric] - median).abs()).groupby(["competition_id", "position_group"])["_abs"].transform("median")
        grid[f"{metric}_z"] = (grid[metric] - mean) / std
        grid[f"{metric}_robust_z"] = (grid[metric] - median) / (1.4826 * mad.replace(0, np.nan))
    return grid


def build_window_aggregates(grid: pd.DataFrame) -> pd.DataFrame:
    sum_columns = [
        "in_match_squad", "played", "started", "minutes_played", "possible_minutes",
        "goals", "assists", "shots", "xg", "npxg", "xa", "key_passes", "npxg_xa",
        "injured_on_match_date",
    ]
    records: list[dict[str, object]] = []
    id_columns = ["transfermarkt_player_id", "player_name", "club_id", "club_name", "competition_id", "league_name", "position_group"]
    for keys, group in grid.groupby(id_columns + ["window"], dropna=False):
        row = dict(zip(id_columns + ["window"], keys))
        row["club_games"] = group["game_id"].nunique()
        for column in sum_columns:
            row[column] = group[column].sum(min_count=1)
        row["minutes_share"] = row["minutes_played"] / row["possible_minutes"] if row["possible_minutes"] else np.nan
        row["start_rate"] = row["started"] / row["club_games"] if row["club_games"] else np.nan
        row["squad_rate"] = row["in_match_squad"] / row["club_games"] if row["club_games"] else np.nan
        row["played_rate"] = row["played"] / row["club_games"] if row["club_games"] else np.nan
        for column in ["goals", "assists", "shots", "xg", "npxg", "xa", "key_passes", "npxg_xa"]:
            row[f"{column}_p90"] = row[column] * 90 / row["minutes_played"] if row["minutes_played"] > 0 and pd.notna(row[column]) else np.nan
        records.append(row)
    return pd.DataFrame(records).sort_values(["transfermarkt_player_id", "club_id", "window"]).reset_index(drop=True)


def build_player_summary(grid: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "transfermarkt_player_id", "player_name", "club_id", "club_name", "competition_id", "league_name",
        "position_group", "date_of_birth", "age_at_world_cup", "country_of_citizenship", "foot",
        "height_in_cm", "treated_world_cup", "fifa_squad_id", "country_name", "statsbomb_player_id",
        "market_value_in_eur", "market_value_log", "valuation_date", "pre_injury_events",
        "pre_injury_days", "long_injury_crossing", "january_transfer", "pre_club_ppg",
        "pre_club_gd_per_game",
    ]
    group_keys = ["transfermarkt_player_id", "club_id"]
    existing = [column for column in base_columns if column in grid.columns and column not in group_keys]
    base = grid.sort_values("match_date").groupby(group_keys, as_index=False)[existing].first()
    values = [
        "club_games", "minutes_played", "in_match_squad", "played", "started", "minutes_share",
        "start_rate", "squad_rate", "played_rate", "goals", "assists", "xg", "npxg", "xa",
        "npxg_xa", "npxg_xa_p90", "injured_on_match_date",
    ]
    wide = windows.pivot_table(
        index=["transfermarkt_player_id", "club_id"], columns="window", values=values, aggfunc="first"
    )
    wide.columns = [f"{window}_{metric}" for metric, window in wide.columns]
    wide = wide.reset_index()
    summary = base.merge(wide, on=["transfermarkt_player_id", "club_id"], how="left")

    pre_clubs = grid[grid["relative_club_match"] < 0].groupby("transfermarkt_player_id")["club_id"].nunique()
    post_clubs = grid[grid["relative_club_match"] > 0].groupby("transfermarkt_player_id")["club_id"].nunique()
    summary["same_club_pre_post"] = (
        summary["transfermarkt_player_id"].map(pre_clubs).fillna(0).eq(1)
        & summary["transfermarkt_player_id"].map(post_clubs).fillna(0).eq(1)
    )
    summary["outfield"] = summary["position_group"].ne("Goalkeeper")
    summary["baseline_minutes_ok"] = summary.get("pre_minutes_played", 0).fillna(0).ge(BASELINE_MINUTES)
    summary["coverage_ok"] = (
        summary.get("pre_club_games", 0).fillna(0).ge(MIN_GAMES_EACH_SIDE)
        & (
            summary.get("post_acute_club_games", 0).fillna(0)
            + summary.get("post_consolidated_club_games", 0).fillna(0)
        ).ge(MIN_GAMES_EACH_SIDE)
    )
    summary["eligible_availability"] = (
        summary["outfield"]
        & summary["same_club_pre_post"]
        & summary["baseline_minutes_ok"]
        & summary["coverage_ok"]
        & summary["january_transfer"].fillna(0).eq(0)
    )
    summary["eligible_main"] = summary["eligible_availability"] & summary["long_injury_crossing"].fillna(0).eq(0)

    def reasons(row: pd.Series) -> str:
        result: list[str] = []
        if not row["outfield"]:
            result.append("goalkeeper")
        if not row["same_club_pre_post"]:
            result.append("club_change_or_missing_window")
        if not row["baseline_minutes_ok"]:
            result.append("pre_minutes_below_180")
        if not row["coverage_ok"]:
            result.append("fewer_than_4_games_each_side")
        if row["january_transfer"]:
            result.append("january_transfer")
        if row["long_injury_crossing"]:
            result.append("long_injury_crossing")
        return "|".join(result)

    summary["exclusion_reasons"] = summary.apply(reasons, axis=1)
    return summary.sort_values(["eligible_main", "treated_world_cup", "league_name", "club_name", "player_name"], ascending=[False, False, True, True, True]).reset_index(drop=True)


def perform_matching(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = candidates.copy()
    feature_columns = [
        "age_at_world_cup", "pre_minutes_played", "pre_start_rate", "pre_npxg",
        "pre_xa", "market_value_log", "pre_injury_days",
    ]
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        medians = frame.groupby(["competition_id", "position_group"])[column].transform("median")
        frame[column] = frame[column].fillna(medians).fillna(frame[column].median()).fillna(0)
        means = frame.groupby(["competition_id", "position_group"])[column].transform("mean")
        stds = frame.groupby(["competition_id", "position_group"])[column].transform("std").replace(0, 1).fillna(1)
        frame[f"z_{column}"] = (frame[column] - means) / stds
    z_columns = [f"z_{column}" for column in feature_columns]
    treated = frame[frame["treated_world_cup"].eq(1)].copy()
    controls = frame[frame["treated_world_cup"].eq(0)].copy()
    controls_by_stratum = {
        key: group.copy() for key, group in controls.groupby(["competition_id", "position_group"])
    }
    scarcity = []
    for row in treated.itertuples(index=False):
        pool = controls_by_stratum.get((row.competition_id, row.position_group), pd.DataFrame())
        same = int(pool["club_id"].eq(row.club_id).sum()) if not pool.empty else 0
        scarcity.append((same, len(pool), row.transfermarkt_player_id, row.club_id))
    treated_order = [item[2] for item in sorted(scarcity)]
    used_controls: set[int] = set()
    assignments: list[dict[str, object]] = []
    treated_index = treated.set_index("transfermarkt_player_id", drop=False)
    for treated_id in treated_order:
        row = treated_index.loc[treated_id]
        pool = controls_by_stratum.get((row["competition_id"], row["position_group"]), pd.DataFrame()).copy()
        pool = pool[~pool["transfermarkt_player_id"].isin(used_controls)]
        if pool.empty:
            assignments.append({"treated_player_id": treated_id, "control_player_id": pd.NA, "match_status": "unmatched_no_pool"})
            continue
        vector = row[z_columns].to_numpy(dtype=float)
        matrix = pool[z_columns].to_numpy(dtype=float)
        pool["distance"] = np.sqrt(np.mean((matrix - vector) ** 2, axis=1))
        pool["same_club"] = pool["club_id"].eq(row["club_id"]).astype(int)
        eligible = pool[pool["distance"] <= MATCH_CALIPER].sort_values(["same_club", "distance"], ascending=[False, True]).head(MATCH_RATIO)
        if eligible.empty:
            assignments.append({"treated_player_id": treated_id, "control_player_id": pd.NA, "match_status": "unmatched_caliper"})
            continue
        count = len(eligible)
        for rank, control in enumerate(eligible.itertuples(index=False), start=1):
            used_controls.add(int(control.transfermarkt_player_id))
            assignments.append(
                {
                    "treated_player_id": int(treated_id),
                    "treated_name": row["player_name"],
                    "control_player_id": int(control.transfermarkt_player_id),
                    "control_name": control.player_name,
                    "competition_id": row["competition_id"],
                    "league_name": row["league_name"],
                    "position_group": row["position_group"],
                    "treated_club_id": int(row["club_id"]),
                    "control_club_id": int(control.club_id),
                    "same_club": int(control.club_id == row["club_id"]),
                    "distance": round(float(control.distance), 6),
                    "control_rank": rank,
                    "treated_weight": 1.0,
                    "control_weight": 1.0 / count,
                    "match_status": "matched",
                }
            )
    assignments_df = pd.DataFrame(assignments)

    matched = assignments_df[assignments_df["match_status"].eq("matched")]
    diagnostics: list[dict[str, object]] = []
    if not matched.empty:
        matched_treated_ids = matched["treated_player_id"].unique()
        matched_control_ids = matched["control_player_id"].unique()
        for feature in feature_columns:
            t_all = treated[feature]
            c_all = controls[feature]
            pooled = math.sqrt((t_all.var(ddof=1) + c_all.var(ddof=1)) / 2) or 1
            before = (t_all.mean() - c_all.mean()) / pooled
            t_after = treated[treated["transfermarkt_player_id"].isin(matched_treated_ids)][feature]
            control_values = controls.set_index("transfermarkt_player_id")[feature]
            weights = matched.groupby("control_player_id")["control_weight"].sum()
            c_after_mean = np.average(control_values.loc[weights.index], weights=weights.values)
            pooled_after = math.sqrt((t_after.var(ddof=1) + control_values.loc[weights.index].var(ddof=1)) / 2) or 1
            after = (t_after.mean() - c_after_mean) / pooled_after
            diagnostics.append(
                {
                    "feature": feature,
                    "treated_mean_before": t_all.mean(),
                    "control_mean_before": c_all.mean(),
                    "smd_before": before,
                    "treated_mean_after": t_after.mean(),
                    "control_mean_after": c_after_mean,
                    "smd_after": after,
                }
            )
    return assignments_df, pd.DataFrame(diagnostics)


def build_qa(
    schedule: pd.DataFrame,
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    matching: pd.DataFrame,
    injury_audit: pd.DataFrame,
) -> pd.DataFrame:
    matched_rows = matching[matching["match_status"].eq("matched")] if not matching.empty else matching
    treated_eligible = summary[summary["eligible_main"] & summary["treated_world_cup"].eq(1)]
    matched_treated = matched_rows["treated_player_id"].nunique() if not matched_rows.empty else 0
    valid_relative_matches = list(range(-8, 0)) + list(range(1, 9))
    pre_dates_valid = grid.loc[grid["relative_club_match"].lt(0), "match_date"].le(PRE_END).all()
    post_dates_valid = grid.loc[grid["relative_club_match"].gt(0), "match_date"].ge(POST_START).all()
    nonnegative_performance = grid[["xg", "npxg", "xa"]].fillna(0).ge(0).all().all()
    club_game_windows = grid[["club_id", "game_id", "window"]].drop_duplicates()
    unique_club_game_window = not club_game_windows.duplicated(["club_id", "game_id"]).any()
    starter_coherence = (
        grid["started"].ne(1)
        | grid["played"].eq(1)
        | grid["lineup_source_inconsistency"].eq(1)
    ).all()
    offensive_columns = ["goals", "assists", "shots", "xg", "npxg", "xa", "key_passes"]
    nonplayer_production = grid["played"].eq(0) & grid[offensive_columns].fillna(0).ne(0).any(axis=1)
    nonplayer_production_documented = (
        ~nonplayer_production | grid["performance_source_inconsistency"].eq(1)
    ).all()
    reviewed_treated = grid["treated_world_cup"].eq(1) & ~grid["sb_link_status"].eq("accepted")
    reviewed_treated_zero_dose = grid.loc[reviewed_treated, "wc_minutes_played"].fillna(0).eq(0).all()
    tests = [
        ("clubs_with_16_games", schedule.groupby("club_id")["game_id"].nunique().eq(16).sum(), schedule["club_id"].nunique(), schedule.groupby("club_id")["game_id"].nunique().eq(16).all()),
        ("schedule_club_games", len(schedule), schedule["club_id"].nunique() * 16, len(schedule) == schedule["club_id"].nunique() * 16),
        ("analytic_unique_player_game", len(grid), len(grid.drop_duplicates(["transfermarkt_player_id", "game_id"])), not grid.duplicated(["transfermarkt_player_id", "game_id"]).any()),
        ("relative_match_valid", int(grid["relative_club_match"].isin(valid_relative_matches).sum()), len(grid), grid["relative_club_match"].isin(valid_relative_matches).all()),
        ("unique_club_game_window", len(club_game_windows), len(club_game_windows.drop_duplicates(["club_id", "game_id"])), unique_club_game_window),
        ("pre_dates_before_cutoff", int(pre_dates_valid), 1, pre_dates_valid),
        ("post_dates_after_cutoff", int(post_dates_valid), 1, post_dates_valid),
        ("minutes_nonnegative", int(grid["minutes_played"].ge(0).sum()), len(grid), grid["minutes_played"].ge(0).all()),
        ("minutes_share_range", int(grid["minutes_share"].between(0, 1).sum()), len(grid), grid["minutes_share"].between(0, 1).all()),
        ("reviewed_statsbomb_links_have_zero_dose", int(grid.loc[reviewed_treated, "wc_minutes_played"].fillna(0).eq(0).sum()), int(reviewed_treated.sum()), reviewed_treated_zero_dose),
        ("nonplayers_zero_minutes", int(grid.loc[grid["played"].eq(0), "minutes_played"].eq(0).sum()), int(grid["played"].eq(0).sum()), grid.loc[grid["played"].eq(0), "minutes_played"].eq(0).all()),
        ("performance_metrics_nonnegative", int(nonnegative_performance), 1, nonnegative_performance),
        ("npxg_le_xg", int((grid["npxg"].isna() | grid["xg"].isna() | (grid["npxg"] <= grid["xg"] + 1e-10)).sum()), len(grid), (grid["npxg"].isna() | grid["xg"].isna() | (grid["npxg"] <= grid["xg"] + 1e-10)).all()),
        ("starter_inconsistencies_documented", int(grid["lineup_source_inconsistency"].sum()), "source exceptions", starter_coherence),
        ("nonplayer_production_documented", int(nonplayer_production.sum()), "source exceptions", nonplayer_production_documented),
        ("roster_assignment_conflicts_resolved", int(grid["roster_assignment_conflict"].sum()), "informational", True),
        ("eligible_main_players", int(summary["eligible_main"].sum()), "informational", summary["eligible_main"].sum() > 500),
        ("eligible_main_treated", len(treated_eligible), "informational", len(treated_eligible) > 100),
        ("treated_matching_rate", matched_treated / max(len(treated_eligible), 1), ">=0.80", matched_treated / max(len(treated_eligible), 1) >= 0.80),
        ("january_transfers_flagged", int(summary["january_transfer"].sum()), "informational", True),
        ("long_injuries_flagged", int(summary["long_injury_crossing"].sum()), "informational", True),
        ("injury_rows_relevant", int(injury_audit.loc[injury_audit["injury_check"].eq("relevant_date_rows"), "value"].iloc[0]), "informational", True),
    ]
    return pd.DataFrame(tests, columns=["check", "observed", "expected_or_threshold", "passed"])


def build_data_dictionary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    units = {
        "analytic_player_match.csv": "jogador + partida",
        "analytic_player_windows.csv": "jogador + clube + janela",
        "analysis_sample_main.csv": "jogador + clube",
        "analysis_sample_availability.csv": "jogador + clube",
        "treated_control_candidates.csv": "jogador + clube",
        "matching_assignments.csv": "tratado + controle",
        "matching_balance.csv": "covariável",
        "sample_exclusions.csv": "jogador + clube",
        "roster_spells.csv": "jogador + clube + vínculo",
        "injury_source_audit.csv": "controle de auditoria",
        "panel_qa_summary.csv": "controle de qualidade",
    }
    descriptions = {
        "transfermarkt_player_id": "Identificador do jogador no Transfermarkt.",
        "game_id": "Identificador da partida no Transfermarkt.",
        "club_id": "Identificador do clube no Transfermarkt.",
        "match_date": "Data da partida.",
        "relative_club_match": "Posição da partida na janela: -8 a -1 ou 1 a 8.",
        "window": "Janela analítica: pre, post_acute ou post_consolidated.",
        "treated_world_cup": "Indicador de convocação para a Copa do Mundo de 2022.",
        "minutes_played": "Minutos registrados na fonte; ausências conhecidas não são imputadas.",
        "minutes_share": "Minutos jogados divididos por 90.",
        "in_match_squad": "Indicador de presença na súmula da partida.",
        "injured_on_match_date": "Indicador de lesão ativa na data da partida.",
        "january_transfer": "Indicador de transferência em janeiro de 2023.",
        "long_injury_crossing": "Indicador de lesão longa atravessando o período da Copa.",
        "market_value_in_eur": "Valor de mercado mais recente, em euros, anterior ao corte.",
        "eligible_main": "Elegibilidade para a amostra analítica principal.",
        "eligible_availability": "Elegibilidade para a amostra de disponibilidade.",
        "exclusion_reasons": "Motivos de exclusão separados por barra vertical.",
        "lineup_source_inconsistency": "Exceção documentada: titular sem participação ou minutos na fonte.",
        "performance_source_inconsistency": "Exceção documentada: produção Understat sem participação na fonte de escalação.",
        "roster_assignment_conflict": "Indica mais de um vínculo de clube elegível antes da regra de desempate.",
        "roster_assignment_rule": "Regra usada para atribuir o jogador a um único clube na partida.",
    }
    rows: list[dict[str, object]] = []
    for filename, frame in outputs.items():
        for column in frame.columns:
            description = descriptions.get(column)
            if description is None:
                if column.startswith("wc_"):
                    description = "Métrica agregada na Copa do Mundo de 2022."
                elif column.startswith("pre_"):
                    description = "Covariável ou outcome agregado na janela pré-Copa."
                elif column.startswith("post_acute_"):
                    description = "Outcome agregado nos quatro primeiros jogos pós-Copa."
                elif column.startswith("post_consolidated_"):
                    description = "Outcome agregado do quinto ao oitavo jogo pós-Copa."
                elif column.endswith("_p90"):
                    description = "Métrica normalizada por 90 minutos."
                elif column.endswith("_z") or column.endswith("_robust_z"):
                    description = "Métrica padronizada por liga e grupo posicional."
                else:
                    description = "Campo preservado ou derivado pelo pipeline; consulte o script para a regra exata."
            rows.append(
                {
                    "file": filename,
                    "unit_of_observation": units.get(filename, "conforme arquivo"),
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT / filename, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {filename}: {len(frame):,} rows x {len(frame.columns)} cols")


def write_manifest() -> None:
    records: list[dict[str, object]] = []
    for path in sorted(OUT.glob("*.csv")):
        if path.name == "manifest.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        with path.open("rb") as stream:
            rows = max(sum(1 for _ in stream) - 1, 0)
        records.append(
            {
                "relative_path": path.name,
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "generated_by": "scripts/analysis/build_analytic_panel.py",
            }
        )
    write_csv(pd.DataFrame(records), "manifest.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs()
    panel = inputs["big5_player_match_2022_23"].copy()
    conn = open_transfermarkt()
    player_ids = set(panel["transfermarkt_player_id"].dropna().astype(int))
    club_ids = set(panel["club_id"].dropna().astype(int))
    transfers, valuations = load_reference_tables(conn, player_ids, club_ids)
    full_schedule, schedule = build_club_schedule(panel)
    spells = build_roster_spells(panel, transfers)
    grid = complete_player_game_grid(spells, schedule, panel)
    grid, injury_summary, injury_audit = add_injuries(grid)
    grid = add_understat(grid, inputs)
    grid = add_treatment_and_covariates(grid, inputs, valuations, injury_summary, transfers, full_schedule)
    grid = add_normalizations(grid)
    windows = build_window_aggregates(grid)
    summary = build_player_summary(grid, windows)

    main_sample = summary[summary["eligible_main"]].copy()
    availability_sample = summary[summary["eligible_availability"]].copy()
    candidates = main_sample.copy()
    matching, balance = perform_matching(candidates)
    exclusions = summary[~summary["eligible_main"]].copy()

    # Convocados Big Five que não chegaram a nenhuma grade recebem exclusão explícita.
    treated_ids = set(
        inputs["fifa_player_crosswalk"].loc[
            inputs["fifa_player_crosswalk"]["big5_at_world_cup"].eq(True), "transfermarkt_player_id"
        ].dropna().astype(int)
    )
    missing_treated = treated_ids - set(summary["transfermarkt_player_id"].astype(int))
    if missing_treated:
        extra = pd.DataFrame(
            {"transfermarkt_player_id": sorted(missing_treated), "treated_world_cup": 1, "eligible_main": False, "exclusion_reasons": "no_roster_panel"}
        )
        exclusions = pd.concat([exclusions, extra], ignore_index=True, sort=False)

    qa = build_qa(schedule, grid, summary, matching, injury_audit)
    output_frames = {
        "roster_spells.csv": spells,
        "analytic_player_match.csv": grid,
        "analytic_player_windows.csv": windows,
        "analysis_sample_main.csv": main_sample,
        "analysis_sample_availability.csv": availability_sample,
        "treated_control_candidates.csv": candidates,
        "matching_assignments.csv": matching,
        "matching_balance.csv": balance,
        "sample_exclusions.csv": exclusions,
        "injury_source_audit.csv": injury_audit,
        "panel_qa_summary.csv": qa,
    }
    dictionary = build_data_dictionary(output_frames)

    write_csv(spells, "roster_spells.csv")
    write_csv(grid, "analytic_player_match.csv")
    write_csv(windows, "analytic_player_windows.csv")
    write_csv(main_sample, "analysis_sample_main.csv")
    write_csv(availability_sample, "analysis_sample_availability.csv")
    write_csv(candidates, "treated_control_candidates.csv")
    write_csv(matching, "matching_assignments.csv")
    write_csv(balance, "matching_balance.csv")
    write_csv(exclusions, "sample_exclusions.csv")
    write_csv(injury_audit, "injury_source_audit.csv")
    write_csv(dictionary, "data_dictionary.csv")
    write_csv(qa, "panel_qa_summary.csv")
    write_manifest()
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(f"QA da Fase 3 falhou: {failed}. Consulte data/processed/panel_qa_summary.csv")
    print("Fase 3 concluída com todos os controles críticos aprovados.")


if __name__ == "__main__":
    main()
