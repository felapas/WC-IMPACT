"""Constrói as tabelas intermediárias auditáveis do estudo FAME 2026.

Entradas brutas nunca são alteradas. As saídas em data/interim podem ser
recriadas executando este script a partir da raiz do repositório.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import pdfplumber  # noqa: E402


RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim"
BIG5 = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
}
BIG5_COUNTRY_CODES = {"ENG", "ESP", "ITA", "GER", "FRA"}
STAGE_ORDER = {
    "Group Stage": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "3rd Place Final": 5,
    "Final": 6,
}
FIFA_TO_STATSBOMB_TEAM = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
}


def normalize(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def name_score(left: object, right: object) -> float:
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    token = len(ta & tb) / max(len(ta | tb), 1)
    containment = min(len(ta & tb) / max(len(ta), 1), len(ta & tb) / max(len(tb), 1))
    return max(seq, 0.65 * containment + 0.35 * token)


def club_score(left: object, right: object) -> float:
    """Similaridade tolerante a nomes legais longos de clubes."""
    stop = {
        "fc", "cf", "club", "football", "futbol", "calcio", "association",
        "associazione", "societa", "sportiva", "sporting", "spa", "sad",
        "team", "the", "de", "del", "da", "do", "s", "a", "d",
    }
    a_tokens = [token for token in normalize(left).split() if token not in stop]
    b_tokens = [token for token in normalize(right).split() if token not in stop]
    if not a_tokens or not b_tokens:
        return name_score(left, right)
    a, b = " ".join(a_tokens), " ".join(b_tokens)
    if a == b:
        return 1.0
    ta, tb = set(a_tokens), set(b_tokens)
    common = len(ta & tb)
    coverage = max(common / len(ta), common / len(tb))
    jaccard = common / max(len(ta | tb), 1)
    seq = SequenceMatcher(None, a, b).ratio()
    # Ajuda abreviações como Inter/Internazionale e Lyon/Lyonnais.
    prefix = max(
        (SequenceMatcher(None, x, y).ratio() for x in a_tokens for y in b_tokens),
        default=0.0,
    )
    return max(seq, 0.80 * coverage + 0.20 * jaccard, 0.92 * prefix)


def best_name_score(names: list[object], candidate_names: list[object]) -> float:
    return max((name_score(a, b) for a in names for b in candidate_names), default=0.0)


def timestamp_seconds(value: str | None) -> float:
    """Converte o timestamp StatsBomb (relativo ao período) em segundos."""
    if not value:
        return 0.0
    hour, minute, second = value.split(":", 2)
    return int(hour) * 3600 + int(minute) * 60 + float(second)


def statsbomb_match_clock(
    events: list[dict[str, object]],
) -> tuple[dict[int, float], dict[int, float], float]:
    """Reconstrói um relógio contínuo, preservando acréscimos reais.

    O timestamp do StatsBomb reinicia a cada período. Somar os relógios de
    períodos anteriores evita confundir, por exemplo, 02:00 do segundo tempo
    com 02:00 de jogo. Disputas por pênaltis (período 5) não contam como
    exposição em campo.
    """
    period_durations: dict[int, float] = {}
    for period in range(1, 5):
        period_events = [event for event in events if int(event.get("period", 0)) == period]
        if not period_events:
            continue
        half_ends = [
            timestamp_seconds(event.get("timestamp"))
            for event in period_events
            if event.get("type", {}).get("name") == "Half End"
        ]
        candidates = half_ends or [timestamp_seconds(event.get("timestamp")) for event in period_events]
        duration = max(candidates, default=0.0)
        if duration <= 0:
            raise RuntimeError(f"Período StatsBomb {period} sem duração positiva")
        period_durations[period] = duration

    if not {1, 2}.issubset(period_durations):
        raise RuntimeError("Partida StatsBomb sem os dois períodos regulamentares")
    if (3 in period_durations) != (4 in period_durations):
        raise RuntimeError("Prorrogação StatsBomb incompleta")

    period_offsets: dict[int, float] = {}
    elapsed = 0.0
    for period in sorted(period_durations):
        period_offsets[period] = elapsed
        elapsed += period_durations[period]
    return period_durations, period_offsets, elapsed


def statsbomb_event_elapsed(event: dict[str, object], offsets: dict[int, float]) -> float:
    period = int(event.get("period", 0))
    return offsets[period] + timestamp_seconds(event.get("timestamp"))


def statsbomb_player_stints(
    events: list[dict[str, object]], lineups: list[dict[str, object]]
) -> tuple[dict[int, dict[str, float | int]], float, float | None]:
    """Calcula entrada/saída a partir de XI inicial, substituições e expulsões."""
    _, offsets, game_end = statsbomb_match_clock(events)
    extra_time_start = offsets.get(3)
    lineup_team = {
        int(player["player_id"]): int(team["team_id"])
        for team in lineups
        for player in team["lineup"]
    }
    starters: dict[int, int] = {}
    time_on: dict[int, float] = {}
    time_off: dict[int, float] = {}
    active: set[int] = set()
    dismissals: dict[int, list[float]] = defaultdict(list)

    match_events = [event for event in events if int(event.get("period", 0)) in offsets]
    match_events.sort(key=lambda event: (statsbomb_event_elapsed(event, offsets), int(event.get("index", 0))))

    for event in match_events:
        if event.get("type", {}).get("name") != "Starting XI":
            continue
        team_id = int(event["team"]["id"])
        tactical_lineup = event.get("tactics", {}).get("lineup", [])
        if len(tactical_lineup) != 11:
            raise RuntimeError(f"XI inicial StatsBomb com {len(tactical_lineup)} jogadores")
        for entry in tactical_lineup:
            player_id = int(entry["player"]["id"])
            if lineup_team.get(player_id) != team_id:
                raise RuntimeError("Jogador do XI inicial ausente da escalação da equipe")
            if player_id in active:
                raise RuntimeError("Jogador duplicado no XI inicial")
            starters[player_id] = 1
            time_on[player_id] = 0.0
            active.add(player_id)

    team_starters = defaultdict(int)
    for player_id in starters:
        team_starters[lineup_team[player_id]] += 1
    if sorted(team_starters.values()) != [11, 11]:
        raise RuntimeError(f"Contagem inválida de titulares por equipe: {dict(team_starters)}")

    card_names = {"Red Card", "Second Yellow"}
    for event in match_events:
        event_type = event.get("type", {}).get("name", "")
        if event_type == "Starting XI":
            continue
        event_time = statsbomb_event_elapsed(event, offsets)
        player = event.get("player") or {}
        player_id = int(player["id"]) if player.get("id") is not None else None

        if event_type == "Substitution":
            replacement = event.get("substitution", {}).get("replacement") or {}
            replacement_id = int(replacement["id"]) if replacement.get("id") is not None else None
            if player_id is None or replacement_id is None:
                raise RuntimeError("Substituição StatsBomb sem jogador de saída ou entrada")
            if player_id not in active:
                raise RuntimeError(f"Substituição retira jogador inativo: {player_id}")
            if replacement_id in active or replacement_id in time_on:
                raise RuntimeError(f"Substituição insere jogador já utilizado: {replacement_id}")
            if lineup_team.get(player_id) != lineup_team.get(replacement_id):
                raise RuntimeError("Substituição entre equipes diferentes")
            time_off[player_id] = event_time
            active.remove(player_id)
            time_on[replacement_id] = event_time
            active.add(replacement_id)
            continue

        card = (
            event.get("foul_committed", {}).get("card", {}).get("name")
            or event.get("bad_behaviour", {}).get("card", {}).get("name")
        )
        if card in card_names and player_id is not None:
            if player_id not in active:
                previous = time_off.get(player_id)
                if previous is not None and abs(previous - event_time) <= 1.0:
                    continue
                raise RuntimeError(f"Expulsão de jogador inativo: {player_id}")
            time_off[player_id] = event_time
            active.remove(player_id)
            dismissals[lineup_team[player_id]].append(event_time)

    stints: dict[int, dict[str, float | int]] = {}
    for player_id, team_id in lineup_team.items():
        on = time_on.get(player_id)
        off = time_off.get(player_id, game_end) if on is not None else None
        if on is None:
            seconds = 0.0
            extra_seconds = 0.0
        else:
            if off is None or off < on or off > game_end + 1e-6:
                raise RuntimeError(f"Intervalo inválido para jogador StatsBomb {player_id}")
            seconds = off - on
            extra_seconds = (
                max(0.0, off - max(on, extra_time_start))
                if extra_time_start is not None
                else 0.0
            )
        stints[player_id] = {
            "started": starters.get(player_id, 0),
            "played": int(seconds > 0),
            "seconds_played": seconds,
            "extra_time_seconds": extra_seconds,
        }

    for team in lineups:
        team_id = int(team["team_id"])
        observed = sum(float(stints[int(player["player_id"])]["seconds_played"]) for player in team["lineup"])
        expected = 11 * game_end - sum(game_end - time for time in dismissals.get(team_id, []))
        if abs(observed - expected) > 0.01:
            raise RuntimeError(
                f"Exposição coletiva incoerente ({team['team_name']}): "
                f"observada={observed:.3f}, esperada={expected:.3f}"
            )
    return stints, game_end, extra_time_start


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    path = OUT / filename
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {filename}: {len(frame):,} rows x {len(frame.columns)} cols")


def extract_fifa_squads() -> pd.DataFrame:
    path = RAW / "fifa" / "wc_2022_squads" / "SquadLists-English.pdf"
    rows: list[dict[str, object]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            country_match = re.search(r"\n([^\n]+) \(([A-Z]{3})\)\n", "\n" + text)
            if not country_match:
                raise RuntimeError(f"País não identificado na página {page_number}")
            country_name, country_code = country_match.groups()
            tables = page.extract_tables()
            # A maioria das seleções levou 26 jogadores; o IR Iran aparece com
            # 25 no documento final. Com o cabeçalho, isso produz 27 ou 26 linhas.
            if not tables or len(tables[0]) not in {26, 27}:
                raise RuntimeError(f"Tabela de elenco inesperada na página {page_number}")
            if tables[0][0][:3] != ["#", "POS", "PLAYER NAME"]:
                raise RuntimeError(f"Cabeçalho de elenco inesperado na página {page_number}")
            for raw in tables[0][1:]:
                if len(raw) != 11:
                    raise RuntimeError(f"Linha FIFA com {len(raw)} colunas na página {page_number}")
                number, pos, player_name, first_names, last_name, shirt_name, dob, club, height, caps, goals = raw
                club_country_match = re.search(r"\(([A-Z]{3})\)\s*$", club or "")
                club_country = club_country_match.group(1) if club_country_match else ""
                rows.append(
                    {
                        "fifa_squad_id": f"{country_code}-{int(number):02d}",
                        "country_name": country_name,
                        "country_code": country_code,
                        "shirt_number": int(number),
                        "fifa_position": pos,
                        "player_name_fifa": player_name,
                        "first_names_fifa": first_names,
                        "last_name_fifa": last_name,
                        "shirt_name_fifa": shirt_name,
                        "date_of_birth": pd.to_datetime(dob, dayfirst=True).date().isoformat(),
                        "club_at_world_cup": club,
                        "club_country_code": club_country,
                        "height_cm_fifa": int(height) if height else pd.NA,
                        "caps_pre_final": int(caps) if caps else pd.NA,
                        "goals_pre_final": int(goals) if goals else pd.NA,
                        "is_goalkeeper": pos == "GK",
                        # Candidato geográfico apenas. A classificação final exige
                        # que o jogador seja ligado a uma equipe da primeira divisão.
                        "club_country_big5_candidate": club_country in BIG5_COUNTRY_CODES,
                        "source_page": page_number,
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 831 or result["country_code"].nunique() != 32:
        raise RuntimeError(f"Cobertura FIFA inválida: {len(result)} jogadores, {result['country_code'].nunique()} seleções")
    return result


def open_transfermarkt() -> duckdb.DuckDBPyConnection:
    path = RAW / "transfermarkt" / "dcaribou" / "transfermarkt-datasets.duckdb"
    return duckdb.connect(str(path), read_only=True)


def build_big5_panel(conn: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = ",".join(f"'{key}'" for key in BIG5)
    games = conn.sql(
        f"""
        SELECT CAST(game_id AS INTEGER) game_id, competition_id, CAST(season AS INTEGER) season,
               round, date, home_club_id, away_club_id, home_club_name, away_club_name,
               home_club_goals, away_club_goals, home_club_position, away_club_position,
               home_club_formation, away_club_formation
        FROM games
        WHERE CAST(season AS INTEGER)=2022 AND competition_id IN ({ids})
        """
    ).df()
    if len(games) != 1826:
        raise RuntimeError(f"Esperados 1.826 jogos Big Five; encontrados {len(games)}")

    panel = conn.sql(
        f"""
        WITH selected_games AS (
            SELECT CAST(game_id AS INTEGER) game_id, competition_id, CAST(season AS INTEGER) season,
                   round, date, home_club_id, away_club_id, home_club_name, away_club_name,
                   home_club_goals, away_club_goals, home_club_position, away_club_position,
                   home_club_formation, away_club_formation
            FROM games
            WHERE CAST(season AS INTEGER)=2022 AND competition_id IN ({ids})
        ), club_schedule AS (
            SELECT game_id, date, home_club_id club_id FROM selected_games
            UNION ALL
            SELECT game_id, date, away_club_id club_id FROM selected_games
        ), rests AS (
            SELECT game_id, club_id,
                   date_diff('day', lag(date) OVER (PARTITION BY club_id ORDER BY date, game_id), date) days_since_club_game
            FROM club_schedule
        ), appearance_one AS (
            SELECT CAST(game_id AS INTEGER) game_id, player_id,
                   max(minutes_played) minutes_played, max(goals) goals, max(assists) assists,
                   max(yellow_cards) yellow_cards, max(red_cards) red_cards
            FROM appearances
            GROUP BY 1,2
        )
        SELECT l.game_id, g.date match_date, g.competition_id,
               l.club_id, CASE WHEN l.club_id=g.home_club_id THEN g.home_club_name ELSE g.away_club_name END club_name,
               CASE WHEN l.club_id=g.home_club_id THEN g.away_club_id ELSE g.home_club_id END opponent_club_id,
               CASE WHEN l.club_id=g.home_club_id THEN g.away_club_name ELSE g.home_club_name END opponent_club_name,
               (l.club_id=g.home_club_id)::INTEGER is_home,
               CASE WHEN l.club_id=g.home_club_id THEN g.home_club_goals ELSE g.away_club_goals END club_goals,
               CASE WHEN l.club_id=g.home_club_id THEN g.away_club_goals ELSE g.home_club_goals END opponent_goals,
               g.round, CASE WHEN l.club_id=g.home_club_id THEN g.home_club_position ELSE g.away_club_position END club_table_position,
               CASE WHEN l.club_id=g.home_club_id THEN g.away_club_position ELSE g.home_club_position END opponent_table_position,
               CASE WHEN l.club_id=g.home_club_id THEN g.home_club_formation ELSE g.away_club_formation END club_formation,
               r.days_since_club_game,
               l.player_id transfermarkt_player_id, l.player_name, l.position lineup_position,
               (l.type='starting_lineup')::INTEGER started, l.team_captain,
               coalesce(a.minutes_played,0) minutes_played, coalesce(a.goals,0) goals,
               coalesce(a.assists,0) assists, coalesce(a.yellow_cards,0) yellow_cards,
               coalesce(a.red_cards,0) red_cards,
               p.date_of_birth, p.position broad_position, p.sub_position,
               p.country_of_citizenship, p.foot, p.height_in_cm
        FROM game_lineups l
        JOIN selected_games g ON CAST(l.game_id AS INTEGER)=g.game_id
        LEFT JOIN appearance_one a ON CAST(l.game_id AS INTEGER)=a.game_id AND l.player_id=a.player_id
        LEFT JOIN players p ON l.player_id=p.player_id
        LEFT JOIN rests r ON g.game_id=r.game_id AND l.club_id=r.club_id
        """
    ).df()
    panel["league_name"] = panel["competition_id"].map(BIG5)
    panel["played"] = (panel["minutes_played"] > 0).astype("int8")
    panel["is_goalkeeper"] = panel["lineup_position"].eq("Goalkeeper") | panel["broad_position"].eq("Goalkeeper")
    panel = panel.sort_values(["match_date", "game_id", "club_id", "transfermarkt_player_id"]).reset_index(drop=True)
    if panel.duplicated(["game_id", "transfermarkt_player_id"]).any():
        raise RuntimeError("Duplicatas game_id + player_id no painel Big Five")
    return panel, games


def wc_event_metrics(events: list[dict[str, object]]) -> dict[int, dict[str, float]]:
    metrics: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    key_pass_xg: dict[str, float] = {}
    for event in events:
        if event.get("type", {}).get("name") == "Shot":
            key = event.get("shot", {}).get("key_pass_id")
            if key:
                key_pass_xg[str(key)] = float(event.get("shot", {}).get("statsbomb_xg") or 0)
    for event in events:
        player = event.get("player")
        if not player:
            continue
        pid = int(player["id"])
        row = metrics[pid]
        event_type = event.get("type", {}).get("name", "")
        row["actions"] += 1
        row["under_pressure_actions"] += int(bool(event.get("under_pressure")))
        if event_type == "Shot":
            shot = event.get("shot", {})
            xg = float(shot.get("statsbomb_xg") or 0)
            row["shots"] += 1
            row["xg"] += xg
            row["npxg"] += xg if shot.get("type", {}).get("name") != "Penalty" else 0
            row["goals"] += int(shot.get("outcome", {}).get("name") == "Goal")
        elif event_type == "Pass":
            data = event.get("pass", {})
            row["passes"] += 1
            row["completed_passes"] += int("outcome" not in data)
            row["xa"] += key_pass_xg.get(str(event.get("id")), 0.0)
            start, end = event.get("location"), data.get("end_location")
            if start and end and end[0] - start[0] >= 10 and end[0] >= 60:
                row["progressive_passes_proxy"] += 1
        elif event_type == "Carry":
            row["carries"] += 1
            start, end = event.get("location"), event.get("carry", {}).get("end_location")
            if start and end and end[0] - start[0] >= 10 and end[0] >= 60:
                row["progressive_carries_proxy"] += 1
        elif event_type == "Pressure":
            row["pressures"] += 1
        elif event_type == "Ball Recovery":
            row["ball_recoveries"] += 1
        elif event_type == "Dribble":
            row["dribbles"] += 1
            row["completed_dribbles"] += int(event.get("dribble", {}).get("outcome", {}).get("name") == "Complete")
        elif event_type == "Duel":
            row["duels"] += 1
            row["duels_won"] += int(event.get("duel", {}).get("outcome", {}).get("name") in {"Won", "Success", "Success In Play"})
    return metrics


def build_wc_player_match() -> tuple[pd.DataFrame, pd.DataFrame]:
    sb = RAW / "statsbomb" / "wc_2022"
    matches = json.loads((sb / "data" / "matches" / "43" / "106.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    metric_columns = [
        "actions", "under_pressure_actions", "shots", "goals", "xg", "npxg", "passes",
        "completed_passes", "xa", "progressive_passes_proxy", "carries",
        "progressive_carries_proxy", "pressures", "ball_recoveries", "dribbles",
        "completed_dribbles", "duels", "duels_won",
    ]
    for match in sorted(matches, key=lambda x: (x["match_date"], x["match_id"])):
        match_id = int(match["match_id"])
        events = json.loads((sb / "data" / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
        lineups = json.loads((sb / "data" / "lineups" / f"{match_id}.json").read_text(encoding="utf-8"))
        stints, game_end, extra_time_start = statsbomb_player_stints(events, lineups)
        extra_time_game = extra_time_start is not None
        event_metrics = wc_event_metrics(events)
        stage = match["competition_stage"]["name"]
        for team in lineups:
            for player in team["lineup"]:
                pid = int(player["player_id"])
                stint = stints[pid]
                seconds = float(stint["seconds_played"])
                primary_position = ""
                for pos in player.get("positions", []):
                    if not primary_position:
                        primary_position = pos.get("position", "")
                row: dict[str, object] = {
                    "statsbomb_match_id": match_id,
                    "match_date": match["match_date"],
                    "stage": stage,
                    "stage_order": STAGE_ORDER.get(stage, 0),
                    "team_name": team["team_name"],
                    "opponent_name": match["away_team"]["away_team_name"] if team["team_name"] == match["home_team"]["home_team_name"] else match["home_team"]["home_team_name"],
                    "statsbomb_player_id": pid,
                    "player_name_statsbomb": player["player_name"],
                    "shirt_number_statsbomb": player.get("jersey_number"),
                    "position_statsbomb": primary_position,
                    "started": int(stint["started"]),
                    "played": int(stint["played"]),
                    "minutes_played": round(seconds / 60, 3),
                    "extra_time_game": int(extra_time_game),
                    "extra_time_minutes": round(float(stint["extra_time_seconds"]) / 60, 3),
                    "match_duration_minutes": round(game_end / 60, 3),
                }
                for column in metric_columns:
                    row[column] = event_metrics.get(pid, {}).get(column, 0.0)
                records.append(row)
    player_match = pd.DataFrame(records).sort_values(["match_date", "statsbomb_match_id", "team_name", "statsbomb_player_id"])
    if player_match.duplicated(["statsbomb_match_id", "statsbomb_player_id"]).any():
        raise RuntimeError("Duplicatas StatsBomb match + player")
    numeric_sum = [
        "started", "played", "minutes_played", "extra_time_game",
        "extra_time_minutes", "match_duration_minutes",
    ] + metric_columns
    tournament = player_match.groupby(
        ["statsbomb_player_id", "player_name_statsbomb", "team_name"], as_index=False
    ).agg(
        **{column: (column, "sum") for column in numeric_sum},
        squad_matchdays=("statsbomb_match_id", "nunique"),
        max_stage_order=("stage_order", "max"),
    )
    for metric in ["xg", "npxg", "xa", "shots", "passes", "progressive_passes_proxy", "progressive_carries_proxy", "pressures"]:
        tournament[f"{metric}_p90"] = tournament[metric] * 90 / tournament["minutes_played"].replace(0, pd.NA)
    return player_match.reset_index(drop=True), tournament


def build_understat_player_match() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = RAW / "understat" / "extracted" / "understats"
    league_specs = {
        "EPL": ("GB1", ";"),
        "La_Liga": ("ES1", ";"),
        "Bundesliga": ("L1", ";"),
        "Serie_A": ("IT1", ","),
        "Ligue_1": ("FR1", ","),
    }
    all_player_match: list[pd.DataFrame] = []
    all_matches: list[pd.DataFrame] = []
    for folder, (competition_id, sep) in league_specs.items():
        shots = pd.read_csv(root / folder / "shot_data.csv", sep=sep)
        matches = pd.read_csv(root / folder / "match_info.csv", sep=sep)
        shots = shots.loc[shots["season"].eq(2022)].copy()
        matches = matches.loc[matches["season"].eq(2022)].copy()
        matches["competition_id"] = competition_id
        matches["match_date"] = pd.to_datetime(matches["date"]).dt.date.astype(str)
        all_matches.append(matches[["id", "competition_id", "match_date", "team_h", "team_a", "h_goals", "a_goals"]].rename(columns={"id": "understat_match_id"}))

        shots["understat_match_id"] = shots["match_id"].astype(int)
        shots["match_date"] = pd.to_datetime(shots["date"]).dt.date.astype(str)
        shots["competition_id"] = competition_id
        shots["team_name"] = shots.apply(lambda r: r["h_team"] if r["h_a"] == "h" else r["a_team"], axis=1)
        shots["opponent_name"] = shots.apply(lambda r: r["a_team"] if r["h_a"] == "h" else r["h_team"], axis=1)
        shots["goal"] = shots["result"].eq("Goal").astype(int)
        shots["shot_on_target"] = shots["result"].isin(["Goal", "SavedShot"]).astype(int)
        shots["penalty"] = shots["situation"].eq("Penalty").astype(int)
        shots["npxg"] = shots["xG"] * (1 - shots["penalty"])
        keys = ["competition_id", "understat_match_id", "match_date", "team_name", "opponent_name", "player_id", "player"]
        shooter = shots.groupby(keys, dropna=False, as_index=False).agg(
            shots=("id", "count"), goals=("goal", "sum"), shots_on_target=("shot_on_target", "sum"),
            penalties=("penalty", "sum"), xg=("xG", "sum"), npxg=("npxg", "sum"),
        ).rename(columns={"player_id": "understat_player_id", "player": "player_name_understat"})

        assists = shots.dropna(subset=["player_assisted"]).groupby(
            ["competition_id", "understat_match_id", "match_date", "team_name", "opponent_name", "player_assisted"],
            as_index=False,
        ).agg(xa=("xG", "sum"), key_passes=("id", "count")).rename(columns={"player_assisted": "player_name_understat"})
        name_to_id = shooter.groupby("player_name_understat")["understat_player_id"].agg(lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA)
        assists["understat_player_id"] = assists["player_name_understat"].map(name_to_id)
        merge_keys = ["competition_id", "understat_match_id", "match_date", "team_name", "opponent_name", "understat_player_id", "player_name_understat"]
        combined = shooter.merge(assists, on=merge_keys, how="outer")
        for column in ["shots", "goals", "shots_on_target", "penalties", "xg", "npxg", "xa", "key_passes"]:
            combined[column] = combined[column].fillna(0)
        all_player_match.append(combined)
    player_match = pd.concat(all_player_match, ignore_index=True)
    matches = pd.concat(all_matches, ignore_index=True)
    if len(matches) != 1826:
        raise RuntimeError(f"Cobertura Understat inválida: {len(matches)} jogos")
    return player_match.sort_values(["match_date", "competition_id", "understat_match_id", "team_name", "player_name_understat"]), matches


def match_understat_games(understat_games: pd.DataFrame, tm_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tm_by_competition = {key: group.copy() for key, group in tm_games.groupby("competition_id")}
    for group in tm_by_competition.values():
        group["date"] = pd.to_datetime(group["date"])
    for row in understat_games.itertuples(index=False):
        competition_games = tm_by_competition.get(row.competition_id, pd.DataFrame())
        target_date = pd.Timestamp(row.match_date)
        candidates = competition_games[(competition_games["date"] - target_date).abs().dt.days <= 2]
        best = None
        best_score = -1.0
        for candidate in candidates.itertuples(index=False):
            home = club_score(row.team_h, candidate.home_club_name)
            away = club_score(row.team_a, candidate.away_club_name)
            scoreline = int(row.h_goals == candidate.home_club_goals and row.a_goals == candidate.away_club_goals)
            day_gap = abs((pd.Timestamp(candidate.date) - target_date).days)
            date_score = {0: 1.0, 1: 0.8, 2: 0.6}[day_gap]
            score = 0.42 * home + 0.42 * away + 0.10 * scoreline + 0.06 * date_score
            if score > best_score:
                best_score, best = score, candidate
        rows.append(
            {
                "competition_id": row.competition_id,
                "understat_match_id": row.understat_match_id,
                "match_date": row.match_date,
                "understat_home": row.team_h,
                "understat_away": row.team_a,
                "transfermarkt_game_id": getattr(best, "game_id", pd.NA),
                "transfermarkt_home": getattr(best, "home_club_name", ""),
                "transfermarkt_away": getattr(best, "away_club_name", ""),
                "match_score": round(best_score, 4),
                "match_status": "accepted" if best_score >= 0.68 else "review",
            }
        )
    return pd.DataFrame(rows)


def link_understat_players(understat_pm: pd.DataFrame, game_map: pd.DataFrame, big5_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = understat_pm.merge(game_map[["understat_match_id", "transfermarkt_game_id", "match_status"]], on="understat_match_id", how="left")
    tm_by_game = {game_id: group for game_id, group in big5_panel.groupby("game_id")}
    mapped_by_player = {
        key: group for key, group in mapped.groupby(["competition_id", "understat_player_id"], dropna=False)
    }
    links: dict[tuple[str, int], dict[str, object]] = {}
    reviews: list[dict[str, object]] = []
    unique_players = mapped[["competition_id", "understat_player_id", "player_name_understat"]].drop_duplicates()
    for player in unique_players.itertuples(index=False):
        appearances = mapped_by_player.get((player.competition_id, player.understat_player_id), pd.DataFrame())
        candidate_scores: dict[int, list[float]] = defaultdict(list)
        candidate_names: dict[int, str] = {}
        for app in appearances.itertuples(index=False):
            candidates = tm_by_game.get(app.transfermarkt_game_id, pd.DataFrame())
            for candidate in candidates.itertuples(index=False):
                score = name_score(player.player_name_understat, candidate.player_name)
                if score >= 0.45:
                    candidate_scores[int(candidate.transfermarkt_player_id)].append(score)
                    candidate_names[int(candidate.transfermarkt_player_id)] = candidate.player_name
        ranked = sorted(
            ((sum(scores) / len(scores), len(scores), pid) for pid, scores in candidate_scores.items()),
            reverse=True,
        )
        top_score, evidence, tm_id = ranked[0] if ranked else (0.0, 0, pd.NA)
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        margin_ok = top_score - second >= 0.04
        accepted = (
            (top_score >= 0.85 and margin_ok)
            or (top_score >= 0.72 and evidence >= 2 and (margin_ok or evidence >= 3))
        )
        record = {
            "competition_id": player.competition_id,
            "understat_player_id": player.understat_player_id,
            "player_name_understat": player.player_name_understat,
            "transfermarkt_player_id": tm_id,
            "player_name_transfermarkt": candidate_names.get(tm_id, ""),
            "link_score": round(top_score, 4),
            "evidence_matches": evidence,
            "second_best_score": round(second, 4),
            "link_status": "accepted" if accepted else ("review" if ranked else "unmatched"),
        }
        links[(player.competition_id, player.understat_player_id)] = record
        if not accepted:
            reviews.append({"source": "understat_to_transfermarkt", **record})
    return pd.DataFrame(links.values()), pd.DataFrame(reviews)


def link_fifa_players(fifa: pd.DataFrame, wc_players: pd.DataFrame, big5_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # O tratamento é definido pela situação imediatamente antes da Copa.
    # Isso impede que uma transferência de janeiro para a Big Five transforme
    # retrospectivamente um jogador de outra liga em tratado.
    pre_world_cup = big5_panel[pd.to_datetime(big5_panel["match_date"]) <= pd.Timestamp("2022-11-13")]
    tm_players = pre_world_cup[[
        "transfermarkt_player_id", "player_name", "date_of_birth", "club_name", "broad_position", "sub_position"
    ]].drop_duplicates(["transfermarkt_player_id"])
    tm_players["date_of_birth"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce").dt.date.astype(str)
    tm_by_dob = {dob: group for dob, group in tm_players.groupby("date_of_birth")}
    sb_unique = wc_players[["statsbomb_player_id", "player_name_statsbomb", "team_name", "shirt_number_statsbomb"]].drop_duplicates(["statsbomb_player_id"])
    sb_teams = list(sb_unique["team_name"].dropna().unique())

    rows: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    for fifa_row in fifa.itertuples(index=False):
        fifa_names = [
            fifa_row.player_name_fifa,
            f"{fifa_row.first_names_fifa} {fifa_row.last_name_fifa}",
            fifa_row.shirt_name_fifa,
        ]
        # FIFA -> Transfermarkt: data de nascimento exata + similaridade de nome.
        tm_candidates = tm_by_dob.get(fifa_row.date_of_birth, pd.DataFrame())
        tm_ranked: list[tuple[float, object]] = []
        for candidate in tm_candidates.itertuples(index=False):
            score = best_name_score(fifa_names, [candidate.player_name])
            club_bonus = 0.06 * name_score(fifa_row.club_at_world_cup, candidate.club_name)
            tm_ranked.append((min(1.0, score + club_bonus), candidate))
        tm_ranked.sort(key=lambda x: x[0], reverse=True)
        tm_score, tm_best = tm_ranked[0] if tm_ranked else (0.0, None)
        tm_second = tm_ranked[1][0] if len(tm_ranked) > 1 else 0.0
        tm_accepted = tm_best is not None and tm_score >= 0.68 and tm_score - tm_second >= 0.02

        # FIFA -> StatsBomb: seleção + número de camisa é a chave primária; nome valida.
        team_best = FIFA_TO_STATSBOMB_TEAM.get(
            fifa_row.country_name,
            max(sb_teams, key=lambda team: name_score(fifa_row.country_name, team)),
        )
        team_score = 1.0 if fifa_row.country_name in FIFA_TO_STATSBOMB_TEAM else name_score(fifa_row.country_name, team_best)
        sb_candidates = sb_unique[sb_unique["team_name"].eq(team_best)] if team_score >= 0.55 else sb_unique.iloc[0:0]
        jersey_candidates = sb_candidates[sb_candidates["shirt_number_statsbomb"].eq(fifa_row.shirt_number)]
        unique_jersey = len(jersey_candidates) == 1
        pool = jersey_candidates if unique_jersey else sb_candidates
        sb_ranked: list[tuple[float, object]] = []
        for candidate in pool.itertuples(index=False):
            score = best_name_score(fifa_names, [candidate.player_name_statsbomb])
            if candidate.shirt_number_statsbomb == fifa_row.shirt_number:
                score = min(1.0, score + 0.18)
            sb_ranked.append((score, candidate))
        sb_ranked.sort(key=lambda x: x[0], reverse=True)
        sb_score, sb_best = sb_ranked[0] if sb_ranked else (0.0, None)
        sb_second = sb_ranked[1][0] if len(sb_ranked) > 1 else 0.0
        sb_accepted = sb_best is not None and (
            (unique_jersey and sb_score >= 0.50)
            or (sb_score >= 0.72 and sb_score - sb_second >= 0.02)
        )

        record = {
            "fifa_squad_id": fifa_row.fifa_squad_id,
            "country_name": fifa_row.country_name,
            "player_name_fifa": fifa_row.player_name_fifa,
            "date_of_birth": fifa_row.date_of_birth,
            "club_at_world_cup": fifa_row.club_at_world_cup,
            "club_country_big5_candidate": fifa_row.club_country_big5_candidate,
            "big5_at_world_cup": tm_accepted,
            "transfermarkt_player_id": getattr(tm_best, "transfermarkt_player_id", pd.NA),
            "player_name_transfermarkt": getattr(tm_best, "player_name", ""),
            "tm_link_score": round(tm_score, 4),
            "tm_second_score": round(tm_second, 4),
            "tm_link_status": "accepted" if tm_accepted else ("review" if tm_best is not None else "unmatched"),
            "statsbomb_player_id": getattr(sb_best, "statsbomb_player_id", pd.NA) if sb_accepted else pd.NA,
            "player_name_statsbomb": getattr(sb_best, "player_name_statsbomb", "") if sb_accepted else "",
            "sb_candidate_player_id": getattr(sb_best, "statsbomb_player_id", pd.NA),
            "sb_candidate_player_name": getattr(sb_best, "player_name_statsbomb", ""),
            "sb_link_score": round(sb_score, 4),
            "sb_second_score": round(sb_second, 4),
            "sb_link_status": "accepted" if sb_accepted else ("review" if sb_best is not None else "unmatched"),
        }
        rows.append(record)
        if (fifa_row.club_country_big5_candidate and not tm_accepted) or not sb_accepted:
            reviews.append({"source": "fifa_master", **record})
    return pd.DataFrame(rows), pd.DataFrame(reviews)


def qa_summary(
    fifa: pd.DataFrame,
    big5_panel: pd.DataFrame,
    tm_games: pd.DataFrame,
    wc_pm: pd.DataFrame,
    wc_tournament: pd.DataFrame,
    understat_pm: pd.DataFrame,
    understat_games: pd.DataFrame,
    game_map: pd.DataFrame,
    fifa_crosswalk: pd.DataFrame,
    understat_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    accepted_sb = fifa_crosswalk["sb_link_status"].eq("accepted")
    team_match = wc_pm.groupby(["statsbomb_match_id", "team_name"], as_index=False).agg(
        starters=("started", "sum")
    )
    tests = [
        ("fifa_squad_rows", len(fifa), 831, len(fifa) == 831),
        ("fifa_teams", fifa["country_code"].nunique(), 32, fifa["country_code"].nunique() == 32),
        ("big5_games", len(tm_games), 1826, len(tm_games) == 1826),
        ("big5_panel_unique_rows", len(big5_panel), len(big5_panel.drop_duplicates(["game_id", "transfermarkt_player_id"])), not big5_panel.duplicated(["game_id", "transfermarkt_player_id"]).any()),
        ("wc_matches", wc_pm["statsbomb_match_id"].nunique(), 64, wc_pm["statsbomb_match_id"].nunique() == 64),
        ("wc_players", len(wc_tournament), "informational", True),
        ("wc_minutes_nonnegative", int(wc_pm["minutes_played"].ge(0).sum()), len(wc_pm), wc_pm["minutes_played"].ge(0).all()),
        ("wc_minutes_within_match", int(wc_pm["minutes_played"].le(wc_pm["match_duration_minutes"] + 0.001).sum()), len(wc_pm), wc_pm["minutes_played"].le(wc_pm["match_duration_minutes"] + 0.001).all()),
        ("wc_exactly_11_starters_per_team", int(team_match["starters"].eq(11).sum()), len(team_match), team_match["starters"].eq(11).all()),
        ("understat_games", len(understat_games), 1826, len(understat_games) == 1826),
        ("understat_shots", int(understat_pm["shots"].sum()), 45853, int(understat_pm["shots"].sum()) == 45853),
        ("understat_games_linked", int(game_map["match_status"].eq("accepted").sum()), 1826, game_map["match_status"].eq("accepted").all()),
        ("fifa_big5_linked_players", int(fifa_crosswalk["big5_at_world_cup"].sum()), "informational", True),
        ("fifa_statsbomb_accepted", int(fifa_crosswalk["sb_link_status"].eq("accepted").sum()), len(fifa_crosswalk), fifa_crosswalk["sb_link_status"].eq("accepted").mean() >= 0.95),
        ("fifa_statsbomb_operational_id_only_if_accepted", int(fifa_crosswalk.loc[~accepted_sb, "statsbomb_player_id"].isna().sum()), int((~accepted_sb).sum()), fifa_crosswalk.loc[~accepted_sb, "statsbomb_player_id"].isna().all()),
        ("fifa_statsbomb_accepted_id_one_to_one", int(fifa_crosswalk.loc[accepted_sb, "statsbomb_player_id"].nunique()), int(accepted_sb.sum()), fifa_crosswalk.loc[accepted_sb, "statsbomb_player_id"].notna().all() and not fifa_crosswalk.loc[accepted_sb, "statsbomb_player_id"].duplicated().any()),
        ("understat_tm_accepted", int(understat_crosswalk["link_status"].eq("accepted").sum()), len(understat_crosswalk), understat_crosswalk["link_status"].eq("accepted").mean() >= 0.90),
    ]
    return pd.DataFrame(tests, columns=["check", "observed", "expected_or_threshold", "passed"])


def write_interim_manifest() -> None:
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
                "generated_by": "scripts/etl/build_interim.py",
            }
        )
    pd.DataFrame(records).to_csv(OUT / "manifest.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote manifest.csv: {len(records):,} files")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fifa = extract_fifa_squads()
    conn = open_transfermarkt()
    big5_panel, tm_games = build_big5_panel(conn)
    wc_pm, wc_tournament = build_wc_player_match()
    understat_pm, understat_games = build_understat_player_match()
    game_map = match_understat_games(understat_games, tm_games)
    understat_crosswalk, review_understat = link_understat_players(understat_pm, game_map, big5_panel)
    fifa_crosswalk, review_fifa = link_fifa_players(fifa, wc_pm, big5_panel)
    reviews = pd.concat([review_fifa, review_understat], ignore_index=True, sort=False)
    qa = qa_summary(
        fifa, big5_panel, tm_games, wc_pm, wc_tournament, understat_pm,
        understat_games, game_map, fifa_crosswalk, understat_crosswalk,
    )

    write_csv(fifa, "fifa_squads_2022.csv")
    write_csv(big5_panel, "big5_player_match_2022_23.csv")
    write_csv(wc_pm, "wc_player_match_2022.csv")
    write_csv(wc_tournament, "wc_player_tournament_2022.csv")
    write_csv(understat_pm, "understat_player_match_2022_23.csv")
    write_csv(game_map, "understat_transfermarkt_game_crosswalk.csv")
    write_csv(understat_crosswalk, "understat_transfermarkt_player_crosswalk.csv")
    write_csv(fifa_crosswalk, "fifa_player_crosswalk.csv")
    write_csv(reviews, "linkage_review.csv")
    write_csv(qa, "qa_summary.csv")
    write_interim_manifest()
    if not qa["passed"].all():
        failed = qa.loc[~qa["passed"], "check"].tolist()
        raise RuntimeError(f"QA falhou: {failed}. Consulte data/interim/qa_summary.csv")
    print("ETL concluído com todos os controles de QA aprovados.")


if __name__ == "__main__":
    main()
