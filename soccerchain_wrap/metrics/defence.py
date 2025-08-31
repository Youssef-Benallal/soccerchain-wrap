# metrics/defence.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .sequence_summary import PitchDims

__all__ = [
    "defensive_intensity",
    "high_line_sweeper_rate",
    "chance_prevention_proxy",
    "defence_table_from_events",
    "add_percentiles",
]

# ------------------------------ config -------------------------------- #
# On-ball touches we allow the opponent to make
_TOUCH_EVENTS: Set[str] = {"pass"}
#"cross", "dribble", "take_on", "shot"

# GK “sweeper” actions OUTSIDE the PA (keeper cannot handle with hands there)
# We restrict to GK role and these ball-winning/relief actions:
_GK_SWEEPER_EVENTS: Set[str] = {"interception", "tackle", "clearance", "pass"}

_GK_ROLES: Tuple[str, ...] = ("GK", "Goalkeeper")


# ------------------------------ helpers ------------------------------- #
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase types/results; coerce XY/time to numeric.
    Adds:
      - t_lc  : lowercase type_name
      - res_lc: lowercase result_name
    """
    d = df.copy()
    d["t_lc"] = d.get("type_name", "").astype("string").str.lower()
    d["res_lc"] = d.get("result_name", "").astype("string").str.lower()
    for c in ("start_x", "start_y", "end_x", "end_y", "time_seconds"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _opponent_map(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (game_id, team_id) → opp_id mapping and merge it.
    Assumes standard 2-team matches; other games are ignored.
    """
    pairs: List[Dict[str, int]] = []
    for gid, teams in events.groupby("game_id")["team_id"].unique().items():  # type: ignore
        if len(teams) != 2:
            continue
        a, b = map(int, teams)
        pairs.append({"game_id": gid, "team_id": a, "opp_id": b})
        pairs.append({"game_id": gid, "team_id": b, "opp_id": a})
    opp_df = pd.DataFrame(pairs)
    return events.merge(opp_df, on=["game_id", "team_id"], how="left")


def _game_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Per-game match length in minutes from max(time_seconds).
    Fallback = 90 mins when time is missing.
    """
    if "time_seconds" not in events.columns:
        mins = events.groupby("game_id").size().rename("dummy").reset_index()
        mins["game_minutes"] = 90.0
        return mins[["game_id", "game_minutes"]]

    g = events.groupby("game_id")["time_seconds"].max().rename("max_t").reset_index()
    g["game_minutes"] = g["max_t"].astype(float) / 60.0
    g["game_minutes"] = g["game_minutes"].replace([np.inf, -np.inf], np.nan).fillna(90.0)
    return g[["game_id", "game_minutes"]]


# ------------------------------ metrics ------------------------------- #
def defensive_intensity(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    last_m: float = 40.0,
) -> pd.DataFrame:
    """
    Defensive INTENSITY (PPDA-style, opponent passes per our defensive actions).

    ORIENTATION: absolute L→R for both teams.

    ZONES (different by design):
      • Opponent touches zone   : opponent's FIRST `last_m` meters → start_x <= last_m
      • Our defensive actions   : our LAST  `last_m` meters        → start_x >= (L - last_m)

    Per (game, team):
      opp_touches  = opponent *passes* in [0, last_m]
      def_actions  = team's (tackles + interceptions) in [L - last_m, L]
      ratio        = opp_touches / def_actions
      intensity    = 100 * def_actions / opp_touches

    Returns
    -------
    pd.DataFrame
        columns: team_id, intensity_raw, intensity
    """
    L = float(dims.length)

    d = _prep(events)
    d = _opponent_map(d)

    # ---------- opponent touches in their first `last_m` meters ----------
    opp_touch = (
        d[(d["t_lc"].isin(_TOUCH_EVENTS)) & (d["start_x"] <= (last_m)) & (d['res_lc']=='success')]
        .groupby(["game_id", "opp_id"])
        .size()
        .rename("opp_touches")
        .reset_index()
        .rename(columns={"opp_id": "team_id"})
        .set_index(["game_id", "team_id"])
    )


    # ---------- our defensive actions in our last `last_m` meters ----------
    team_def = (
        d[(d["t_lc"].isin({"tackle", "interception", "foul"})) & (d["start_x"] >= (L-last_m))]
        .groupby(["game_id", "team_id"])
        .size()
        .rename("def_actions")
        .reset_index()
        .set_index(["game_id", "team_id"])
    )
    per_gt = opp_touch.join(team_def, how="outer").fillna(0).reset_index()
    per_gt["intensity"] = np.where(
        per_gt["opp_touches"] > 0, 100.0 * per_gt["def_actions"] / per_gt["opp_touches"], np.nan
    )

    out = per_gt.reset_index().groupby("team_id").agg(
        intensity=("intensity", "mean"),
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index()



def high_line_sweeper_rate(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
) -> pd.Series:
    """
    HIGH LINE metric (absolute LTR orientation):

        (GK sweeper OUTSIDE PA + opponent OFFSIDES)
        per (opponent PASSES INTO FINAL THIRD), averaged per game.

    • GK sweeper OUTSIDE PA:
        player starting_position in {"GK","Goalkeeper"} AND
        type_name in {"interception","tackle","clearance","pass"} AND
        min(start_x, L - start_x) > box_length  (outside both penalty areas)
    • Offsides: any row where result_name contains 'offside' (case-insensitive).
    • Denominator: opponent passes/crosses with end_x >= 2/3 * pitch length.
    """
    d = _prep(events)
    d = _opponent_map(d)
    d["role"] = d.get("starting_position", "").astype("string")

    # Outside BOTH penalty areas (orientation-agnostic)
    dist_near_goal = np.minimum(d["start_x"], dims.length - d["start_x"])
    is_outside_pa = dist_near_goal > float(dims.box_length)

    # Team GK sweeper actions outside PA
    sweeper = d[
        is_outside_pa
        & d["role"].isin(_GK_ROLES)
        & d["t_lc"].isin(_GK_SWEEPER_EVENTS)
    ]
    gk_cnt = (
        sweeper.groupby(["game_id", "team_id"])
        .size()
        .rename("gk_sweeper")
        .reset_index()
        .set_index(["game_id", "team_id"])
    )

    # Opponent offsides (result_name contains 'offside')
    is_offside = d["res_lc"].str.contains("offside", na=False)
    opp_off = (
        d[is_offside]
        .groupby(["game_id", "opp_id"])
        .size()
        .rename("opp_offsides")
        .reset_index()
        .rename(columns={"opp_id": "team_id"})
        .set_index(["game_id", "team_id"])
    )

    # Opponent passes into final third (denominator)
    x_ft = 2.0 * dims.length / 3.0
    opp_ft = (
        d[(d["t_lc"].isin({"pass", "cross"})) & (d["end_x"] >= x_ft)]
        .groupby(["game_id", "opp_id"])
        .size()
        .rename("opp_final3_passes")
        .reset_index()
        .rename(columns={"opp_id": "team_id"})
        .set_index(["game_id", "team_id"])
    )

    per_gt = gk_cnt.join(opp_off, how="outer").join(opp_ft, how="outer").fillna(0)
    per_gt["numer"] = per_gt["gk_sweeper"] + per_gt["opp_offsides"]
    per_gt["rate"] = np.where(
        per_gt["opp_final3_passes"] > 0,
        per_gt["numer"] / per_gt["opp_final3_passes"],
        np.nan,
    )

    return (
        per_gt.reset_index()
        .groupby("team_id")["rate"]
        .mean()
        .fillna(0.0)
        .rename("high_line")
    )


def chance_prevention_proxy(events: pd.DataFrame) -> pd.DataFrame:
    """
    CHANCE PREVENTION proxy (xG not available):
      - npsa_per90 = non-penalty shots against per 90
          (shots against = opponent events of type 'shot' or 'shot_freekick')
      - chance_prevention = 1 / (1 + npsa_per90)    (higher = better)

    Returns: team_id, npsa_per90, chance_prevention
    """
    d = _prep(events)
    d = _opponent_map(d)

    # Shots against per (game, team)
    npsa = (
        d[d["t_lc"].isin({"shot", "shot_freekick"})]
        .groupby(["game_id", "opp_id"])
        .size()
        .rename("shots_against")
        .reset_index()
        .rename(columns={"opp_id": "team_id"})
    )

    # Minutes per game
    gmins = _game_minutes(d)

    per_gt = npsa.merge(gmins, on="game_id", how="left")
    per_gt["per90"] = per_gt["shots_against"] / (
        per_gt["game_minutes"] / 90.0
    ).replace(0, np.nan)

    by_team = (
        per_gt.groupby("team_id")["per90"]
        .mean()
        .fillna(0.0)
        .rename("npsa_per90")
        .reset_index()
    )
    by_team["chance_prevention"] = 1.0 / (1.0 + by_team["npsa_per90"])
    return by_team


# ----------------------- consolidated + percentiles ------------------- #
def defence_table_from_events(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    team_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compact team table with defensive metrics (absolute LTR orientation):

      - intensity         : tackles per 100 opp touches in HIGH-X zone (avg per game)
      - high_line         : (GK-sweeper(outside PA) + opp offsides) / opp final-3rd passes
                            (avg per game)
      - chance_prevention : inverse of non-penalty shots against per 90

    Also returns npsa_per90 for transparency.
    """
    inten = defensive_intensity(events, dims=PitchDims(), last_m=70.0)[["team_id", "intensity"]]
    highl = high_line_sweeper_rate(events, dims=dims).reset_index()
    chprev = chance_prevention_proxy(events)

    out = (
        inten.merge(highl, on="team_id", how="outer")
        .merge(chprev, on="team_id", how="outer")
        .fillna(0.0)
    )

    if team_lookup is not None and {"team_id", "team_name"}.issubset(team_lookup.columns):
        out = out.merge(
            team_lookup[["team_id", "team_name"]].drop_duplicates(),
            on="team_id",
            how="left",
        )

    cols = ["team_id", "team_name", "intensity", "high_line", "chance_prevention", "npsa_per90"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].copy()


def add_percentiles(df: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    """
    Add league percentiles (0–100; higher = better) for given metric columns.
    Constant columns → 50 for everyone.
    """
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        if out[col].nunique(dropna=True) <= 1:
            out[col + "_pct"] = 50.0
        else:
            out[col + "_pct"] = 100.0 * out[col].rank(pct=True, method="average")
    return out
