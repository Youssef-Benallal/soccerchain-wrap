# metrics/possession.py
from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .sequence_summary import PitchDims

__all__ = [
    "pass_share",
    "press_resistance_touch_ratio",
    "deep_buildup_inv_launch",
    "possession_table_from_events",
    "add_percentiles",
]

# ------------------------------ config -------------------------------- #
_PASS_EVENTS: Set[str] = {"pass", "cross"}          # attempted passes
_TOUCH_EVENTS: Set[str] = {"pass", "cross", "dribble", "take_on", "shot"}
_GK_ROLES: Tuple[str, ...] = ("GK", "Goalkeeper")   # role label(s) for keepers


# ------------------------------ helpers ------------------------------- #
def _lc(s) -> str:
    return str(s).strip().lower()


def _prep_numeric_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce XY columns to numeric; keep a lowercase type column."""
    d = df.copy()
    for c in ("start_x", "start_y", "end_x", "end_y"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "type_name" in d.columns:
        d["t_lc"] = d["type_name"].astype("string").str.lower()
    else:
        d["t_lc"] = ""
    return d


# ------------------------------ metrics ------------------------------- #
def pass_share(events: pd.DataFrame) -> pd.Series:
    """
    Possession proxy: share of passes attempted (avg per game).

    pass_share_team_game = team_passes / all_passes_in_game
    final metric = mean over that team's games

    Uses type_name in {"pass", "cross"}.
    """
    d = _prep_numeric_xy(events)
    passes = d[d["t_lc"].isin(_PASS_EVENTS)]

    if passes.empty:
        return pd.Series(dtype=float, name="pass_share")

    by_gt = (
        passes.groupby(["game_id", "team_id"])
        .size()
        .rename("team_passes")
        .reset_index()
    )
    by_gt["game_passes"] = by_gt.groupby("game_id")["team_passes"].transform("sum")
    by_gt["pass_share"] = by_gt["team_passes"] / by_gt["game_passes"].replace(0, np.nan)

    return by_gt.groupby("team_id")["pass_share"].mean().fillna(0.0)


def press_resistance_touch_ratio(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    zone_max_x: Optional[float] = None,
) -> pd.Series:
    """
    Press resistance (higher is better):
        touches_per_opponent_tackle in the first two-thirds of the pitch.

    For each game:
      touches_team = count of on-ball events (pass, cross, dribble, take_on, shot)
                     with start_x <= zone_max_x (default: 2/3 * pitch length)
      opp_tackles  = (total tackles in game & zone) - (team tackles in zone)
      ratio        = touches_team / opp_tackles   (NaN if opp_tackles == 0)

    Final metric = mean ratio across games for that team.
    """
    x_max = float(zone_max_x) if zone_max_x is not None else (2.0 * dims.length / 3.0)

    d = _prep_numeric_xy(events)
    if not {"game_id", "team_id", "start_x"}.issubset(d.columns):
        return pd.Series(dtype=float, name="press_resistance")

    in_zone = d["start_x"] <= x_max

    # touches by team/game in zone
    touches = (
        d[in_zone & d["t_lc"].isin(_TOUCH_EVENTS)]
        .groupby(["game_id", "team_id"])
        .size()
        .rename("touches")
        .reset_index()
    )

    # tackles by team/game in zone
    tackles = (
        d[in_zone & d["t_lc"].eq("tackle")]
        .groupby(["game_id", "team_id"])
        .size()
        .rename("tackles_team")
        .reset_index()
    )

    if touches.empty and tackles.empty:
        return pd.Series(dtype=float, name="press_resistance")

    # total tackles per game in zone
    tot_tackles = tackles.groupby("game_id")["tackles_team"].sum().rename("tackles_game")

    g = touches.merge(tackles, on=["game_id", "team_id"], how="left").fillna(0)
    g = g.merge(tot_tackles, on="game_id", how="left").fillna(0)
    g["opp_tackles"] = (g["tackles_game"] - g["tackles_team"]).clip(lower=0)

    # touches per opponent tackle (NaN when opp_tackles == 0)
    g["ratio"] = np.where(g["opp_tackles"] > 0, g["touches"] / g["opp_tackles"], np.nan)

    return g.groupby("team_id")["ratio"].mean().fillna(0.0).rename("press_resistance")


def deep_buildup_inv_launch(
    events: pd.DataFrame,
    *,
    launch_threshold_m: float = 36.6,  # ≈ 40 yards
) -> pd.Series:
    """
    Deep build-up = 1 - GK launch rate.

    GK distributions considered:
      • 'goalkick'
      • any 'pass' from a player with starting_position in {"GK","Goalkeeper"}

    A 'launch' is a distribution with path length >= launch_threshold_m.
    Metric is computed per team across all games (pooled); higher = more short play.
    """
    need = {"type_name", "team_id", "start_x", "start_y", "end_x", "end_y", "starting_position"}
    if not need.issubset(events.columns):
        return pd.Series(dtype=float, name="deep_buildup")

    d = _prep_numeric_xy(events)
    d["role"] = events["starting_position"].astype("string")

    is_gk_pass = d["t_lc"].eq("pass") & d["role"].isin(_GK_ROLES)
    is_goal_kick = d["t_lc"].eq("goalkick")
    gk = d[is_gk_pass | is_goal_kick].copy()

    if gk.empty:
        return pd.Series(dtype=float, name="deep_buildup")

    # distance
    gk["dist"] = np.hypot(gk["end_x"] - gk["start_x"], gk["end_y"] - gk["start_y"])

    per_team = gk.groupby("team_id")
    total = per_team.size().rename("gk_total")
    launched = per_team.apply(lambda s: (s["dist"] >= float(launch_threshold_m)).sum()).rename(
        "gk_launched"
    )

    out = pd.concat([total, launched], axis=1).fillna(0.0)
    out["launch_rate"] = out["gk_launched"] / out["gk_total"].replace(0, np.nan)
    out["deep_buildup"] = 1.0 - out["launch_rate"]

    return out["deep_buildup"].fillna(0.0)


# ----------------------- consolidated + percentiles ------------------- #
def possession_table_from_events(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    team_lookup: pd.DataFrame | None = None,
    wheel_alias: bool = True,
) -> pd.DataFrame:
    """
    Compact team table from raw events using the requested definitions.

    Returns columns:
      - pass_share
      - press_resistance_touch_ratio
      - deep_buildup_inv_launch

    If `wheel_alias=True`, duplicates with wheel-friendly names:
      - sequence_share      <- pass_share
      - press_resistance    <- press_resistance_touch_ratio
      - deep_buildup        <- deep_buildup_inv_launch
    """
    poss = pass_share(events).rename("pass_share")
    press = press_resistance_touch_ratio(events, dims=dims).rename(
        "press_resistance_touch_ratio"
    )
    deep = deep_buildup_inv_launch(events).rename("deep_buildup_inv_launch")

    out = pd.concat([poss, press, deep], axis=1).reset_index()

    if team_lookup is not None and {"team_id", "team_name"}.issubset(team_lookup.columns):
        out = out.merge(
            team_lookup[["team_id", "team_name"]].drop_duplicates(),
            on="team_id",
            how="left",
        )

    if wheel_alias:
        out["sequence_share"] = out["pass_share"]
        out["press_resistance"] = out["press_resistance_touch_ratio"]
        out["deep_buildup"] = out["deep_buildup_inv_launch"]

    return out.fillna(0.0)


def add_percentiles(df: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    """
    Add league percentiles (0–100; higher = better) for given columns.
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
