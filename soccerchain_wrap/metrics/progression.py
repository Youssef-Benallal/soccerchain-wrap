# metrics/progression.py
from __future__ import annotations

from typing import Iterable, Optional, Set
import numpy as np
import pandas as pd

from .sequence_summary import PitchDims

__all__ = [
    "central_progression_inv_cross_rate",
    "circulate_inv_progress_share",
    "field_tilt_final_third_pass_share",
    "progression_table_from_events",
    "add_percentiles",
]

# ------------------------------ config -------------------------------- #
_PASS_EVENTS: Set[str] = {"pass", "cross"}                 # attempted passes/crosses
_MOVE_EVENTS: Set[str] = {"pass", "cross", "dribble", "take_on"}  # ball-moving actions


# ------------------------------ helpers ------------------------------- #
def _prep_numeric_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce XY to numeric and add a lowercased type column."""
    d = df.copy()
    for c in ("start_x", "start_y", "end_x", "end_y"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t_lc"] = d.get("type_name", "").astype("string").str.lower()
    return d


# ------------------------------ metrics ------------------------------- #
def central_progression_inv_cross_rate(events: pd.DataFrame) -> pd.Series:
    """
    Central progression (higher = more central play):
        1 - (crosses / all passes), averaged per game.

    • all passes = type_name in {"pass", "cross"}
    • per-game ratio computed, then averaged across each team's games
    """
    d = _prep_numeric_xy(events)
    if not {"game_id", "team_id"}.issubset(d.columns):
        return pd.Series(dtype=float, name="central_progression")

    use = d[d["t_lc"].isin(_PASS_EVENTS)]
    if use.empty:
        return pd.Series(dtype=float, name="central_progression")

    by_gt = (
        use.assign(is_cross=use["t_lc"].eq("cross"))
        .groupby(["game_id", "team_id"])
        .agg(all_passes=("t_lc", "size"), crosses=("is_cross", "sum"))
        .reset_index()
    )
    by_gt["ratio"] = by_gt["crosses"] / by_gt["all_passes"].replace(0, np.nan)
    by_gt["central_progression"] = 1.0 - by_gt["ratio"]

    return (
        by_gt.groupby("team_id")["central_progression"].mean().fillna(0.0)
    )


def circulate_inv_progress_share(
    events: pd.DataFrame,
) -> pd.Series:
    """
    Circulate (higher = more circulation/less direct progression):
        1 - (progressive_distance / total_distance), averaged per game.

    • progressive_distance per event = max(0, end_x - start_x)
    • total_distance per event       = hypot(end_x - start_x, end_y - start_y)
    • events considered: pass, cross, dribble, take_on
    """
    d = _prep_numeric_xy(events)
    need = {"game_id", "team_id", "start_x", "start_y", "end_x", "end_y"}
    if not need.issubset(d.columns):
        return pd.Series(dtype=float, name="circulate")

    use = d[d["t_lc"].isin(_MOVE_EVENTS)].copy()
    if use.empty:
        return pd.Series(dtype=float, name="circulate")

    dx = use["end_x"] - use["start_x"]
    dy = use["end_y"] - use["start_y"]
    use["prog_dx"] = np.clip(dx, 0.0, None)                # forward-only
    use["seg_len"] = np.hypot(dx, dy)

    by_gt = use.groupby(["game_id", "team_id"]).agg(
        prog_dist=("prog_dx", "sum"),
        total_dist=("seg_len", "sum"),
    ).reset_index()

    by_gt["share"] = by_gt["prog_dist"] / by_gt["total_dist"].replace(0, np.nan)
    by_gt["circulate"] = 1.0 - by_gt["share"]

    return by_gt.groupby("team_id")["circulate"].mean().fillna(0.0)


def field_tilt_final_third_pass_share(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
) -> pd.Series:
    """
    Field tilt:
        share of a team's passes that originate in the final third,
        averaged per game.

    • final third threshold: start_x >= 2/3 * pitch length (default 70m on 105m)
    • passes considered: type_name in {"pass", "cross"}
    """
    d = _prep_numeric_xy(events)
    if not {"game_id", "team_id", "start_x"}.issubset(d.columns):
        return pd.Series(dtype=float, name="field_tilt")

    use = d[d["t_lc"].isin(_PASS_EVENTS)].copy()
    if use.empty:
        return pd.Series(dtype=float, name="field_tilt")

    x_thr = 2.0 * dims.length / 3.0
    use["in_final_third"] = use["start_x"] >= x_thr

    by_gt = use.groupby(["game_id", "team_id"]).agg(
        all_passes=("t_lc", "size"),
        final3=("in_final_third", "sum"),
    ).reset_index()

    by_gt["field_tilt"] = by_gt["final3"] / by_gt["all_passes"].replace(0, np.nan)

    return by_gt.groupby("team_id")["field_tilt"].mean().fillna(0.0)


# ----------------------- consolidated + percentiles ------------------- #
def progression_table_from_events(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    team_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compact team table with the three progression metrics (events-based).
    Returns: team_id (+ team_name if provided) and:
      - central_progression
      - circulate
      - field_tilt
    """
    central = central_progression_inv_cross_rate(events).rename("central_progression")
    circ = circulate_inv_progress_share(events).rename("circulate")
    tilt = field_tilt_final_third_pass_share(events, dims=dims).rename("field_tilt")

    out = pd.concat([central, circ, tilt], axis=1).reset_index()

    if team_lookup is not None and {"team_id", "team_name"}.issubset(team_lookup.columns):
        out = out.merge(
            team_lookup[["team_id", "team_name"]].drop_duplicates(),
            on="team_id",
            how="left",
        )
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
