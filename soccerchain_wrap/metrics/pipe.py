# soccerchain_wrap/pipelines/pipe.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import pandas as pd
import numpy as np

from soccerchain_wrap.metrics.sequence_summary import PitchDims
from soccerchain_wrap.metrics.possession import (
    possession_table_from_events as _pos_from_events,
)
from soccerchain_wrap.metrics.progression import (
    progression_table_from_events as _pro_from_events,
)
from soccerchain_wrap.metrics.defence import (
    defence_table_from_events as _def_from_events,
)

__all__ = [
    "metric_categories",
    "required_metric_keys",
    "build_league_table",
    "metric_percentile_table_by_team",
]


# --------------------------- categories & keys --------------------------- #
def metric_categories() -> Dict[str, List[str]]:
    """Category → metric keys used in the consolidated tables."""
    return {
        "Possession": ["sequence_share", "press_resistance", "deep_buildup"],
        "Progression": ["central_progression", "circulate", "field_tilt"],
        "Defence": ["intensity", "high_line", "chance_prevention"],
    }


def required_metric_keys() -> List[str]:
    """Flat list of all metric keys across categories."""
    keys: List[str] = []
    for vals in metric_categories().values():
        keys.extend(vals)
    return keys


# --------------------------- small helpers ------------------------------ #
def _add_percentiles(
    df: pd.DataFrame,
    metric_cols: Iterable[str],
    *,
    suffix: str = "_pct",
) -> pd.DataFrame:
    """Add 0–100 league percentiles (higher = better). Constant → 50.0."""
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        if out[col].nunique(dropna=True) <= 1:
            out[f"{col}{suffix}"] = 50.0
        else:
            out[f"{col}{suffix}"] = 100.0 * out[col].rank(pct=True, method="average")
    return out


# --------------------------- public API --------------------------------- #
def build_league_table(
    events_open: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    team_id_col: str = "team_id",
    team_name_col: str = "team_name",
) -> pd.DataFrame:
    """
    Build a consolidated RAW metrics table by team (events-based).

    Returns
    -------
    pd.DataFrame with:
      team_id, team_name +
      Possession: sequence_share, press_resistance, deep_buildup
      Progression: central_progression, circulate, field_tilt
      Defence    : intensity, high_line, chance_prevention
    """
    # team lookup
    if team_name_col in events_open.columns:
        teams = events_open[[team_id_col, team_name_col]].drop_duplicates()
    else:
        teams = pd.DataFrame({team_id_col: events_open[team_id_col].drop_duplicates()})
        teams[team_name_col] = teams[team_id_col].astype(str)

    teams_std = teams.rename(
        columns={team_id_col: "team_id", team_name_col: "team_name"}
    )

    pos = _pos_from_events(events_open, dims=dims, team_lookup=teams_std, wheel_alias=True)
    pro = _pro_from_events(events_open, dims=dims, team_lookup=teams_std)
    deff = _def_from_events(events_open, dims=dims, team_lookup=teams_std)

    league = (
        pos.merge(pro, on=["team_id", "team_name"], how="outer")
           .merge(deff, on=["team_id", "team_name"], how="outer")
           .fillna(0.0)
    )

    cols = ["team_id", "team_name"] + required_metric_keys()
    return league[[c for c in cols if c in league.columns]].copy()


def metric_percentile_table_by_team(
    events_open: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    team_id_col: str = "team_id",
    team_name_col: str = "team_name",
    pct_suffix: str = "_pct",
    include_raw: bool = True,
    sort_by: Optional[str] = "team_name",
) -> pd.DataFrame:
    """
    Build a league-wide percentile table by team (0..100) from events.
    Includes Possession, Progression, and Defence metrics.
    """
    league_raw = build_league_table(
        events_open, dims=dims, team_id_col=team_id_col, team_name_col=team_name_col
    )

    keys = required_metric_keys()
    league_pct = _add_percentiles(league_raw, keys, suffix=pct_suffix)

    if not include_raw:
        keep = ["team_id", "team_name"] + [f"{m}{pct_suffix}" for m in keys]
        league_pct = league_pct[[c for c in keep if c in league_pct.columns]]

    if sort_by and sort_by in league_pct.columns:
        league_pct = league_pct.sort_values(sort_by)

    num_cols = league_pct.select_dtypes(include=[np.number], exclude=['bool']).columns
    league_pct[num_cols] = league_pct[num_cols].round().astype("Int64")

    return league_pct.reset_index(drop=True)
