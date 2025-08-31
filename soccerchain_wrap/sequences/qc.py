# soccerchain_wrap/actions/sequences/qc.py
from __future__ import annotations
from typing import Literal, Iterable
import numpy as np
import pandas as pd


# -------- gap diagnostics (sequence-agnostic) -------- #
def add_sequence_gap_flag(
    df: pd.DataFrame,
    *,
    dx_threshold: float = 20.0,
    dy_threshold: float = 13.0,
) -> pd.DataFrame:
    """
    Compute per-sequence spatial gaps and flag suspicious sequences.

      dx_gap = | next.start_x - this.end_x |
      dy_gap = | next.start_y - this.end_y |

    Adds (constant within sequence):
      - seq_max_dx_gap, seq_max_dy_gap
      - seq_gap_dx_thresh, seq_gap_dy_thresh
      - sequence_gap (True if dx or dy exceeds threshold)
    """
    d = df.copy()
    if {"in_sequence", "sequence_valid"}.issubset(d.columns):
        d = d[(d["in_sequence"] == True) & (d["sequence_valid"] == True)]  # noqa: E712
    d = d.dropna(subset=["sequence_id"]).sort_values(
        ["sequence_id", "time_seconds"], kind="mergesort"
    )

    g = d.groupby("sequence_id", sort=False)
    dx_gap = (g["start_x"].shift(-1) - d["end_x"]).abs()
    dy_gap = (g["start_y"].shift(-1) - d["end_y"]).abs()

    per_seq = pd.DataFrame(
        {
            "seq_max_dx_gap": dx_gap.groupby(d["sequence_id"], sort=False).max(),
            "seq_max_dy_gap": dy_gap.groupby(d["sequence_id"], sort=False).max(),
        }
    ).reset_index()

    per_seq["seq_gap_dx_thresh"] = float(dx_threshold)
    per_seq["seq_gap_dy_thresh"] = float(dy_threshold)
    per_seq["sequence_gap"] = (
        (per_seq["seq_max_dx_gap"] >= dx_threshold)
        | (per_seq["seq_max_dy_gap"] >= dy_threshold)
    )

    out = df.merge(per_seq, on="sequence_id", how="left")
    out["sequence_gap"] = out["sequence_gap"].fillna(False)
    return out


def interpolate_small_gaps_with_carries(
    df: pd.DataFrame,
    *,
    min_gap: float = 1.0,
    operate_on: Literal["clean_sequences", "flagged", "all"] = "clean_sequences",
) -> pd.DataFrame:
    """
    Insert synthetic 'dribble' rows between consecutive events inside sequences.

    Operate on:
      - "clean"  : sequence_gap == False   (default, safest)
      - "flagged": sequence_gap == True
      - "all"    : ignore the sequence_gap flag

    Insert when (i -> i+1):
      1) |sx_{i+1} - ex_i| > min_gap  OR  |sy_{i+1} - ey_i| > min_gap
      2) |sx_{i+1} - ex_i| < seq_gap_dx_thresh  AND
         |sy_{i+1} - ey_i| < seq_gap_dy_thresh
    """
    if "sequence_id" not in df.columns:
        return df

    out = df.copy()
    if "is_synthetic_carry" not in out.columns:
        out["is_synthetic_carry"] = False

    # choose target sequences
    default_flag = pd.Series(False, index=out.index)
    flag = out.get("sequence_gap", default_flag).fillna(False)
    if operate_on == "clean":
        seq_mask = ~flag
    elif operate_on == "flagged":
        seq_mask = flag
    else:  # "all"
        seq_mask = pd.Series(True, index=out.index)

    def _bool_col(name: str, default: bool) -> pd.Series:
        return out[name].astype(bool) if name in out else pd.Series(default, index=out.index)

    mseq = (
        out["sequence_id"].notna()
        & seq_mask
        & _bool_col("in_sequence", True)
        & _bool_col("sequence_valid", True)
    )

    d = out.loc[mseq].sort_values(["sequence_id", "time_seconds"], kind="mergesort")
    if d.empty:
        return out

    g = d.groupby("sequence_id", sort=False)
    d["nx_sx"] = g["start_x"].shift(-1)
    d["nx_sy"] = g["start_y"].shift(-1)
    d["nx_t"] = g["time_seconds"].shift(-1)

    for col in (
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "jersey_number",
        "period_id",
        "game_id",
    ):
        if col in d.columns:
            d[f"nx_{col}"] = g[col].shift(-1)

    coords_ok = d[["end_x", "end_y", "nx_sx", "nx_sy"]].notna().all(axis=1)
    dx = (d["nx_sx"] - d["end_x"]).abs()
    dy = (d["nx_sy"] - d["end_y"]).abs()
    has_gap = (dx > float(min_gap)) | (dy > float(min_gap))

    if {"seq_gap_dx_thresh", "seq_gap_dy_thresh"}.issubset(d.columns):
        below_thr = (dx < d["seq_gap_dx_thresh"]) & (dy < d["seq_gap_dy_thresh"])
    else:
        # Without thresholds we don't add carries (safer default)
        below_thr = pd.Series(False, index=d.index)

    keep = coords_ok & has_gap & below_thr
    if not keep.any():
        return out

    idx = d.index[keep]
    t0 = d.loc[idx, "time_seconds"].to_numpy()
    t1 = d.loc[idx, "nx_t"].to_numpy()
    mid_t = (t0 + t1) / 2.0  # type: ignore
    mid_t = np.where(~np.isfinite(mid_t) | (mid_t == t0), t0 + 1e-3, mid_t)

    new = {
        "sequence_id": d.loc[idx, "sequence_id"].to_numpy(),
        "game_id": d.loc[idx, "nx_game_id"].to_numpy() if "nx_game_id" in d else np.nan,
        "period_id": d.loc[idx, "nx_period_id"].to_numpy() if "nx_period_id" in d else np.nan,
        "team_id": d.loc[idx, "nx_team_id"].to_numpy() if "nx_team_id" in d else np.nan,
        "team_name": d.loc[idx, "nx_team_name"].to_numpy() if "nx_team_name" in d else np.nan,
        "player_id": d.loc[idx, "nx_player_id"].to_numpy() if "nx_player_id" in d else np.nan,
        "player_name": d.loc[idx, "nx_player_name"].to_numpy() if "nx_player_name" in d else np.nan,
        "jersey_number": d.loc[idx, "nx_jersey_number"].to_numpy() if "nx_jersey_number" in d else np.nan,
        "type_name": np.repeat("dribble", len(idx)),
        "start_x": d.loc[idx, "end_x"].to_numpy(),
        "start_y": d.loc[idx, "end_y"].to_numpy(),
        "end_x": d.loc[idx, "nx_sx"].to_numpy(),
        "end_y": d.loc[idx, "nx_sy"].to_numpy(),
        "time_seconds": mid_t,
        "in_sequence": True,
        "sequence_valid": True,
        "is_synthetic_carry": True,
    }
    new_df = pd.DataFrame(new).reindex(columns=list(out.columns), fill_value=np.nan)

    return (
        pd.concat([out, new_df], ignore_index=True)
        .sort_values(["sequence_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )


# ---------------------------- QC layer ----------------------------- #
def qc_layer(
    df: pd.DataFrame,
    *,
    steps: Iterable[str] = ("gap_flag", "interpolate"),
    # gap flag params
    dx_gap_thresh: float = 20.0,
    dy_gap_thresh: float = 13.0,
    # interpolate params
    interpolate_min_gap: float = 1.0,
    interpolate_operate_on: Literal["clean", "flagged", "all"] = "clean",
) -> pd.DataFrame:
    """
    Apply one or more QC steps in sequence to a detected dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains sequence annotations
        (sequence_id, in_sequence, sequence_valid).
    steps : Iterable[str]
        Ordered list of steps to execute. Supported:
          - "gap_flag"     : run add_sequence_gap_flag()
          - "interpolate"  : run interpolate_small_gaps_with_carries()
        Default order is ("gap_flag", "interpolate").
    dx_gap_thresh, dy_gap_thresh : float
        Thresholds used by gap flagging.
    interpolate_min_gap : float
        Minimum spatial gap to consider inserting a carry.
    interpolate_operate_on : {"clean","flagged","all"}
        Which sequences to target when inserting carries.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with QC outputs merged (and possibly new carry rows).
    """
    out = df.copy()
    for step in steps:
        s = step.lower().strip()
        if s == "gap_flag":
            out = add_sequence_gap_flag(
                out, dx_threshold=dx_gap_thresh, dy_threshold=dy_gap_thresh
            )
        elif s == "interpolate":
            out = interpolate_small_gaps_with_carries(
                out,
                min_gap=interpolate_min_gap,
                operate_on=interpolate_operate_on,  # type: ignore
            )
        else:
            raise ValueError(
                f"qc_layer: unknown step '{step}'. Supported: 'gap_flag', 'interpolate'."
            )
    return out
