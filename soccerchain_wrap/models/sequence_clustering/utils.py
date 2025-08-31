# sequence_paths/utils.py
import pandas as pd
import numpy as np

def trim_sequence(
    df: pd.DataFrame,
    seq_col: str = "sequence_id",
    startx: str = "start_x",
    starty: str = "start_y",
    endx: str = "end_x",
    endy: str = "end_y",
    threshold: float = 80.0,
) -> pd.DataFrame:
    """Cut each sequence at the first event with end_x ≥ threshold.
    If that event starts before the threshold, set end_x = threshold
    and end_y is interpolated along the segment.
    
    Important: this utility is important for buildup sequences.
    """
    if df.empty or seq_col not in df.columns:
        return df.copy()

    d = df.copy()

    parts = []
    for _, g in d.groupby(seq_col, sort=False):
        cross = pd.to_numeric(g[endx], errors="coerce").ge(threshold)
        if not cross.any():                  # no entry → keep as is
            parts.append(g); continue

        idx = cross.idxmax()                 # first row with end_x ≥ threshold
        pre = g.loc[g.index < idx]           # keep rows strictly before

        r = g.loc[idx].copy()
        sx, sy = float(r[startx]), float(r[starty])
        ex, ey = float(r[endx]),  float(r[endy])

        if sx < threshold:                   # clip this row to x = threshold
            t = 0.0 if ex == sx else (threshold - sx) / (ex - sx)
            r[endx] = threshold
            r[endy] = sy + t * (ey - sy)
            parts.append(pd.concat([pre, r.to_frame().T]))
        else:                                # already inside final third → drop this row too
            parts.append(pre)

    return pd.concat(parts, ignore_index=True)

# ---------------------- CLUSTERED SEQUENCES METRICS HELPERS ----------------------
def _to_float_array(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

def seq_directness(ev_df: pd.DataFrame) -> float:
    sx = _to_float_array(ev_df["start_x"])
    ex = _to_float_array(ev_df["end_x"])
    dx_seg = ex - sx
    dy_seg = _to_float_array(ev_df["end_y"]) - _to_float_array(ev_df["start_y"])
    seg_len = np.hypot(dx_seg, dy_seg)
    L = float(np.nansum(seg_len))
    dx_net = float((ex[-1] - sx[0]) if len(ex) > 0 else 0.0)
    if not np.isfinite(L) or L <= 0:
        return 0.0
    return float(np.clip(dx_net / L, 0.0, 1.0))

def seq_width_index(points: np.ndarray, pitch_width: float = 68.0) -> float:
    if points is None or len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    spread = float(np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1]))
    return float(np.clip(spread / pitch_width, 0.0, 1.0))
