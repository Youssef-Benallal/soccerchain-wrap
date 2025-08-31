# # metrics/sequence_summary.py
# from __future__ import annotations

# from dataclasses import dataclass
# import numpy as np
# import pandas as pd

# __all__ = ["PitchDims", "summarize_open_play_sequences"]


# @dataclass(frozen=True)
# class PitchDims:
#     """
#     Pitch geometry in meters (default: 105 x 68). 
#     Penalty Area (PA) assumed 40.32 x 16.5 centered on pitch width.
#     """
#     length: float = 105.0
#     width: float = 68.0
#     box_width: float = 40.32
#     box_length: float = 16.5

#     @property
#     def halfway_x(self) -> float:
#         return self.length / 2.0

#     @property
#     def box_x_min(self) -> float:
#         # x-coordinate of penalty area start (attacking from left to right)
#         return self.length - self.box_length

#     @property
#     def box_half_width(self) -> float:
#         return self.box_width / 2.0

#     def in_box(self, x: float, y: float) -> bool:
#         """True if (x,y) is inside the attacking penalty area (using end point)."""
#         y0 = self.width / 2.0
#         return (x >= self.box_x_min) and (abs(y - y0) <= self.box_half_width)
    
# def summarize_open_play_sequences(
#     events: pd.DataFrame,
#     *,
#     dims: PitchDims = PitchDims(),
#     long_ball_min_len: float = 30.0,
#     cross_requires_pass: bool = False,  # <-- new: set True if you want crosses ⊂ passes
# ) -> pd.DataFrame:
#     """
#     One row per *open-play* sequence with geometry, timing, and shape metrics.
#     (Preserves `team_name` if present.)

#     Cross detection:
#       - is_cross / pass_cross [bool], or any text column containing "cross"
#       - By default does NOT require pass flag (set cross_requires_pass=True to enforce)

#     Long-ball detection:
#       - is_long_ball / pass_long_ball [bool]
#       - OR any text column contains "long ball"  (counts even if not a pass)
#       - OR (pass_length >= long_ball_min_len)   (requires pass)
#       - Fallback: seg_len >= long_ball_min_len on rows flagged as passes
#     """
#     has_team_name = "team_name" in events.columns

#     def _empty_df():
#         cols = [
#             "sequence_id", "game_id", "team_id",
#             "start_x", "start_y", "end_x", "end_y",
#             "t_start", "t_end", "duration", "dx",
#             "path_len", "med_y", "max_x",
#             "forward", "reach_half", "box_entry",
#             "directness", "width_index",
#             "n_passes", "n_crosses", "n_long_balls",
#         ]
#         if has_team_name:
#             cols.insert(cols.index("team_id") + 1, "team_name")
#         return pd.DataFrame(columns=cols)

#     if events.empty:
#         return _empty_df()

#     in_seq = events["in_sequence"].astype(bool) if "in_sequence" in events else pd.Series(True, index=events.index)
#     valid = events["sequence_valid"].astype(bool) if "sequence_valid" in events else pd.Series(True, index=events.index)

#     data = events.loc[in_seq & valid].copy()
#     if data.empty:
#         return _empty_df()

#     for c in ("start_x", "start_y", "end_x", "end_y", "time_seconds"):
#         data[c] = pd.to_numeric(data[c], errors="coerce")

#     dx_ev = data["end_x"] - data["start_x"]
#     dy_ev = data["end_y"] - data["start_y"]
#     data["seg_len"] = np.hypot(dx_ev, dy_ev)

#     # --- Flexible event flags ---
#     # Pass
#     if "is_pass" in data:
#         pass_flag = data["is_pass"].astype(bool)
#     else:
#         pass_flag = pd.Series(False, index=data.index)
#         for col in ("event_type", "type", "type_name", "event", "action", "sub_type", "subtype", "pass_type"):
#             if col in data:
#                 pass_flag |= data[col].astype(str).str.lower().str.contains(r"\bpass\b", na=False)
#         if "pass_length" in data:
#             pass_flag |= data["pass_length"].notna()

#     # Cross
#     if "is_cross" in data:
#         cross_flag = data["is_cross"].astype(bool)
#     elif "pass_cross" in data:
#         cross_flag = data["pass_cross"].astype(bool)
#     else:
#         cross_flag = pd.Series(False, index=data.index)
#         for col in ("type_name", "event_type", "type", "event", "sub_type", "subtype", "pass_type"):
#             if col in data:
#                 cross_flag |= data[col].astype(str).str.lower().str.contains("cross", na=False)
#     if cross_requires_pass:
#         cross_flag &= pass_flag  # optional

#     # Long ball
#     if "is_long_ball" in data:
#         long_ball_flag = data["is_long_ball"].astype(bool)
#     elif "pass_long_ball" in data:
#         long_ball_flag = data["pass_long_ball"].astype(bool)
#     else:
#         text_flag = pd.Series(False, index=data.index)
#         for col in ("type_name", "event_type", "type", "event", "sub_type", "subtype", "pass_type"):
#             if col in data:
#                 text_flag |= data[col].astype(str).str.lower().str.contains("long ball", na=False)

#         len_flag = pd.Series(False, index=data.index)
#         if "pass_length" in data:
#             len_flag = pd.to_numeric(data["pass_length"], errors="coerce") >= float(long_ball_min_len)

#         geom_flag = (data["seg_len"] >= float(long_ball_min_len)) & pass_flag

#         # If text explicitly says "long ball", count it even if not a pass.
#         long_ball_flag = text_flag | ((len_flag | geom_flag) & pass_flag)

#     data["is_pass"] = pass_flag
#     data["is_cross"] = cross_flag
#     data["is_long_ball"] = long_ball_flag

#     # Box flag at event level (using end point)
#     y0 = dims.width / 2.0
#     data["in_box"] = (
#         (data["end_x"] >= dims.box_x_min)
#         & (data["end_y"].between(y0 - dims.box_half_width, y0 + dims.box_half_width))
#     )

#     grp = data.sort_values(["sequence_id", "time_seconds"]).groupby("sequence_id", sort=False)

#     y_min = pd.concat([grp["start_y"].min(), grp["end_y"].min()], axis=1).min(axis=1).astype(float)
#     y_max = pd.concat([grp["start_y"].max(), grp["end_y"].max()], axis=1).max(axis=1).astype(float)
#     y_spread = (y_max - y_min).astype(float)

#     seq_dict = {
#         "game_id": grp["game_id"].first(),
#         "team_id": grp["team_id"].first(),
#         "start_x": grp["start_x"].first().astype(float),
#         "start_y": grp["start_y"].first().astype(float),
#         "end_x": grp["end_x"].last().astype(float),
#         "end_y": grp["end_y"].last().astype(float),
#         "t_start": grp["time_seconds"].first().astype(float),
#         "t_end": grp["time_seconds"].last().astype(float),
#         "path_len": grp["seg_len"].sum().astype(float),
#         "med_y": grp["end_y"].median().astype(float),
#         "max_x": grp["end_x"].max().astype(float),
#         "box_entry": grp["in_box"].any(),
#         "y_spread": y_spread,
#         "n_passes": grp["is_pass"].sum().astype(int),
#         "n_crosses": grp["is_cross"].sum().astype(int),
#         "n_long_balls": grp["is_long_ball"].sum().astype(int),
#     }
#     if has_team_name:
#         seq_dict["team_name"] = grp["team_name"].first()

#     seq = pd.DataFrame(seq_dict).reset_index()

#     seq["duration"] = (seq["t_end"] - seq["t_start"]).clip(lower=1e-3)
#     seq["dx"] = seq["end_x"] - seq["start_x"]
#     seq["forward"] = seq["dx"] > 0
#     seq["reach_half"] = seq["max_x"] >= dims.halfway_x

#     seq["directness"] = np.where(
#         seq["path_len"] > 0,
#         np.clip(seq["dx"] / seq["path_len"], 0.0, 1.0),
#         0.0,
#     ).astype(float)

#     seq["width_index"] = np.clip(seq["y_spread"] / dims.width, 0.0, 1.0).astype(float)

#     order = [
#         "sequence_id", "game_id", "team_id",
#         "start_x", "start_y", "end_x", "end_y",
#         "t_start", "t_end", "duration", "dx",
#         "path_len", "med_y", "max_x",
#         "forward", "reach_half", "box_entry",
#         "directness", "width_index",
#         "n_passes", "n_crosses", "n_long_balls",
#     ]
#     if has_team_name:
#         order.insert(order.index("team_id") + 1, "team_name")

#     return seq[order]
# metrics/sequence_summary.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

__all__ = ["PitchDims", "summarize_open_play_sequences"]


@dataclass(frozen=True)
class PitchDims:
    """
    Pitch geometry in meters (default: 105 x 68).
    Penalty Area (PA) assumed 40.32 x 16.5 centered on pitch width.
    """
    length: float = 105.0
    width: float = 68.0
    box_width: float = 40.32
    box_length: float = 16.5

    @property
    def halfway_x(self) -> float:
        return self.length / 2.0

    @property
    def box_x_min(self) -> float:
        return self.length - self.box_length

    @property
    def box_half_width(self) -> float:
        return self.box_width / 2.0

    def in_box(self, x: float, y: float) -> bool:
        y0 = self.width / 2.0
        return (x >= self.box_x_min) and (abs(y - y0) <= self.box_half_width)


def summarize_open_play_sequences(
    events: pd.DataFrame,
    *,
    dims: PitchDims = PitchDims(),
    long_ball_min_len: float = 30.0,
    cross_requires_pass: bool = False,
) -> pd.DataFrame:
    """
    One row per *open-play* sequence with geometry, timing, and shape metrics.
    Adds: `def3rd_to_beyond_half` — started in defensive third AND finished beyond halfway.
    """
    has_team_name = "team_name" in events.columns

    def _empty_df():
        cols = [
            "sequence_id", "game_id", "team_id",
            "start_x", "start_y", "end_x", "end_y",
            "t_start", "t_end", "duration", "dx",
            "path_len", "med_y", "max_x",
            "forward", "reach_half", "def3rd_to_beyond_half", "box_entry",
            "directness", "width_index",
            "n_passes", "n_crosses", "n_long_balls",
        ]
        if has_team_name:
            cols.insert(cols.index("team_id") + 1, "team_name")
        return pd.DataFrame(columns=cols)

    if events.empty:
        return _empty_df()

    in_seq = events["in_sequence"].astype(bool) if "in_sequence" in events else pd.Series(True, index=events.index)
    valid = events["sequence_valid"].astype(bool) if "sequence_valid" in events else pd.Series(True, index=events.index)

    data = events.loc[in_seq & valid].copy()
    if data.empty:
        return _empty_df()

    for c in ("start_x", "start_y", "end_x", "end_y", "time_seconds"):
        data[c] = pd.to_numeric(data[c], errors="coerce")

    dx_ev = data["end_x"] - data["start_x"]
    dy_ev = data["end_y"] - data["start_y"]
    data["seg_len"] = np.hypot(dx_ev, dy_ev)

    # --- Flexible event flags ---
    # Pass
    if "is_pass" in data:
        pass_flag = data["is_pass"].astype(bool)
    else:
        pass_flag = pd.Series(False, index=data.index)
        for col in ("event_type", "type", "type_name", "event", "action", "sub_type", "subtype", "pass_type"):
            if col in data:
                pass_flag |= data[col].astype(str).str.lower().str.contains(r"\bpass\b", na=False)
        if "pass_length" in data:
            pass_flag |= data["pass_length"].notna()

    # Cross
    if "is_cross" in data:
        cross_flag = data["is_cross"].astype(bool)
    elif "pass_cross" in data:
        cross_flag = data["pass_cross"].astype(bool)
    else:
        cross_flag = pd.Series(False, index=data.index)
        for col in ("type_name", "event_type", "type", "event", "sub_type", "subtype", "pass_type"):
            if col in data:
                cross_flag |= data[col].astype(str).str.lower().str.contains("cross", na=False)
    if cross_requires_pass:
        cross_flag &= pass_flag

    # Long ball
    if "is_long_ball" in data:
        long_ball_flag = data["is_long_ball"].astype(bool)
    elif "pass_long_ball" in data:
        long_ball_flag = data["pass_long_ball"].astype(bool)
    else:
        text_flag = pd.Series(False, index=data.index)
        for col in ("type_name", "event_type", "type", "event", "sub_type", "subtype", "pass_type"):
            if col in data:
                text_flag |= data[col].astype(str).str.lower().str.contains("long ball", na=False)

        len_flag = pd.Series(False, index=data.index)
        if "pass_length" in data:
            len_flag = pd.to_numeric(data["pass_length"], errors="coerce") >= float(long_ball_min_len)

        geom_flag = (data["seg_len"] >= float(long_ball_min_len)) & pass_flag
        long_ball_flag = text_flag | ((len_flag | geom_flag) & pass_flag)

    data["is_pass"] = pass_flag
    data["is_cross"] = cross_flag
    data["is_long_ball"] = long_ball_flag

    # Box flag at event level (using end point)
    y0 = dims.width / 2.0
    data["in_box"] = (
        (data["end_x"] >= dims.box_x_min)
        & (data["end_y"].between(y0 - dims.box_half_width, y0 + dims.box_half_width))
    )

    # Aggregate per sequence
    grp = data.sort_values(["sequence_id", "time_seconds"]).groupby("sequence_id", sort=False)
    y_min = pd.concat([grp["start_y"].min(), grp["end_y"].min()], axis=1).min(axis=1).astype(float)
    y_max = pd.concat([grp["start_y"].max(), grp["end_y"].max()], axis=1).max(axis=1).astype(float)
    y_spread = (y_max - y_min).astype(float)

    seq_dict = {
        "game_id": grp["game_id"].first(),
        "team_id": grp["team_id"].first(),
        "start_x": grp["start_x"].first().astype(float),
        "start_y": grp["start_y"].first().astype(float),
        "end_x": grp["end_x"].last().astype(float),
        "end_y": grp["end_y"].last().astype(float),
        "t_start": grp["time_seconds"].first().astype(float),
        "t_end": grp["time_seconds"].last().astype(float),
        "path_len": grp["seg_len"].sum().astype(float),
        "med_y": grp["end_y"].median().astype(float),
        "max_x": grp["end_x"].max().astype(float),
        "box_entry": grp["in_box"].any(),
        "y_spread": y_spread,
        "n_passes": grp["is_pass"].sum().astype(int),
        "n_crosses": grp["is_cross"].sum().astype(int),
        "n_long_balls": grp["is_long_ball"].sum().astype(int),
    }
    if has_team_name:
        seq_dict["team_name"] = grp["team_name"].first()

    seq = pd.DataFrame(seq_dict).reset_index()

    # Derived fields
    seq["duration"] = (seq["t_end"] - seq["t_start"]).clip(lower=1e-3)
    seq["dx"] = seq["end_x"] - seq["start_x"]
    seq["forward"] = seq["dx"] > 0
    seq["reach_half"] = seq["max_x"] >= dims.halfway_x

    # New: started in defensive third AND finished beyond halfway
    def_third_max_x = dims.length / 3.0
    seq["def3rd_to_beyond_half"] = (seq["start_x"] < def_third_max_x) & (seq["end_x"] >= dims.halfway_x)

    # Metrics (0–1)
    seq["directness"] = np.where(
        seq["path_len"] > 0,
        np.clip(seq["dx"] / seq["path_len"], 0.0, 1.0),
        0.0,
    ).astype(float)
    seq["width_index"] = np.clip(seq["y_spread"] / dims.width, 0.0, 1.0).astype(float)

    order = [
        "sequence_id", "game_id", "team_id",
        "start_x", "start_y", "end_x", "end_y",
        "t_start", "t_end", "duration", "dx",
        "path_len", "med_y", "max_x",
        "forward", "reach_half", "def3rd_to_beyond_half", "box_entry",
        "directness", "width_index",
        "n_passes", "n_crosses", "n_long_balls",
    ]
    if has_team_name:
        order.insert(order.index("team_id") + 1, "team_name")

    return seq[order]
