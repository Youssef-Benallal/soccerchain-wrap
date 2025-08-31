from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from tslearn.metrics import dtw


# ---------------------------------------------------
# Functions
# ---------------------------------------------------

def build_sequence_polylines(events: pd.DataFrame, pitch_length=105, pitch_width=68):
    sort_cols = ["game_id", "sequence_id", "period_id", "time_seconds"]
    ev = events.sort_values(sort_cols).copy()
    seq_events, seq_points = {}, {}
    for seq_id, g in ev.groupby("sequence_id"):
        g = g.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
        if g.empty:
            continue
        xs = [g.iloc[0]["start_x"]]
        ys = [g.iloc[0]["start_y"]]
        xs.extend(g["end_x"].to_list())
        ys.extend(g["end_y"].to_list())
        pts = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
        pts[:, 0] = np.clip(pts[:, 0], 0, pitch_length)
        pts[:, 1] = np.clip(pts[:, 1], 0, pitch_width)
        keep = [True]
        for i in range(1, len(pts)):
            keep.append(not np.allclose(pts[i], pts[i-1]))
        pts = pts[np.array(keep, bool)]
        if len(pts) < 2:
            continue
        seq_events[seq_id] = g
        seq_points[seq_id] = pts
    return seq_events, seq_points

def pairwise_dtw_distance_matrix(X_list):
    N = len(X_list)
    D = np.zeros((N, N), dtype=float)
    for i, j in combinations(range(N), 2):
        dist = dtw(X_list[i], X_list[j])
        D[i, j] = D[j, i] = dist
    return D

# ---------------------------------------------------
# A) Cluster (no plots)
# ---------------------------------------------------
def cluster_buildups_affprop(df,
                             pitch_length=105,
                             pitch_width=68,
                             preference=None,
                             random_state=42):
    seq_events, seq_points = build_sequence_polylines(df, pitch_length, pitch_width)
    seq_ids = list(seq_points.keys())
    X_list = [seq_points[sid] for sid in seq_ids]
    D = pairwise_dtw_distance_matrix(X_list)
    S = -D
    ap = AffinityPropagation(affinity="precomputed",
                             preference=preference,
                             random_state=random_state)
    labels = ap.fit_predict(S)
    centers_idx = ap.cluster_centers_indices_
    medoid_seq_ids = [seq_ids[i] for i in centers_idx]
    return {
        "seq_ids": seq_ids,
        "labels": labels,
        "medoid_seq_ids": medoid_seq_ids,
        "distance_matrix": D,
        "similarity_matrix": S
    }

# ---------------------------------------------------
# OOP wrapper
# ---------------------------------------------------

class SequencePathCluster:
    """
    Thin wrapper exposing an object-oriented sequence clustering
    """

    def __init__(self, pitch_length=105, pitch_width=68, preference=None, random_state=42):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.preference = preference
        self.random_state = random_state

    def fit_predict(self, df: pd.DataFrame):
        return cluster_buildups_affprop(
            df,
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            preference=self.preference,
            random_state=self.random_state,
        )
