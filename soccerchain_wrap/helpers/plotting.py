from __future__ import annotations
from mplsoccer import Pitch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from IPython.display import display
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
from typing import Iterable, Mapping, Sequence, Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mplsoccer import PyPizza
from matplotlib.colors import LinearSegmentedColormap
import re
import os
from soccerchain_wrap.models.sequence_clustering.sequence_paths import(
    build_sequence_polylines
)
from soccerchain_wrap.models.sequence_clustering.utils import (
    seq_directness,
    seq_width_index
)
from typing import Optional
import pandas as pd, numpy as np, matplotlib.pyplot as plt


# ---------- one-shot legend (unchanged) ----------
def draw_sequence_legend():
    """Standalone legend figure so it doesn't hide the pitch drawings."""
    CLR_PASS = "#005EB8"
    CLR_CROSS = "#7D7D7D"
    CLR_DRIB = "#228B22"
    CLR_SHOT = "#C8102E"

    handles = [
        Line2D([0], [0], color=CLR_PASS, lw=3, label="Pass"),
        Line2D([0], [0], color=CLR_CROSS, lw=3, label="Cross"),
        Line2D([0], [0], color=CLR_DRIB, lw=2.5, linestyle="--", label="Dribble/Carry"),
        Line2D([0], [0], color=CLR_SHOT, lw=3.0, label="Shot"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=CLR_DRIB,
               markersize=9, label="Take-on"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=8, lw=0, label="Node: jersey # (small = step)"),
    ]

    fig = plt.figure(figsize=(4.8, 1.5))
    fig.legend(handles=handles, loc="center", frameon=True, framealpha=0.95,
               ncol=3, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ------------------- main plotting (now supports sequence_ids) --------------
def plot_team_sequences_with_data(
    df: pd.DataFrame,
    team_name: Optional[str],
    game_id: Optional[int] = None,
    max_sequences: Optional[int] = 5,
    show_table: bool = False,
    node_radius: float = 1.25,
    jersey_fs: int = 9,
    step_fs: int = 7,
    sequence_ids: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
):
    """
    Opta-style sequence plot with refined, small arrowheads.
    Nodes: white circle with jersey number inside + tiny step number.
    No legend is drawn here (use draw_sequence_legend()).

    You can filter by:
      • sequence_ids (str/int or iterable)  -> takes priority if provided
      • team_name (optional)
      • game_id   (optional)
    """

    # ---------- normalize sequence_ids ----------
    user_seq_ids: Optional[List[Union[str, int]]] = None
    if sequence_ids is not None:
        if isinstance(sequence_ids, (str, int)):
            user_seq_ids = [sequence_ids]
        else:
            user_seq_ids = list(sequence_ids)

    # ---------- filter rows ----------
    df_filt = df.copy()
    if "in_sequence" in df_filt.columns:
        df_filt = df_filt[df_filt["in_sequence"] == True]  # noqa: E712
    if team_name is not None:
        df_filt = df_filt[df_filt["team_name"] == team_name]
    if game_id is not None and "game_id" in df_filt.columns:
        df_filt = df_filt[df_filt["game_id"] == game_id]
    if user_seq_ids is not None:
        df_filt = df_filt[df_filt["sequence_id"].isin(user_seq_ids)]

    if df_filt.empty:
        print("No sequences found for the given filters.")
        return

    # ---------- define plotting order ----------
    if user_seq_ids is not None:
        present = pd.unique(df_filt["sequence_id"])
        present_set = set(present)
        seq_order = [sid for sid in user_seq_ids if sid in present_set]
        if max_sequences is not None:
            seq_order = seq_order[:max_sequences]   # <- enforce cap here too
    else:
        seq_order = list(pd.unique(df_filt["sequence_id"].dropna()))
        if max_sequences is not None:
            seq_order = seq_order[:max_sequences]


    if not seq_order:
        print("No matching sequence_id after filtering.")
        return

    # ---------- pitch & palette ----------
    pitch = Pitch(
        pitch_type="custom", pitch_length=105, pitch_width=68,
        pitch_color="white", line_color="black", linewidth=1.1, goal_type="box"
    )
    CLR_PASS = "#005EB8"
    CLR_CROSS = "#7D7D7D"
    CLR_DRIB = "#228B22"
    CLR_SHOT = "#C8102E"
    CLR_MISC = "#A0A0A0"

    # ---------- helpers ----------
    def _safe_cols(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        keep = [c for c in cols if c in frame.columns]
        return frame[keep].copy()

    _HEAD_KW = dict(headwidth=5.0, headlength=5.0, headaxislength=4.2)

    def _arrow_refined(ax, x0, y0, x1, y1, color, width=2.0, alpha=0.97, z=3):
        pitch.arrows(
            x0, y0, x1, y1, ax=ax,
            color=to_rgba(color, alpha),
            width=width,
            zorder=z,
            **_HEAD_KW
        )

    def _carry_refined(ax, x0, y0, x1, y1, color, width=2.0, alpha=0.95, z=3):
        ax.plot(
            [x0, x1], [y0, y1],
            linestyle=(0, (4, 4)),
            linewidth=width,
            color=to_rgba(color, alpha),
            solid_capstyle="round",
            zorder=z,
        )

    def _format_jersey(val):
        if pd.isna(val):
            return None
        try:
            return str(int(float(val))) if float(val).is_integer() else str(val)
        except Exception:
            return str(val)

    def _node(ax, x, y, edge_color, jersey_number=None, step_no=None):
        circ = Circle((x, y), radius=node_radius,
                      facecolor="white", edgecolor=edge_color,
                      linewidth=2.0, zorder=5)
        ax.add_patch(circ)
        if jersey_number is not None:
            ax.text(x, y, jersey_number, ha="center", va="center",
                    fontsize=jersey_fs, color="black", weight="bold", zorder=6)
        if step_no is not None:
            ax.text(x + node_radius * 0.9, y - node_radius * 0.9, f"{step_no}",
                    ha="left", va="top", fontsize=step_fs, color="black", zorder=6)

    # ---------- render ----------
    for sequence_id in seq_order:
        seq_df = df_filt[df_filt["sequence_id"] == sequence_id].sort_values("time_seconds").copy()
        if seq_df.empty:
            continue

        if "type_name" in seq_df.columns:
            seq_df["pass_number"] = (seq_df["type_name"] == "pass").cumsum()

        if show_table:
            table_cols = [
                "game_id", "team_name", "player_name", "jersey_number",
                "type_name", "start_x", "start_y", "end_x", "end_y",
                "sequence_id", "period_id", "pass_number", "is_synthetic_carry"
            ]
            display_df = _safe_cols(seq_df, table_cols)
            if "time_seconds" in seq_df.columns:
                display_df["timestamp_min"] = (seq_df["time_seconds"] / 60).round(2)
            display(display_df.reset_index(drop=True))

        fig, ax = pitch.draw(figsize=(12.5, 6.8))
        ax.axvline(x=52.5, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(x=33,   color="#E74C3C", linestyle=(0, (2, 4)), linewidth=1.2)

        for i, row in seq_df.iterrows():
            x0, y0 = row.get("start_x"), row.get("start_y")
            x1, y1 = row.get("end_x"), row.get("end_y")
            t = row.get("type_name", "")
            if pd.isna(x0) or pd.isna(y0):
                continue

            step_no = seq_df.index.get_loc(i) + 1
            jersey = _format_jersey(row.get("jersey_number"))

            edge_color = (
                CLR_SHOT if t in {"shot", "shot_freekick", "shot_penalty"}
                else CLR_CROSS if t == "cross"
                else CLR_DRIB if t in {"dribble", "take_on"}
                else CLR_PASS if t == "pass"
                else CLR_MISC
            )

            # draw trajectory first
            if t == "dribble":
                if not (pd.isna(x1) or pd.isna(y1)):
                    _carry_refined(ax, x0, y0, x1, y1, color=CLR_DRIB)
            elif t == "take_on":
                if not (pd.isna(x1) or pd.isna(y1)):
                    _carry_refined(ax, x0, y0, x1, y1, color=CLR_DRIB, alpha=0.6)
                    pitch.scatter(x1, y1, ax=ax, s=90, marker="s",
                                  color=CLR_DRIB, zorder=6)
            elif t in {"shot", "shot_freekick", "shot_penalty"}:
                if not (pd.isna(x1) or pd.isna(y1)):
                    _arrow_refined(ax, x0, y0, x1, y1, color=CLR_SHOT, width=2.2)
            elif t == "cross":
                if not (pd.isna(x1) or pd.isna(y1)):
                    _arrow_refined(ax, x0, y0, x1, y1, color=CLR_CROSS, width=2.0)
            elif t == "pass":
                if not (pd.isna(x1) or pd.isna(y1)):
                    _arrow_refined(ax, x0, y0, x1, y1, color=CLR_PASS, width=2.0)
            else:
                if not (pd.isna(x1) or pd.isna(y1)):
                    pitch.lines(x0, y0, x1, y1, ax=ax,
                                color=to_rgba(CLR_MISC, 0.9),
                                linewidth=2.0, zorder=2)

            _node(ax, x0, y0, edge_color=edge_color,
                  jersey_number=jersey, step_no=step_no)

        # Title (team auto if not provided)
        team_for_title = team_name or (seq_df["team_name"].iloc[0] if "team_name" in seq_df.columns else "")
        match = seq_df["game"].iloc[0] if "game" in seq_df.columns else ""
        score = ""
        if "home_score" in seq_df.columns and "away_score" in seq_df.columns:
            score = f"{seq_df['home_score'].iloc[0]} - {seq_df['away_score'].iloc[0]}"
        seq_label = sequence_id.split("-")[-1] if isinstance(sequence_id, str) and "-" in sequence_id else str(sequence_id)
        title = f"{team_for_title} | Sequence #{seq_label}"
        if match:
            title += f" | {match}"
        if score:
            title += f" | {score}"
        ax.set_title(title, fontsize=16, pad=10)

        plt.tight_layout()
        plt.show()


# --------- convenience wrapper: plot directly by sequence_id(s) -------------
def plot_sequences_by_id(
    df: pd.DataFrame,
    sequence_ids: Union[str, int, Iterable[Union[str, int]]],
    **kwargs,
):
    """
    Convenience wrapper to plot one or multiple sequence_ids directly.
    team_name is optional here (auto-inferred per sequence for titles).
    """
    return plot_team_sequences_with_data(
        df=df,
        team_name=kwargs.pop("team_name", None),
        sequence_ids=sequence_ids,
        **kwargs,
    )


# --------- Initiating passes clustering -------------
def plot_pass_clusters(df, team_name, num_clusters=30, num_cols=4):
    """Visualize top pass clusters for a team, distinguishing successful (green) and failed (gray) passes."""

    team_passes = df[df["team_name"] == team_name]
    top_clusters = team_passes["buildup_cluster"].value_counts().index[:num_clusters]

    num_rows = -(-len(top_clusters) // num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), constrained_layout=True)
    axes = axes.flatten()

    pitch = Pitch(positional=True, pitch_type='custom', pitch_width=68, pitch_length=105, shade_middle=True, shade_alpha=0.7, # type: ignore
                  positional_color='#cbcbbc1c', shade_color='#f2f2f2')

    for idx, cluster_id in enumerate(top_clusters):
        cluster_data = team_passes[team_passes["buildup_cluster"] == cluster_id]
        centroid = cluster_data[['start_x', 'start_y', 'end_x', 'end_y']].mean()
        medoid_idx = cluster_data.index[np.argmin(cdist(cluster_data[['start_x', 'start_y', 'end_x', 'end_y']], [centroid]))]
        success_rate = (cluster_data["result_name"] == "success").mean() * 100  

        ax = axes[idx]
        pitch.draw(ax=ax)

        for _, row in cluster_data.iterrows():
            pitch.arrows(row["start_x"], row["start_y"], row["end_x"], row["end_y"],
                         width=3 if row.name == medoid_idx else 1, headwidth=3, headlength=2,
                         color="#1F5F8C" if row.name == medoid_idx else "gray",
                         alpha=1 if row.name == medoid_idx else 0.1,
                         ax=ax)

        ax.set_title(f"Cluster {int(cluster_id)} - {cluster_data.shape[0]} passes - {success_rate:.0f}% Success", fontsize=12, color='#1F5F8C')

    for ax in axes[len(top_clusters):]:  # Hide unused subplots
        ax.set_visible(False)

    plt.show()


# ---------- settings ----------
BG = "#EBEBE9"
CLR_POS = "#0e746d"   # possession
CLR_PRO = "#f19a2a"   # progression
CLR_DEF = "#d84b4b"   # defence

VAL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.95,
                boxstyle="round,pad=0.12", lw=0)

# metric keys (use the *_pct columns in your dataframe) + display labels
METRICS = [
    # Possession
    ("sequence_share_pct",          "Possession"),
    ("press_resistance_pct",        "Press resist"),
    ("deep_buildup_pct",            "Deep build"),
    # Progression
    ("central_progression_pct",     "Central"),
    ("circulate_pct",               "Circulate"),
    ("field_tilt_pct",              "Field tilt"),
    # Defence
    ("intensity_pct",               "Intensity"),
    ("high_line_pct",               "High line"),
    ("chance_prevention_pct",       "Chance prev."),
]
# slice colors: 3 per category
SLICE_COLORS = [CLR_POS]*3 + [CLR_PRO]*3 + [CLR_DEF]*3


# -------------------- logo helpers --------------------
def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def _find_logo_path(team_name: str, logo_dir: Optional[str], logo_map: Optional[Dict[str, str]]) -> Optional[str]:
    """Return a path to the logo PNG for team_name if found, else None."""
    if logo_map and team_name in logo_map and os.path.isfile(logo_map[team_name]):
        return logo_map[team_name]
    if not logo_dir:
        return None
    slug = _slug(team_name)
    for ext in (".png", ".PNG"):
        for name in (f"{team_name}{ext}", f"{slug}{ext}"):
            p = os.path.join(logo_dir, name)
            if os.path.isfile(p):
                return p
    return None

def _add_logo(ax: plt.Axes, img_path: str, zoom: float = 0.26) -> None:
    """Place a logo image roughly at the pizza center (tweak if needed)."""
    try:
        arr = plt.imread(img_path)
    except Exception:
        return
    ab = AnnotationBbox(
        OffsetImage(arr, zoom=zoom),
        (5.0, -19),                 # tweak position if your PyPizza center differs
        xycoords="data",
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=10,
    )
    ax.add_artist(ab)


# -------------------- pizza drawing --------------------
def _pizza_on_ax(ax: plt.Axes, values, title: str) -> None:
    values = [int(round(float(v))) for v in values]  # integer labels (no .0)
    params = [lab for _, lab in METRICS]

    baker = PyPizza(
        params=params,
        background_color=BG,
        straight_line_color=BG,
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    baker.make_pizza(
        values,
        ax=ax,
        param_location=120,
        color_blank_space="same",
        slice_colors=SLICE_COLORS,
        value_colors=["#000000"] * len(values),
        value_bck_colors=SLICE_COLORS,
        blank_alpha=0.40,
        kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
        kwargs_params=dict(color="#000000", fontsize=9, va="center"),
        kwargs_values=dict(
            color="#000000",
            fontsize=9,
            zorder=6,
            clip_on=False,
            bbox=VAL_BBOX,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
        )
    )
    ax.set_title(title, fontsize=11, pad=23, fontweight="bold")


def plot_league_pizza_grid(
    df_pct: pd.DataFrame,
    *,
    team_name_col: str = "team_name",
    ncols: int = 4,
    nrows: int = 5,
    figsize=(16, 20),
    sort_by: Optional[str] = "team_name",
    wspace: float = 0.30,
    hspace: float = 0.45,
    logo_dir: Optional[str] = None,
    logo_map: Optional[Dict[str, str]] = None,
    logo_zoom: float = 0.26,
) -> plt.Figure:
    # ensure all required percentile columns exist
    need_cols = [k for k, _ in METRICS]
    missing = [c for c in need_cols if c not in df_pct.columns]
    if missing:
        raise KeyError(f"Missing percentile columns in df: {missing}")

    d = df_pct.copy()
    if sort_by and sort_by in d.columns:
        d = d.sort_values(sort_by)
    d = d.head(ncols * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, subplot_kw={"projection": "polar"},
        figsize=figsize, facecolor=BG
    )
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, d.iterrows()):
        vals = [row[k] for k, _ in METRICS]
        team = str(row[team_name_col])
        _pizza_on_ax(ax, vals, title=team)
        ax.set_facecolor(BG)

        # add logo if available
        path = _find_logo_path(team, logo_dir=logo_dir, logo_map=logo_map)
        if path:
            _add_logo(ax, path, zoom=logo_zoom)

    # hide unused axes
    for ax in axes[len(d):]:
        ax.set_visible(False)

    return fig


# ---------- fade helpers ----------
def _scale_pitch_alpha(ax, scale: float):
    """
    Multiply alpha of artists created by mplsoccer.Pitch.draw() by `scale`.
    Call right AFTER pitch.draw(ax=...) and BEFORE drawing arrows/text.
    """
    # axes face
    if ax.patch is not None:
        base = ax.patch.get_alpha()
        ax.patch.set_alpha((1.0 if base is None else base) * scale)
    # lines, patches, collections added by the pitch
    for ln in list(ax.lines):
        base = ln.get_alpha()
        ln.set_alpha((1.0 if base is None else base) * scale)
    for pt in list(ax.patches):
        base = getattr(pt, "get_alpha", lambda: 1.0)()
        try:
            pt.set_alpha((1.0 if base is None else base) * scale)
        except Exception:
            pass
    for coll in list(ax.collections):
        base = coll.get_alpha()
        coll.set_alpha((1.0 if base is None else base) * scale)

# ---------- Carry-dribble helper ----------
def _is_carry_or_dribble(ev_row):
    if "is_synthetic_carry" in ev_row and pd.notna(ev_row["is_synthetic_carry"]):
        if bool(ev_row["is_synthetic_carry"]):
            return True
    if "type_name" in ev_row and isinstance(ev_row["type_name"], str):
        name = ev_row["type_name"].strip().lower()
        if "carry" in name or "dribble" in name:
            return True
    return False

def _to_float_array(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
# ---------- grid plotter ----------
def plot_seq_clusters_grid(
        df_events: pd.DataFrame,
        results_dict: dict,
        ncols: int = 3,
        pitch_length: float = 105,
        pitch_width: float = 68,
        fade_share_thresh: float = 5.0,   # fade cluster if share < this (%)
        fade_rank_from: int = 100,          # also fade clusters ranked >= this
        faded_scale: float = 0.6         # global scale for faded subplots
    ):
    """
    - Sort clusters by size (desc)
    - Medoid in blue #1F5F8C; others grey
    - If (share < fade_share_thresh) OR (rank >= fade_rank_from),
      scale alpha of EVERYTHING in that subplot (pitch, medoid, others, card).
    - Subplot title = medoid seq_id only
    - Card: avg directness (0–1), width index (0–1), share %
    """
    seq_events, seq_points = build_sequence_polylines(df_events, pitch_length, pitch_width)
    seq_ids = list(results_dict["seq_ids"])
    labels = results_dict["labels"]
    medoid_seq_ids = list(results_dict["medoid_seq_ids"])

    # cluster -> members
    K = len(medoid_seq_ids)
    members = {k: [] for k in range(K)}
    for sid, lab in zip(seq_ids, labels):
        members[lab].append(sid)

    # sort clusters by size desc
    cluster_sizes = [(k, len(members[k])) for k in range(K)]
    order = [k for k, _ in sorted(cluster_sizes, key=lambda x: x[1], reverse=True)]
    nclusters = len(order)

    nrows = int(np.ceil(nclusters / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4.8*nrows))
    axes = np.atleast_1d(axes).ravel()
    total_sequences = len(seq_ids)

    for rank_pos, k in enumerate(order, start=1):
        ax = axes[rank_pos-1]

        # draw pitch first
        pitch = Pitch(pitch_type='custom',
                      pitch_length=pitch_length,
                      pitch_width=pitch_width,
                      line_color='gray')
        pitch.draw(ax=ax)

        group = members[k]
        if not group:
            ax.set_title("—")
            continue

        med_sid = medoid_seq_ids[k] if (k < len(medoid_seq_ids)) else group[0]

        # cluster metrics
        directness_vals = [seq_directness(seq_events[sid]) for sid in group if sid in seq_events]
        width_vals = [seq_width_index(seq_points[sid], pitch_width=pitch_width) for sid in group if sid in seq_points]
        avg_direct = float(np.mean(directness_vals)) if directness_vals else 0.0
        avg_width  = float(np.mean(width_vals))     if width_vals     else 0.0
        share_pct  = 100.0 * len(group) / total_sequences if total_sequences > 0 else 0.0

        # decide fade
        fade = (share_pct < fade_share_thresh) or (rank_pos >= fade_rank_from)
        scale = faded_scale if fade else 1.0

        # if faded, scale PITCH artists now (before drawing arrows/text)
        if fade:
            _scale_pitch_alpha(ax, scale)

        # Alphas for elements (all multiplied by scale)
        alpha_med   = 1.0 * scale
        alpha_other = 0.15 * scale
        alpha_card  = 0.85 * scale

        # draw medoid (blue)
        if med_sid in seq_events:
            for _, ev in seq_events[med_sid].iterrows():
                sx, sy, ex, ey = ev["start_x"], ev["start_y"], ev["end_x"], ev["end_y"]
                if pd.isna(sx) or pd.isna(sy) or pd.isna(ex) or pd.isna(ey):
                    continue
                if _is_carry_or_dribble(ev):
                    pitch.lines(float(sx), float(sy), float(ex), float(ey), ax=ax,
                                linestyle="--", linewidth=2.6, color="#1F5F8C", alpha=alpha_med)
                else:
                    ax.annotate("", xy=(float(ex), float(ey)), xytext=(float(sx), float(sy)),
                                arrowprops=dict(arrowstyle="->", lw=2.8, color="#1F5F8C", alpha=alpha_med))

        # draw other members (grey)
        for sid in group:
            if sid == med_sid or sid not in seq_events:
                continue
            for _, ev in seq_events[sid].iterrows():
                sx, sy, ex, ey = ev["start_x"], ev["start_y"], ev["end_x"], ev["end_y"]
                if pd.isna(sx) or pd.isna(sy) or pd.isna(ex) or pd.isna(ey):
                    continue
                if _is_carry_or_dribble(ev):
                    pitch.lines(float(sx), float(sy), float(ex), float(ey), ax=ax,
                                linestyle="--", linewidth=1.2, color="grey", alpha=alpha_other)
                else:
                    ax.annotate("", xy=(float(ex), float(ey)), xytext=(float(sx), float(sy)),
                                arrowprops=dict(arrowstyle="->", lw=1.2, color="grey", alpha=alpha_other))

        # title = medoid id only (also scaled alpha)
        ttl = ax.set_title(f"Medoid ID - {med_sid}")
        ttl.set_alpha(scale)

        # metrics card (text + box both scaled)
        card_txt = (f"Directness: {avg_direct:.1f}\n"
                    f"Width: {avg_width:.1f}\n"
                    f"Share: {share_pct:.0f}%")
        txt = ax.text(2, pitch_width - 2, card_txt, va="top", color = "white", ha="left",
                      bbox=dict(boxstyle="round,pad=0.4", alpha=alpha_card),
                      alpha=scale,  # text alpha
                      zorder=10)

    # hide unused axes
    for j in range(nclusters, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def prepare_initiations(df: pd.DataFrame) -> pd.DataFrame:
    # keep only rows that carry a buildup_cluster label
    d = df[df["buildup_cluster"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["sequence_id","team_name","player_name","buildup_cluster"])

    d["time_seconds"] = pd.to_numeric(d["time_seconds"], errors="coerce")

    # earliest event per sequence (tie-break by original_event_id if present)
    sort_cols = ["sequence_id", "time_seconds"]
    if "original_event_id" in d.columns:
        sort_cols.append("original_event_id")
    idx = (
        d.sort_values(sort_cols)
         .groupby("sequence_id", sort=False)
         .head(1)
         .index
    )

    keep_cols = [c for c in ["sequence_id","team_name","player_name","player_id",
                             "buildup_cluster","former_position"] if c in d.columns]
    out = d.loc[idx, keep_cols].copy()
    out["buildup_cluster"] = out["buildup_cluster"].astype(int)
    return out


def plot_cluster_to_player_share_heatmap(df_init: pd.DataFrame,
                                         top_n: int = 15,
                                         add_others: bool = True,
                                         annot_min: float = 1.0,
                                         tick_size: int = 7,
                                         annot_size: int = 6):
    if df_init.empty:
        print("No initiations."); return

    ct = pd.crosstab(df_init["player_name"], df_init["buildup_cluster"])
    col_pct = 100 * ct.div(ct.sum(0).replace(0, np.nan), axis=1).fillna(0.0)

    totals = ct.sum(1).sort_values(ascending=False)
    top_players = [p for p in totals.index if p in col_pct.index][:top_n]
    col_pct = col_pct.loc[top_players]

    if add_others and len(ct) > len(top_players):
        others = (100 - col_pct.sum(0)).clip(lower=0)
        col_pct = pd.concat([col_pct, pd.DataFrame([others], index=["Others"])])

    col_pct = col_pct.reindex(sorted(col_pct.columns), axis=1)
    if "Others" in col_pct.index:
        col_pct = col_pct.loc[[p for p in col_pct.index if p != "Others"] + ["Others"]]

    cmap = LinearSegmentedColormap.from_list(
        "bluegrad", ["#F6FAFF", "#CFE1F2", "#9CBEDA", "#5E8FB8", "#1F5F8C"]
    )

    r, c = col_pct.shape
    cell = 0.4
    fig, ax = plt.subplots(figsize=(cell*c + 2.0, cell*r + 2.0))
    ax.imshow(col_pct.values, cmap=cmap, vmin=0, vmax=100, aspect="equal", interpolation="nearest")

    for s in ax.spines.values(): s.set_visible(False)

    ax.set_xticks(range(c)); ax.set_xticklabels(col_pct.columns, color="#1F5F8C", fontsize=tick_size)
    ax.set_yticks(range(r)); ax.set_yticklabels(col_pct.index,   color="#1F5F8C", fontsize=tick_size)

    for i in range(r):
        for j in range(c):
            v = col_pct.iat[i, j]
            if v >= annot_min:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=annot_size, color="#1F5F8C")

    fig.tight_layout()
    return fig


def make_team_table(
    seq_df: pd.DataFrame,
    team_col: str = "team_name",
    min_sequences: int = 1,
    *,
    pitch_length: float = 105.0,
    escape_target: str = "final_third",  # "final_third" or "halfway"
) -> pd.DataFrame:
    """
    Aggregate sequence-level rows to team-level metrics.

    Escape rate:
      - start in defensive third (start_x < L/3)
      - finish in target:
          * final_third: end_x >= 2L/3
          * halfway:     end_x >= L/2

    Also computes p75 directness per team.
    """
    if team_col not in seq_df.columns or seq_df[team_col].isna().all():
        team_col = "team_id"

    L = float(pitch_length)
    start_def3rd = pd.to_numeric(seq_df["start_x"], errors="coerce") < (L / 3.0)

    if escape_target == "final_third":
        dest_flag = pd.to_numeric(seq_df["end_x"], errors="coerce") >= (2.0 * L / 3.0)
        escaped = seq_df.get("def3rd_to_final_third", start_def3rd & dest_flag)
    else:
        dest_flag = pd.to_numeric(seq_df["end_x"], errors="coerce") >= (L / 2.0)
        escaped = seq_df.get("def3rd_to_beyond_half", start_def3rd & dest_flag)

    df = seq_df.copy()
    df["start_def3rd"] = start_def3rd
    df["escaped"] = escaped

    team = (
        df.groupby(team_col, dropna=False)
          .agg(
              sequences=("sequence_id", "nunique"),
              directness=("directness", "mean"),
              directness_p75=("directness", lambda s: s.quantile(0.75)),
              width_index=("width_index", "mean"),
              passes_per_sequence=("n_passes", "mean"),
              crosses_per_sequence=("n_crosses", "mean"),
              long_balls_per_sequence=("n_long_balls", "mean"),
              def3rd_sequences=("start_def3rd", "sum"),
              escape_count=("escaped", "sum"),
          )
          .reset_index()
          .rename(columns={team_col: "team"})
    )

    team["escape_rate"] = np.where(
        team["def3rd_sequences"] > 0,
        team["escape_count"] / team["def3rd_sequences"],
        np.nan,
    )

    # per-metric ranks (higher is better); robust to NaNs
    for m in [
        "directness_p75",
        "width_index",
        "passes_per_sequence",
        "crosses_per_sequence",
        "long_balls_per_sequence",
        "escape_rate",
    ]:
        valid = team[m].notna()
        rk = pd.Series(np.nan, index=team.index, dtype="float64")
        rk.loc[valid] = team.loc[valid, m].rank(ascending=False, method="min")
        team[m + "_rank"] = rk.astype("Int64")

    team = team.loc[team["sequences"] >= min_sequences].reset_index(drop=True)
    return team


def plot_columns_ranked_by_metric(
    team_tbl: pd.DataFrame,
    metrics=(
        ("directness_p75", "Directness /Sequence (75th %ile)", "{:.2f}"),
        ("width_index", "Width Index /Sequence", "{:.2f}"),
        ("passes_per_sequence", "Passes /Sequence", "{:.2f}"),
        ("crosses_per_sequence", "Crosses /Sequence", "{:.2f}"),
        ("long_balls_per_sequence", "Long Balls /Sequence", "{:.2f}"),
        ("escape_rate", "Escape Rate (Def 1/3 → Final 1/3)", "{:.0%}"),
    ),
    cmap: str = "Blues",
    cell_h: float = 0.24,        # row height
    col_w: float = 2.4,          # width per metric column
    pad_w: float = 0.55,         # spacing between columns
    base_label_area: float = 1.2,# left space for "rank. team"
    title_size: int = 9,
    label_size: int = 9,
    value_size: int = 8,
    max_team_chars: int | None = None,
    return_fig: bool = False,
    height_mult: float = 1.45,   # << NEW: multiply computed height
    dpi: int = 180,              # << NEW: crisper rendering
):
    """
    Render a single figure with one thin heat column per metric.
    Each column is independently sorted by its own value and displays
    left-side "rank. team" labels and the value in the cell.
    """
    nteams = len(team_tbl)
    ncols = len(metrics)
    if nteams == 0 or ncols == 0:
        fig, _ = plt.subplots(figsize=(6, 2))
        plt.axis("off")
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig if return_fig else plt.show()

    # estimate longest "rank. team" to size left margin
    label_samples = []
    for col, _, _ in metrics:
        s = team_tbl.sort_values(col, ascending=False)
        label_samples += [
            f"{('-' if pd.isna(r) else int(r))}.  {t}"
            for r, t in zip(s[col + "_rank"], s["team"].astype(str))
        ]
    max_chars = max(len(x) for x in label_samples) if label_samples else 12
    label_area = base_label_area + 0.055 * max_chars

    tbl = team_tbl.copy()
    if max_team_chars:
        tbl["team"] = tbl["team"].astype(str).apply(
            lambda t: t if len(t) <= max_team_chars else t[: max_team_chars - 1] + "…"
        )

    fig_h = (cell_h * nteams + 1.1) * height_mult
    fig_w = ncols * col_w + (ncols - 1) * pad_w
    fig, axes = plt.subplots(
        nrows=1, ncols=ncols, figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=pad_w / col_w), dpi=dpi
    )
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, (col, title, fmt) in zip(axes, metrics):
        s = tbl.sort_values(col, ascending=False, kind="stable").reset_index(drop=True)
        vals = s[col].to_numpy()
        ranks = s[col + "_rank"].to_numpy()
        teams = s["team"].astype(str).tolist()

        # guard against all-NaN column
        try:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = 0.0, 1.0
        except ValueError:
            vmin, vmax = 0.0, 1.0

        # heat strip
        ax.imshow(vals.reshape(-1, 1), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # widen axis for labels
        ax.set_xlim(-label_area, 0.5)
        ax.set_ylim(nteams - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])

        # left labels: "rank. team"
        xlab = -label_area + 0.06
        for i, (r, t) in enumerate(zip(ranks, teams)):
            rtxt = "-" if pd.isna(r) else f"{int(r)}"
            ax.text(xlab, i, f"{rtxt}.  {t}", ha="left", va="center", fontsize=label_size)

        # numeric value in the cell
        for i, v in enumerate(vals):
            ax.text(0, i, "–" if pd.isna(v) else fmt.format(v),
                    ha="center", va="center", fontsize=value_size)

        ax.set_title(title, fontsize=title_size, pad=5)
        for sp in ax.spines.values():
            sp.set_visible(False)

    plt.tight_layout()
    return fig if return_fig else plt.show()


# --------------------------- SHOT ENDING SEQUENCES ---------------------------
def shot_seq_donut_mpl(df: pd.DataFrame,
                       team: str,
                       short_max: int = 4,
                       long_min: int = 10,
                       title: Optional[str] = None,
                       annot_min: float = 2.0,
                       label_size: int = 12,
                       label_pos: float = 0.5):  # 0..1 across the ring; 0.5 = centered
    """Donut of shot-ending sequences by length (Short/Medium/Long) for one team."""
    d = df[(df["team_name"] == team) & df["sequence_valid"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["bucket","count","percent"]), None

    # passes per sequence (prefer precomputed)
    if "seq_num_passes" in d and d["seq_num_passes"].notna().any():
        seq_len = (d.dropna(subset=["sequence_id"])
                    .groupby("sequence_id")["seq_num_passes"].max().astype(int))
    else:
        seq_len = (d.assign(is_pass=d["type_name"].astype(str).str.lower().eq("pass"))
                    .groupby("sequence_id")["is_pass"].sum().astype(int))

    # buckets
    def bucket(n): return "Short" if n <= short_max else ("Long" if n >= long_min else "Medium")
    dist = seq_len.map(bucket).value_counts().reindex(["Short","Medium","Long"], fill_value=0)
    out = dist.rename_axis("bucket").reset_index(name="count")
    out["percent"] = out["count"] / out["count"].sum() * 100

    # donut
    colors = ["#D7E6F5", "#6FA1C7", "#1F5F8C"]  # light -> dark brand blues
    vals, labels = out["count"].to_numpy(), out["bucket"].tolist()
    total = int(vals.sum())

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ring_width = 0.42
    wedges, _ = ax.pie(
        vals, startangle=90, counterclock=False, colors=colors,
        wedgeprops=dict(width=ring_width, edgecolor="white", linewidth=2)
    )

    # place % at the geometric center of each slice (radius midway across the ring)
    r_in = 1.0 - ring_width
    r_lab = r_in + ring_width * np.clip(label_pos, 0.0, 1.0)

    for w, pct in zip(wedges, out["percent"].to_numpy()):
        if pct < annot_min:
            continue
        ang = np.deg2rad((w.theta2 + w.theta1) / 2.0)
        x, y = r_lab * np.cos(ang), r_lab * np.sin(ang)
        r, g, b, _ = w.get_facecolor()
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        txt_color = "white" if luminance < 0.55 else "#1F5F8C"
        ax.text(x, y, f"{pct:.0f}%", ha="center", va="center",
                fontsize=label_size, color=txt_color, clip_on=False)

    # legend (labels only)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax.set(aspect="equal")
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title((title or f"{team} — Shot-Ending Sequences Length") + f" ({total} Sequences)",
                 color="#1F5F8C", pad=12, fontsize=12)
    fig.tight_layout()
    return out, fig

def plot_top_support_players(
    shot_end_df: pd.DataFrame,
    team: str,
    top_n: int = 5,
    color: str = "#1F5F8C",
    fig_width: float = 8.0,            # constant width across teams
    bar_height: float = 0.55,          # bar thickness / vertical spacing
    player_fontsize: int = 9,
    inside_label_size: int = 12,
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Rank players by involvement in shot-ending sequences, excluding the SHOT
    and the final pre-shot action. Counts are unique per sequence.

    Returns (summary_df, fig) with columns:
      player_name, former_position, sequences, pct_of_sequences
    """
    d = shot_end_df[
        (shot_end_df["team_name"] == team) & shot_end_df["sequence_valid"].astype(bool)
    ].copy()
    if d.empty:
        return pd.DataFrame(columns=["player_name","former_position","sequences","pct_of_sequences"]), plt.figure()

    # order events within each sequence
    d["time_seconds"] = pd.to_numeric(d["time_seconds"], errors="coerce")
    sort_cols = ["sequence_id", "time_seconds"]
    if "original_event_id" in d.columns:
        sort_cols.append("original_event_id")
    d = d.sort_values(sort_cols)

    # index within sequence and remove last two events (pre-shot + shot)
    d["ord_in_seq"] = d.groupby("sequence_id").cumcount()
    last_ord = d.groupby("sequence_id")["ord_in_seq"].transform("max")
    core = d.loc[d["ord_in_seq"] <= (last_ord - 2), ["sequence_id","player_name","former_position"]]
    core = core.dropna(subset=["player_name"])

    # unique player participation per sequence
    part = core.drop_duplicates(["sequence_id","player_name"])

    # totals
    n_seq = d["sequence_id"].nunique()
    counts = (part.groupby(["player_name","former_position"]).size()
              .sort_values(ascending=False).head(top_n)
              .rename("sequences").reset_index())
    counts["pct_of_sequences"] = (counts["sequences"] / max(n_seq, 1) * 100).round(1)

    # y labels: "Player (POS)"
    counts["label"] = counts.apply(
        lambda r: f"{r['player_name']} ({str(r['former_position'])})" if pd.notna(r["former_position"]) else r["player_name"],
        axis=1
    )

    # --- plotting (clean, Opta/Athletic-ish) ---
    n = len(counts)
    fig_height = max(3.2, bar_height * n + 1.2)  # width fixed, height adapts to #bars
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y = counts["label"][::-1]
    v = counts["sequences"][::-1].to_numpy()
    pct = counts["pct_of_sequences"][::-1].to_numpy()

    ax.barh(y, v, color=color, alpha=0.9)

    # % labels inside bars (white text if bar is dark)
    vmax = v.max() if n else 1
    pad = max(0.02 * vmax, 0.5)  # distance from right edge
    for i, (c, p) in enumerate(zip(v, pct)):
        x = max(c - pad, c * 0.6)  # keep inside even for small bars
        txt_color = "white"  # dark blue bars → white pops best
        ax.text(x, i, f"{p:.0f}%", ha="center", va="center",
                fontsize=inside_label_size, color=txt_color)

    # style: no grids, no x axis labels/ticks, tidy spines
    ax.set_xlim(0, vmax * 1.05 if n else 1)
    ax.set_xlabel("")
    ax.set_xticks([]); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("")
    ax.set_yticks(range(n)); ax.set_yticklabels(y, color=color, fontsize=player_fontsize)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    #ax.set_title(f"{team} — Top {n} supporters in shot-ending sequences\n(excluding last two actions)",
    #            color=color, fontsize=12, pad=10)

    fig.tight_layout()
    return counts[[
        "player_name",
        "former_position",
        "sequences",
        "pct_of_sequences"
    ]], fig