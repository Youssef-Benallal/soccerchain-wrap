from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Tuple
import streamlit as st
import streamlit_shadcn_ui as ui

import joblib
import matplotlib.pyplot as plt
from soccerchain_wrap.actions.spadl_io import SpadlIO
from soccerchain_wrap.metrics.pipe import metric_percentile_table_by_team
from soccerchain_wrap.metrics.sequence_summary import PitchDims
from soccerchain_wrap.helpers.plotting import plot_league_pizza_grid, plot_pass_clusters
from soccerchain_wrap.sequences.buildup import detect_buildup_sequences
from soccerchain_wrap.sequences.open_play import detect_open_play_sequences
from soccerchain_wrap.sequences.open_play_shot_ending import detect_shot_ending_sequences

from soccerchain_wrap.models.sequence_clustering.utils import trim_sequence
from soccerchain_wrap.models.sequence_clustering.sequence_paths import SequencePathCluster
from soccerchain_wrap.helpers.plotting import plot_seq_clusters_grid
from soccerchain_wrap.metrics.sequence_summary import summarize_open_play_sequences, PitchDims

from soccerchain_wrap.helpers.plotting import (
    plot_cluster_to_player_share_heatmap,
    prepare_initiations,
    make_team_table, 
    plot_columns_ranked_by_metric, 
    shot_seq_donut_mpl,
    plot_top_support_players
)
import io



# ---------------- Config + CSS ----------------
st.set_page_config(page_title="Soccerchain-Wrap", page_icon="⚽")
st.set_page_config(layout="wide")

st.markdown("""
<style>
   button[data-baseweb="tab"] {
   font-size: 24px;
   margin: 0;
   width: 100%;
   }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 0.2rem !important;  /* smaller than default 6rem */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* make sidebar narrower */
[data-testid="stSidebar"] {
    min-width: 300px;
    max-width: 300px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Folder discovery ----------------
def discover_spadl(root: str = "data/spadl") -> Tuple[Dict[str, List[str]], Dict[Tuple[str,str], Path]]:
    leagues, lookup, base = {}, {}, Path(root)
    if not base.exists(): return {}, {}
    for d in base.iterdir():
        if not d.is_dir(): continue
        parts = d.name.rsplit("-", 1)
        if len(parts)==2:  # Flat: ENG-Premier League-2024
            lg, season = parts
            leagues.setdefault(lg, []).append(season); lookup[(lg,season)] = d
        else:              # Nested: ENG-Premier League/2024
            for s in (p for p in d.iterdir() if p.is_dir()):
                leagues.setdefault(d.name, []).append(s.name); lookup[(d.name,s.name)] = s
    for lg in leagues: leagues[lg] = sorted(leagues[lg], reverse=True)
    return leagues, lookup

leagues, _ = discover_spadl()


# ---------------- Caching helpers ----------------
@st.cache_resource(show_spinner=False)
def load_init_pass_cluster_model(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def predict_clusters_cached(actions_df, _model):   # _model ignored by hasher
    return _model.predict(actions_df)

@st.cache_data(show_spinner=False)
def detect_buildup_sequences_cached(df, **kwargs):
    return detect_buildup_sequences(df, **kwargs)

# ---------------- Sidebar ----------------
st.sidebar.title("SOCCERCHAIN-WRAP")
st.sidebar.caption("Unveiling Soccer Chain Patterns")
st.sidebar.image("assets/soccerchain_wrap_logo.png", width=280)

lg = st.sidebar.selectbox("League", sorted(leagues) if leagues else [])
seasons = leagues.get(lg, [])
sz = st.sidebar.selectbox("Season", seasons or ["No seasons found"])
y0 = int(re.match(r"\d{4}", str(sz)).group(0))  # season start

# placeholder for tab-specific sidebar filters
seq_filters_ph = st.sidebar.empty()
st.session_state.setdefault("seq_team", None)

# caches for actions and style figs
st.session_state.setdefault("cache_actions", {})
st.session_state.setdefault("cache_fig", {})
st.session_state.setdefault("seq_clustered", {})


def load_actions_once(lg: str, y0: int):
    key = (lg,y0)
    if key not in st.session_state["cache_actions"]:
        st.session_state["cache_actions"][key] = SpadlIO(leagues=[lg], seasons=[y0]).load_spadl()[f"{lg}-{y0}"]
    return st.session_state["cache_actions"][key]

def build_style_fig_once(lg: str, y0: int):
    key = (lg,y0)
    if key in st.session_state["cache_fig"]:
        return st.session_state["cache_fig"][key]
    # first time: show progress
    prog_ph, stat_ph = st.empty(), st.empty(); p = prog_ph.progress(0)
    with stat_ph.status("Preparing…", expanded=False) as s:
        s.update(label="Fetching actions"); p.progress(25)
        actions = load_actions_once(lg,y0)

        s.update(label="Computing style metrics"); p.progress(60)
        df = metric_percentile_table_by_team(actions, dims=PitchDims(), include_raw=False)

        s.update(label="Rendering style framework"); p.progress(90)
        fig = plot_league_pizza_grid(df, team_name_col="team_name", ncols=4, nrows=5,
                                     figsize=(16,20), sort_by="team_name", logo_dir="assets", logo_zoom=0.035)
        fig.patch.set_facecolor("white")
        p.progress(100); s.update(label="Ready", state="complete")
    prog_ph.empty(); stat_ph.empty()
    st.session_state["cache_fig"][key] = fig
    return fig


# ---------------- Tabs ----------------
tab_style, tab_seq = st.tabs(["Playing Style Framework", "Open Play Sequence Framework"])

with tab_style:
    st.subheader(f"Playing Style Wheel — {lg} · {y0}/{y0+1}")
    fig = build_style_fig_once(lg, y0)
    # centered badges
    _, c1,c2,c3,c4,_ = st.columns([1,1,1,1,1,1], gap="medium")
    with c1: st.badge("Progression", color="orange")
    with c2: st.badge("Possession", color="green")
    with c3: st.badge("Defence",   color="red")
    with c4: st.badge("Attacking", color="blue")
    st.pyplot(fig)

with tab_seq:
    actions = load_actions_once(lg, y0)

    seq_tab1, seq_tab2, seq_tab3 = st.tabs([
        "Open Play Sequence Insight Metrics",
        "Unveil Deep Buildup Patterns",
        "Unveil Shot Ending Patterns"
    ])

    # ---- cache the sequence detection + summarization ----
    @st.cache_data
    def get_seq_df(pl_actions_data, *, pitch_length=105.0, pitch_width=68.0):
        pl_openplay_sequences = detect_open_play_sequences(
            pl_actions_data,
            apply_qc=True,
            qc_steps=("gap_flag", "interpolate"),
            dx_gap_thresh=20.0,
            dy_gap_thresh=13.0,
            interpolate_min_gap=1.0,
            interpolate_operate_on="clean",
        )
        dims = PitchDims(length=pitch_length, width=pitch_width)
        return summarize_open_play_sequences(
            pl_openplay_sequences,
            dims=dims,
            long_ball_min_len=30.0,
            cross_requires_pass=False,
        ), dims

    # === somewhere earlier you have pl_actions_data ===
    seq_df, dims = get_seq_df(actions)

    
    # ================== TAB CONTENT ==================
    with seq_tab1:
        st.markdown("### Open Play Sequence Metrics")

        @st.cache_data
        def _team_tbl_fixed(_seq_df, _L):
            return make_team_table(
                _seq_df,
                min_sequences=10,
                pitch_length=_L,
                escape_target="final_third",
            )

        team_tbl = _team_tbl_fixed(seq_df, dims.length)

        if team_tbl.empty:
            st.info("No teams pass the minimum sequences filter (min=10).")
        else:
            # Make the figure wider & taller (Streamlit-side only by passing args)
            fig = plot_columns_ranked_by_metric(
                team_tbl,
                metrics=(
                    ("directness_p75", "Directness /Sequence (75th %ile)", "{:.2f}"),
                    ("width_index", "Width Index /Sequence", "{:.2f}"),
                    ("passes_per_sequence", "Passes /Sequence", "{:.2f}"),
                    ("crosses_per_sequence", "Crosses /Sequence", "{:.2f}"),
                    ("long_balls_per_sequence", "Long Balls /Sequence", "{:.2f}"),
                    ("escape_rate", "Escape Rate (Def 1/3 → Final 1/3)", "{:.0%}"),
                ),
                # ---- only Streamlit-side sizing tweaks ----
                col_w=3.6,            # wider columns -> wider figure
                base_label_area=1.7,  # a bit more room for team names
                cell_h=0.26,          # slightly taller rows
                height_mult=1.85,     # scale total height
                dpi=200,              # crisper rendering on wide layout
                label_size=10, value_size=9, title_size=10,
                max_team_chars=24,
                return_fig=True
            )

            # Render full-width (no extra columns → maximizes width)
            st.pyplot(fig, use_container_width=True)

            # (Optional) download
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button(
                "Download chart (PNG)",
                data=buf.getvalue(),
                file_name="open_play_sequence_metrics.png",
                mime="image/png",
            )

    with seq_tab2:
        seq_key = (lg, y0)
        if seq_key not in st.session_state["seq_clustered"]:
            prog_ph, stat_ph = st.empty(), st.empty(); p = prog_ph.progress(0)
            with stat_ph.status("Starting…", expanded=False) as s:
                s.update(label="Loading model"); p.progress(15)
                model = load_init_pass_cluster_model("models/initiating_pass_cluster.joblib")
                s.update(label="Predicting initiating pass clusters"); p.progress(55)
                clustered = predict_clusters_cached(actions, _model=model)
                p.progress(100); s.update(label="Completed", state="complete")
            prog_ph.empty(); stat_ph.empty()
            st.session_state["seq_clustered"][seq_key] = clustered

        pl_actions_data_cluster = st.session_state["seq_clustered"][seq_key]

        # sidebar team filter
        with seq_filters_ph.container():
            st.markdown("##### Sequence Filters") 
            teams = sorted(pl_actions_data_cluster["team_name"].unique())
            st.selectbox("Team", teams, key="seq_team")
        team_name = st.session_state["seq_team"] or teams[0]

        # compute buildup sequences (cached)
        pl_buildup_sequences = detect_buildup_sequences_cached(
            pl_actions_data_cluster,
            defensive_third_max_x=35.0, halfway_x=52.5,
            apply_qc=True, qc_steps=("gap_flag","interpolate"),
            dx_gap_thresh=20.0, dy_gap_thresh=13.0,
            interpolate_min_gap=1.0, interpolate_operate_on="clean"
        )

        c1, c2 = st.columns([5, 4], gap="medium")
        TARGET_SCALE = 1.35   # make both charts bigger; tweak 1.2–1.6

        # ---- RIGHT: heatmap first (to set target height) ----
        with c2:
            df_init = prepare_initiations(pl_buildup_sequences.query("team_name == @team_name"))
            fig_hm = plot_cluster_to_player_share_heatmap(df_init, top_n=10, add_others=False,
                                                        annot_min=1.0, tick_size=8, annot_size=7)
            if fig_hm is not None:
                w_hm, h_hm = fig_hm.get_size_inches()
                H = max(8.0, h_hm * TARGET_SCALE)
                fig_hm.set_size_inches(w_hm * (H / h_hm), H)
                fig_hm.set_dpi(180)
                st.pyplot(fig_hm, use_container_width=True)
                st.markdown(
                    f"<div style='text-align:center; font-size:18px;'><b>{team_name}</b> Initiating Passes Matrix</div>"
                    "<div style='text-align:center; font-size:10px; color:gray;'>"
                    "Distribution of Initiating Passes Cluster x Player."
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                H = 8.0
                st.info("No initiations heatmap for this team.")

        # ---- LEFT: initiating-pass clusters (match height, slightly wider) ----
        with c1:
            plot_pass_clusters(pl_actions_data_cluster, team_name, num_clusters=30, num_cols=4)
            fig_left = plt.gcf()
            w_l, h_l = fig_left.get_size_inches()
            fig_left.set_size_inches(w_l * (H / max(h_l, 1e-6)) * 1.10, 1.14*H)  # 10% wider, same height
            # minimal fix: match column width ratio 5/4 exactly
            fig_left.set_dpi(180)
            st.pyplot(fig_left, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center; font-size:18px;'><b>{team_name}</b> Initiating Passes Clusters</div>"
                "<div style='text-align:center; font-size:10px; color:gray;'>"
                "Initiating passes are the first forward passes that start a buildup sequence from the defensive third."
                "</div>",
                unsafe_allow_html=True
            )

        # ---- Explore Buildup Sequence Paths (unchanged) ----
        st.divider(); st.markdown("##### Explore Buildup Sequence Paths")
        clusters = (
            pl_buildup_sequences.loc[pl_buildup_sequences.team_name == team_name, "buildup_cluster"]
            .dropna().astype(int).sort_values().unique().tolist()
        )
        if not clusters:
            st.info("No buildup clusters for this team.")
        else:
            k = st.selectbox("Buildup cluster", clusters, key=f"bk_{lg}_{y0}_{team_name}")
            seq_ids = pl_buildup_sequences.query(
                "team_name == @team_name & buildup_cluster == @k"
            )["sequence_id"].dropna().unique()
            if seq_ids.size == 0:
                st.warning("No sequences match the selection.")
            else:
                ev = pl_buildup_sequences[pl_buildup_sequences["sequence_id"].isin(seq_ids)].copy()
                ev = trim_sequence(ev, threshold=85.0)
                res = SequencePathCluster().fit_predict(ev)
                plot_seq_clusters_grid(df_events=ev, results_dict=res, ncols=3, pitch_length=105, pitch_width=68)
                st.pyplot(plt.gcf())



    # ---- cache shot-ending sequences once ----
    @st.cache_data(show_spinner=False)
    def get_shot_ending_sequences_cached(df):
        return detect_shot_ending_sequences(
            df,
            apply_qc=True,
            qc_steps=("gap_flag", "interpolate"),
            dx_gap_thresh=20.0,
            dy_gap_thresh=13.0,
            interpolate_min_gap=1.0,
            interpolate_operate_on="clean",
        )

    with seq_tab3:
        st.markdown("#### Unveil Shot Ending Patterns")

        # 1) actions from cache + sequences (cached)
        actions = load_actions_once(lg, y0)
        shot_ending_sequences = get_shot_ending_sequences_cached(actions)

        # 2) reuse team filter from seq_tab2 (fallback to first)
        all_teams = sorted(shot_ending_sequences["team_name"].dropna().unique())
        team_name = st.session_state.get("seq_team", (all_teams[0] if all_teams else None))

        if not all_teams or team_name is None:
            st.info("Select a team in the Sequence Filters.")
        else:
            # Two columns
            c1, c2 = st.columns([1, 1], gap="medium")

            # RIGHT first -> use its height as target
            with c2:
                tbl_sup, fig_sup = plot_top_support_players(
                    shot_ending_sequences,
                    team=team_name,
                    top_n=5,
                    fig_width=8.0,
                    inside_label_size=9,
                    player_fontsize=9,
                )

            # LEFT donut
            with c1:
                tbl_donut, fig_donut = shot_seq_donut_mpl(
                    shot_ending_sequences,
                    team=team_name,
                    label_pos=0.42,
                    label_size=11,
                )

            # 3) Align heights & margins so visuals are not déphasés
            if fig_donut is not None and fig_sup is not None:
                TARGET_H = 7.6   # adjust 7.2–8.2 if needed
                TOP = 0.87       # common top margin (0–1)

                for F in (fig_donut, fig_sup):
                    # same height for both figures
                    w, h = F.get_size_inches()
                    F.set_size_inches(w * (TARGET_H / max(h, 1e-6)), TARGET_H)
                    F.set_dpi(180)
                    # remove internal titles and unify margins
                    try: F.suptitle("")
                    except Exception: pass
                    for ax in F.axes:
                        ax.set_title("")
                    F.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=TOP)

                with c1:
                    st.pyplot(fig_donut, use_container_width=True)
                with c2:
                    st.pyplot(fig_sup, use_container_width=True)
            else:
                # graceful fallback
                if fig_donut is not None:
                    st.pyplot(fig_donut, use_container_width=True)
                if fig_sup is not None:
                    st.pyplot(fig_sup, use_container_width=True)


