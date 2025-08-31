from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Iterable
import pandas as pd

from .core import Rules, SequenceDetector
from .qc import qc_layer


def _default_interruptions() -> Set[str]:
    """
    Events that interrupt a buildup sequence.
    Adjust to your provider's taxonomy if needed.
    """
    return {
        "foul",
        "clearance",
        "goalkick",
        "throw_in",
        "corner_crossed",
        "corner_short",
        "freekick_crossed",
        "freekick_short",
        "bad_touch",
        "interception",
        "tackle",
        "keeper_save",
        "shot_freekick",
    }


# ---- Minimal fast-path precompute (no logic change) ----
def _prep_flags(df: pd.DataFrame, rules: "BuildupRules") -> pd.DataFrame:
    d = df.copy()

    type_lc = d["type_name"].astype("string").str.lower()
    sx = pd.to_numeric(d["start_x"], errors="coerce")
    ex = pd.to_numeric(d["end_x"], errors="coerce")

    has_cluster = d.get("buildup_cluster", pd.Series(index=d.index, dtype="float")).notna()
    is_pass = type_lc.eq("pass")

    d["_is_new_init"] = is_pass & has_cluster
    d["_start_ok"] = (
        is_pass
        & has_cluster
        & sx.notna()
        & ex.notna()
        & ((ex - sx) >= 0)
        & (sx <= float(rules.defensive_third_max_x))
    )
    d["_is_interrupt"] = type_lc.isin({s.lower() for s in (rules.interruptions or set())})
    d["_end_valid"] = ex.notna() & (ex >= float(rules.halfway_x))
    return d


_HELPER_COLS = ["_is_new_init", "_start_ok", "_is_interrupt", "_end_valid"]


# helpers to read Series OR namedtuple uniformly
def _has(ev, name: str) -> bool:
    return (name in ev) if isinstance(ev, pd.Series) else hasattr(ev, name)

def _get(ev, name: str, default=None):
    return ev.get(name, default) if isinstance(ev, pd.Series) else getattr(ev, name, default)


@dataclass
class BuildupRules(Rules):
    """
    Buildup definition:

      • START when: event.type_name == 'pass'
                    AND start_x <= defensive_third_max_x
      • CONTINUE while: no interruption AND team doesn't change
        (team-change is handled by the engine)
      • VALID END when: last event's end_x >= halfway_x
      • IDs reset per (game_id, team_id): "<game_id>-<team_id>-<seq_no>"
    """
    defensive_third_max_x: float = 33.0
    halfway_x: float = 52.5
    interruptions: Optional[Set[str]] = None

    def __post_init__(self) -> None:
        if self.interruptions is None:
            self.interruptions = _default_interruptions()

    # ---- Rules interface (use precomputed flags when present) ----
    def is_start(self, ev, prev_team_id: Optional[int]) -> bool:
        if _has(ev, "_start_ok"):
            return bool(_get(ev, "_start_ok"))
        # Fallback (original)
        if str(_get(ev, "type_name") or "").lower() != "pass":
            return False
        if pd.isna(_get(ev, "buildup_cluster")):
            return False
        sx = _get(ev, "start_x")
        ex = _get(ev, "end_x")
        if pd.isna(sx) or pd.isna(ex):
            return False
        if float(ex) - float(sx) < 0:
            return False
        return float(sx) <= self.defensive_third_max_x

    def is_interrupt(self, ev) -> bool:
        if _has(ev, "_is_interrupt"):
            return bool(_get(ev, "_is_interrupt"))
        return _get(ev, "type_name") in self.interruptions  # type: ignore[arg-type]

    def is_valid_end(self, last_ev) -> bool:
        if _has(last_ev, "_end_valid"):
            return bool(_get(last_ev, "_end_valid"))
        ex = _get(last_ev, "end_x")
        return pd.notna(ex) and float(ex) >= self.halfway_x

    def custom_interrupt(
        self,
        ev,
        *,
        current_seq: list[int],
        df: pd.DataFrame,
    ) -> bool:
        """
        Split when a NEW initiating pass appears inside the same sequence.

        Since SequenceDetector calls this ONLY when already in a sequence,
        and sequences start with an initiating pass by definition (is_start),
        the condition "another initiating pass appears" reduces to:
            current event is initiating pass.
        """
        if _has(ev, "_is_new_init"):
            return bool(_get(ev, "_is_new_init"))

        # Fallback (no flags): compute for the current event only
        tname = str(_get(ev, "type_name") or "").lower()
        return (tname == "pass") and pd.notna(_get(ev, "buildup_cluster"))


# -------- Public convenience API (keeps your old signature) -----------------
# Detection with optional QC layer
def detect_buildup_sequences(
    df: pd.DataFrame,
    *,
    defensive_third_max_x: float = 33.0,
    halfway_x: float = 52.5,
    # QC layer toggles/params:
    apply_qc: bool = False,
    qc_steps: Iterable[str] = ("gap_flag", "interpolate"),
    dx_gap_thresh: float = 20.0,
    dy_gap_thresh: float = 13.0,
    interpolate_min_gap: float = 1.0,
    interpolate_operate_on: str = "clean_sequences",  # "clean" | "flagged" | "all"
) -> pd.DataFrame:
    """
    Detect buildup sequences. If `apply_qc` is True, pass the result
    through qc_layer(...) with the provided parameters.
    """
    rules = BuildupRules(defensive_third_max_x=defensive_third_max_x, halfway_x=halfway_x)

    # Precompute flags once; use the same core engine
    work = _prep_flags(df, rules)
    out = SequenceDetector(rules, min_events=2).annotate(work)

    if apply_qc:
        out = qc_layer(
            out,
            steps=qc_steps,
            dx_gap_thresh=dx_gap_thresh,
            dy_gap_thresh=dy_gap_thresh,
            interpolate_min_gap=interpolate_min_gap,
            interpolate_operate_on=interpolate_operate_on,  # type: ignore
        )
        out = out[out['sequence_gap']!=1]
        out = out[out['sequence_id'].notnull()]

    # Remove helper columns to preserve original output schema
    to_drop = [c for c in _HELPER_COLS if c in out.columns]
    return out.drop(columns=to_drop) if to_drop else out
