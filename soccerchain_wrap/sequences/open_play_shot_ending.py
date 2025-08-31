# sequences/shot_ending_sequences.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Set
import pandas as pd

from .core import Rules, SequenceDetector
from .qc import qc_layer
from .open_play import _default_interruptions  # reuse your defaults

def _lc(s) -> str: return str(s).strip().lower()
def _get(ev, name: str, default=None):
    return ev.get(name, default) if isinstance(ev, pd.Series) else getattr(ev, name, default)

# --- rules: start on PASS, stop on SHOT; also break on default interruptions ---
@dataclass
class ShotEndingRules(Rules):
    interruptions: Optional[Set[str]] = None     # lowercased names that break
    shot_types: Optional[Set[str]] = None        # lowercased names counted as shots

    def __post_init__(self) -> None:
        if self.interruptions is None:
            self.interruptions = _default_interruptions()
        if self.shot_types is None:
            self.shot_types = {"shot"}

    def is_start(self, ev, prev_team_id: Optional[int]) -> bool:
        return _lc(_get(ev, "type_name", "")) == "pass"

    def is_interrupt(self, ev) -> bool:
        t = _lc(_get(ev, "type_name", ""))
        # team-change is already handled by the engine; here we add domain breaks
        return t in (self.interruptions or set()) or t in (self.shot_types or set())

    def is_valid_end(self, last_ev) -> bool:
        # valid iff we ended on a shot
        return _lc(_get(last_ev, "type_name", "")) in (self.shot_types or set())

    def custom_interrupt(self, ev, *, current_seq: list[int], df: pd.DataFrame) -> bool:
        # ensure the SHOT itself is INCLUDED as the last event
        if _lc(_get(ev, "type_name", "")) in (self.shot_types or set()):
            try: current_seq.append(getattr(ev, "Index"))   # itertuples path
            except Exception: current_seq.append(getattr(ev, "name"))  # Series path
            return True
        return False

# --- public API ---
def detect_shot_ending_sequences(
    df: pd.DataFrame,
    *,
    min_events: int = 2,
    shot_types: Iterable[str] = ("shot",),
    extra_interruptions: Optional[Iterable[str]] = None,
    # QC layer surface (same as open-play)
    apply_qc: bool = True,
    qc_steps: Iterable[str] = ("gap_flag", "interpolate"),
    dx_gap_thresh: float = 20.0,
    dy_gap_thresh: float = 13.0,
    interpolate_min_gap: float = 1.0,
    interpolate_operate_on: str = "clean_sequences",  # "clean" | "flagged" | "all"
) -> pd.DataFrame:
    """
    Shot-ending sequences:
      • start with a pass
      • break on: SHOT, default interruptions, or possession change
      • sequence is valid only if it ends on a shot
    Adds: `seq_num_passes` (number of passes inside each valid sequence).
    """
    rules = ShotEndingRules(
        shot_types={_lc(s) for s in shot_types},
        interruptions=_default_interruptions(),
    )
    if extra_interruptions:
        rules.interruptions |= { _lc(s) for s in extra_interruptions }  # type: ignore

    out = SequenceDetector(rules, min_events=min_events).annotate(df.copy())

    # tag number of passes per valid sequence
    valid = out["in_sequence"].astype(bool) & out["sequence_valid"].astype(bool)
    pass_mask = out["type_name"].astype(str).str.lower().eq("pass")
    pass_count = (
        out.loc[valid & pass_mask]
          .groupby("sequence_id")["type_name"].size()
          .rename("seq_num_passes")
    )
    out["seq_num_passes"] = out["sequence_id"].map(pass_count)

    # QC layer
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

    return out
