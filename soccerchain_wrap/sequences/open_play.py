# sequences/open_play_sequences.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Iterable
import pandas as pd

from .core import Rules, SequenceDetector
from .qc import qc_layer

# ---------- helpers ----------
def _lc(s) -> str:
    return str(s).strip().lower()

def _get(ev, name: str, default=None):
    return ev.get(name, default) if isinstance(ev, pd.Series) else getattr(ev, name, default)

# ---------- simple interruptions ----------
def _default_interruptions() -> Set[str]:
    """
    Minimal set that *breaks* an open-play sequence.
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

@dataclass
class OpenPlayRules(Rules):
    """
    Open-play sequences (simple):
      • START  : first event is a pass
      • STOP   : when an interruption event appears (list above)
      • END OK : always True (≥2 events handled by engine)
    """
    interruptions: Optional[Set[str]] = None

    def __post_init__(self) -> None:
        if self.interruptions is None:
            self.interruptions = _default_interruptions()

    def is_start(self, ev, prev_team_id: Optional[int]) -> bool:
        return _lc(_get(ev, "type_name", "")) == "pass"

    def is_interrupt(self, ev) -> bool:
        return _lc(_get(ev, "type_name", "")) in (self.interruptions or set())

    def is_valid_end(self, last_ev) -> bool:
        return True  # engine enforces min_events

    def custom_interrupt(self, ev, *, current_seq: list[int], df: pd.DataFrame) -> bool:
        return False

# -------- public API --------
def detect_open_play_sequences(
    df: pd.DataFrame,
    *,
    min_events: int = 2,
    extra_interruptions: Optional[Iterable[str]] = None,
    # QC layer toggles/params (mirrors buildup.py)
    apply_qc: bool = True,
    qc_steps: Iterable[str] = ("gap_flag", "interpolate"),
    dx_gap_thresh: float = 20.0,
    dy_gap_thresh: float = 13.0,
    interpolate_min_gap: float = 1.0,
    interpolate_operate_on: str = "clean_sequences",  # "clean" | "flagged" | "all"
) -> pd.DataFrame:
    """
    Annotate open-play sequences with minimal assumptions.
    Optionally run the QC layer (gap flag + carry interpolation).
    """
    rules = OpenPlayRules()
    if extra_interruptions:
        rules.interruptions |= {str(s).lower() for s in extra_interruptions}  # type: ignore

    out = SequenceDetector(rules, min_events=min_events).annotate(df.copy())

    if apply_qc:
        out = qc_layer(
            out,
            steps=qc_steps,
            dx_gap_thresh=dx_gap_thresh,
            dy_gap_thresh=dy_gap_thresh,
            interpolate_min_gap=interpolate_min_gap,
            interpolate_operate_on=interpolate_operate_on,  # type: ignore
        )

    return out