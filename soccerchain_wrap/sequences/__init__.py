# Re-export the minimal public API for sequences
from .core import Rules, SequenceDetector
from .buildup import BuildupRules, detect_buildup_sequences
from .open_play import OpenPlayRules, detect_open_play_sequences
from .open_play_shot_ending import ShotEndingRules, detect_shot_ending_sequences
from .qc import add_sequence_gap_flag, interpolate_small_gaps_with_carries, qc_layer

__all__ = [
    "Rules",
    "SequenceDetector",
    "BuildupRules",
    "ShotEndingRules",
    "detect_shot_ending_sequences",
    "detect_buildup_sequences",
    "OpenPlayRules",
    "detect_open_play_sequences",
    "add_sequence_gap_flag",
    "interpolate_small_gaps_with_carries",
    "qc_layer",
]

