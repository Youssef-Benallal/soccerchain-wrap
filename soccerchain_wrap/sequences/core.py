from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Dict, List, Optional, Iterable, Tuple
import pandas as pd

REQUIRED_COLS: Tuple[str, ...] = ("game_id", "team_id", "type_name", "start_x", "end_x")


@runtime_checkable
class Rules(Protocol):
    def is_start(self, ev: pd.Series, prev_team_id: Optional[int]) -> bool: ...
    def is_interrupt(self, ev: pd.Series) -> bool: ...
    def is_valid_end(self, last_ev: pd.Series) -> bool: ...

    # optional, domain-specific breaker (returns True to split)
    def custom_interrupt(
        self,
        ev: pd.Series,
        *,
        current_seq: List[int],
        df: pd.DataFrame,
    ) -> bool:
        return False


@dataclass
class SequenceDetector:
    rules: Rules
    min_events: int = 2

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        _validate(df, REQUIRED_COLS)
        # Avoid re-sorting if already monotonic
        out = df.copy()
        if not out.index.is_monotonic_increasing:
            out.sort_index(inplace=True)

        if "sequence_id" not in out:
            out["sequence_id"] = pd.NA
        if "in_sequence" not in out:
            out["in_sequence"] = False
        if "sequence_valid" not in out:
            out["sequence_valid"] = False

        for game_id in out["game_id"].drop_duplicates().tolist():
            self._process_game(out, int(game_id))
        return out

    def _process_game(self, df: pd.DataFrame, game_id: int) -> None:
        gdf = df.loc[df["game_id"] == game_id]

        # bind rule methods once (hot-loop micro-optimization)
        r_is_start = self.rules.is_start
        r_is_interrupt = self.rules.is_interrupt
        r_custom_interrupt = self.rules.custom_interrupt

        counters: Dict[int, int] = {}
        in_sequence = False
        poss_team: Optional[int] = None
        current: List[int] = []
        prev_team_id: Optional[int] = None

        # iterate as namedtuples (fast); exposes columns as attributes
        for row in gdf.itertuples(index=True, name="E"):
            idx = row.Index
            team_id = int(row.team_id)

            if not in_sequence:
                if r_is_start(row, prev_team_id):
                    in_sequence = True
                    poss_team = team_id
                    current = [idx]
                prev_team_id = team_id
                continue

            team_changed = team_id != poss_team
            base_interrupt = r_is_interrupt(row) or team_changed

            # domain-specific split (e.g., “another initiating pass appeared”)
            custom_break = bool(r_custom_interrupt(row, current_seq=current, df=df))
            interrupted = base_interrupt or custom_break

            if interrupted:
                self._finalize_if_valid(df, current, counters, game_id, poss_team)
                in_sequence = False
                poss_team = None
                current = []
                # if custom_break (same team) and this event is a valid start → restart on it
                if (custom_break and not team_changed) and r_is_start(row, prev_team_id):
                    in_sequence = True
                    poss_team = team_id
                    current = [idx]
            else:
                current.append(idx)

            prev_team_id = team_id

        self._finalize_if_valid(df, current, counters, game_id, poss_team)

    def _finalize_if_valid(
        self,
        df: pd.DataFrame,
        current: List[int],
        counters: Dict[int, int],
        game_id: int,
        team_id: Optional[int],
    ) -> None:
        if not current:
            return
        if len(current) < self.min_events:
            return
        last_ev = df.loc[current[-1]]
        if self.rules.is_valid_end(last_ev):  # type: ignore
            team = int(df.loc[current[0], "team_id"]) if team_id is None else int(team_id)
            n = counters.get(team, 1)
            seq_id = f"{game_id}-{team}-{n}"
            counters[team] = n + 1
            df.loc[current, "in_sequence"] = True
            df.loc[current, "sequence_id"] = seq_id
            df.loc[current, "sequence_valid"] = True


def _validate(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
