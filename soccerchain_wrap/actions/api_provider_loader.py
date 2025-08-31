import soccerdata as sd
import pandas as pd
from socceraction.data.opta.loader import OptaLoader
from typing import Tuple


def get_whoscored_object_api(league: str, season: int) -> Tuple[pd.DataFrame, OptaLoader]:
    """
    Load schedule and OptaLoader (SPADL API) from WhoScored.

    Parameters
    ----------
    league : str
        League code (e.g. "ENG-Premier League")
    season : int
        Season year (e.g. 2024)

    Returns
    -------
    schedule : pd.DataFrame
        DataFrame containing the schedule.
    api : OptaLoader
        SoccerAction API for accessing SPADL-converted event data.
    """
    ws = sd.WhoScored(leagues=league, seasons=season)
    schedule = ws.read_schedule().reset_index()

    api = ws.read_events(
        output_fmt="loader",
        match_id=list(schedule["game_id"].unique())
    )

    return schedule, api
