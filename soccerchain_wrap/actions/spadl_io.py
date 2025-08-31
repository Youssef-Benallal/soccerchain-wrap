import os
import logging
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import re
from typing import Dict
import pandas as pd
from socceraction.data.opta.loader import OptaLoader
import socceraction.spadl as spadl
import tqdm
from soccerchain_wrap.actions.api_provider_loader import (
    get_whoscored_object_api
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SpadlIO:
    def __init__(
        self,
        leagues: list[str],
        seasons: list[int],
        root: str = None # type: ignore
    ):  
        if root is None:
            root = (
                Path(__file__).resolve().parents[2] / "data" / "spadl"
            )

        self.leagues = leagues
        self.seasons = seasons
        self.root = root


    def save_spadl(self):
        """Store SPADL data for all league-season combinations."""
        for league in self.leagues:
            for season in self.seasons:
                try:
                    logger.info(f"Fetching/Storing {league} - {season} data")
                    schedule, api = get_whoscored_object_api(league, season)
                    self._save_league_season_spadl(api, schedule, league, season)
                except Exception as e:
                    logger.exception(f"Failed: {league} - {season} — {e}")
    

    def load_spadl(self) -> Dict[str, pd.DataFrame]:
        """
        Load SPADL HDF5 files and return a dict: {(league-season): actions_df}.
        Skips folders with invalid names or missing files.
        """
        if not os.path.exists(self.root):
            logger.warning(f"SPADL folder not found: {self.root}")
            return {}

        spadl_data = {}
        for folder in os.listdir(self.root):
            path = os.path.join(self.root, folder)
            if not os.path.isdir(path) or not re.match(r".+-\d{4}$", folder):
                continue

            season = folder.split("-")[-1]
            h5_file = os.path.join(path, f"{season}-spadl-opta.h5")
            if not os.path.exists(h5_file):
                logger.warning(f"Missing HDF5 in: {folder}")
                continue

            try:
                with pd.HDFStore(h5_file) as store:
                    games = store["games"]
                    teams = store["teams"]
                    players = store["players"]
                    actions = []
                    for game in games.itertuples():
                        a = store[f"actions/game_{game.game_id}"]
                        a = spadl.add_names(a)
                        actions.append(a)
                    df = pd.concat(actions).merge(
                        teams, on="team_id"
                    ).merge(
                        players, on=[
                            "game_id", 
                            "team_id", 
                            "player_id"
                        ], how="left"
                    ).merge(
                        games[[
                            "game_id",
                            'game',
                            'home_score',
                            'away_score'
                        ]], on="game_id",
                        how='left'
                    )
                    spadl_data[folder] = df
            except Exception as e:
                logger.warning(f"Failed to load {folder}: {e}")

        return spadl_data


    def _save_league_season_spadl(
        self,
        api: OptaLoader,
        schedule: pd.DataFrame,
        league: str,
        season: int
    ):
        """Store SPADL components for a single league-season."""
        folder = os.path.join(self.root, f"{league}-{season}")
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{season}-spadl-opta.h5")

        logger.info(f"Accessing {league} - {season} SPADL data")
        teams, players, events = [], [], []
        actions = {}
        games_verbose = tqdm.tqdm(
            list(schedule.itertuples()), desc="Loading game data"
        )
        for game in games_verbose:
            # load data
            teams.append(api.teams(game.game_id)) # type: ignore
            players.append(api.players(game.game_id)) # type: ignore
            events.append(api.events(game.game_id)) # type: ignore
            # convert data to actions
            actions[game.game_id] = spadl.opta.convert_to_actions(
                api.events(game.game_id),  # type: ignore
                home_team_id=game.home_team_id # type: ignore
            )
        # Concat games datas
        events = pd.concat(events)
        teams = pd.concat(teams).drop_duplicates(
            subset="team_id"
        )
        players = pd.concat(players)
        players = self._add_player_former_positions(players)

        # Store
        with pd.HDFStore(filepath) as store:
            logger.info(f"Storing {league} - {season} SPADL data to {filepath}")
            store["games"] = schedule
            store["teams"] = teams
            store["events"] = events
            store["players"] = players
            # Store actions data
            for game_id, action_df in actions.items():
                store[f"actions/game_{game_id}"] = action_df

        logger.info(f"Successfull Storing: {filepath}")


    @staticmethod
    def _add_player_former_positions(players: pd.DataFrame) -> pd.DataFrame:
        """Add a 'former_position' column to players DataFrame."""
        counts = (
            players.groupby([
                "player_id", 
                "player_name", 
                "starting_position"
            ])
            .size()
            .reset_index(name="count")
            .sort_values(
                ["player_id", "count"], 
                ascending=[True, False]
            )
        )

        former_pos = (
            counts.groupby([
                "player_id", 
                "player_name"
            ])["starting_position"]
            .apply(
                lambda x: x.iloc[1]
                if x.iloc[0] == "Sub" and len(x) > 1
                else x.iloc[0]
            )
            .reset_index(name="former_position")
        )

        players = (
            players
            .merge(former_pos, on=[
                "player_id", 
                "player_name"
            ], how="left")
            .drop_duplicates(subset=[
                "player_id", 
                "game_id"
            ])
        )

        return players
