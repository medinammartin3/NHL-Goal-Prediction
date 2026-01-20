import requests
import numpy as np
import pandas as pd
import logging
from typing import Dict
import os
import sys

sys.path.append(os.path.abspath("../"))
from ift6758.features.Pbp_DataFrame import Pbp_to_DataFrame
from ift6758.features.data_engeneering_2 import apply_feature_engineering

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GameClient:
    def __init__(self, serving_client=None):
        """
        serving_client = Instance de ServingClient pour envoyer les predictions
        """
        self.serving_client = serving_client
        # Handler to convert the raw NHL JSON into pandas dataframe
        self.pbp_handler = Pbp_to_DataFrame()
        # Internal tracker: dict {game_id: set(event_ids)}
        self.processed_events: Dict[str, set] = {}

    def get_schedule(self, date_obj):
        """
        Fetches the list of games for a specific date (YYYY-MM-DD).
        Returns a list of dicts: [{'game_id': ..., 'label': 'MTL vs BOS'}]
        """
        date_str = date_obj.strftime("%Y-%m-%d")
        try:
            url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            games_list = []
            # The API returns a 'gameWeek' list; we find the day matching our date
            for day in data.get("gameWeek", []):
                if day["date"] == date_str:
                    for game in day["games"]:
                        g_id = game["id"]
                        home = game["homeTeam"]["commonName"]["default"]
                        away = game["awayTeam"]["commonName"]["default"]
                        # Optional: Add score if game is finished/live
                        games_list.append({
                            "id": g_id,
                            "label": f"{away} @ {home}"
                        })
                    break
            return games_list
        except Exception as e:
            logger.error(f"Failed to fetch schedule: {e}")
            return []


    def compute_score(self, df):
        """ Compute the current score from the processed events dataframe"""
        goals = df[df["Event Type"] == "Goal"]

        # Normalize team names for robust comparison (remove spaces, lowercase)
        team_norm = goals["Team"].astype(str).str.strip().str.lower()
        home_norm = goals["Home Team"].astype(str).str.strip().str.lower()

        # Count goals for home and away teams
        score_home = (team_norm == home_norm).sum()
        score_away = (team_norm != home_norm).sum()
        return score_home, score_away

    def fetch_live_game(self, game_id: str, chunk_size: int = 5) -> pd.DataFrame:
        """
        Recupere tous les evenements dun jeu live et retourne seulement les evenements non traite
        """
        if not hasattr(self, 'simulation_cache'):
            self.simulation_cache = {}

        # Initial fetch and full feature engineering
        # If the game data hasnt been fetched and processed yet
        if game_id not in self.simulation_cache:
            try:
                # Fetch the complete JSON from NHL API
                response = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play")
                response.raise_for_status()
                game_data = response.json()
            except requests.RequestException as e:
                logger.error(f"Erreur lors de la recuperation du jeu {game_id}: {e}")
                return pd.DataFrame(), None, None, None, None  # return an empty dataframe and none for other variables

            # Build the full dataframe of events and apply all necessary feature engineering
            df_full = self.pbp_handler.build_game_DataFrame(game_id, game=game_data)
            # Apply all feature engineering functions one by one
            df_full = apply_feature_engineering(df_full)

            # Cache the full, processed dataframe along with team names.
            self.simulation_cache[game_id] = {
                "df_full": df_full,
                "home_team": game_data["homeTeam"]["commonName"]["default"],
                "away_team": game_data["awayTeam"]["commonName"]["default"],
            }

            # Initialize the event tracker for this game if it doesnt exist yet
            if game_id not in self.processed_events:
                self.processed_events[game_id] = set()

        # Filter new events and create chunk
        cached = self.simulation_cache[game_id]
        df_full = cached["df_full"]

        # Filter the full dataframe to include ONLY events whose ID is NOT in our processed set
        new_events_df = df_full[~df_full['ID'].isin(self.processed_events[game_id])]

        # Take only a chunk of the requested size (simulating a small batch update)
        new_chucks = new_events_df.iloc[:chunk_size].copy()

        # If no new events are detected, return an empty dataframe and the current score
        if new_events_df.empty:
            logger.info(f"Aucun nouvel evenement pour le jeu {game_id}")
            # Get the current score from already processed events
            processed_df = df_full[df_full["ID"].isin(self.processed_events[game_id])]
            score_home, score_away = self.compute_score(processed_df)
            return (
                pd.DataFrame(),
                cached["home_team"],
                cached["away_team"],
                score_home,
                score_away,
            )

        logger.info(f"Game {game_id}: {len(new_events_df)} nouveaux evenements detectes")

        # Update the tracker with the IDs of the events we are about to process
        self.processed_events[game_id].update(new_chucks['ID'].tolist())

        # Calculate the current score based on ALL processed events up to this point
        processed_df = df_full[df_full["ID"].isin(self.processed_events[game_id])]
        score_home, score_away = self.compute_score(processed_df)

        return (new_chucks, cached["home_team"], cached["away_team"], score_home, score_away)


    def process_and_predict(self, game_id: str) -> pd.DataFrame:
        """
        Recupere les nouveaux evenements et envoie les features au servide de prediction
        """
        # Get the next chunk of new events
        new_events_df, home_team, away_team, score_home, score_away = self.fetch_live_game(game_id)

        # Handle case where fetch_live_game returns None due to API error
        if new_events_df is None:
            return pd.DataFrame(), home_team, away_team, score_home, score_away

        # Handle case where fetch_live_game returns an empty dataframe (no new events)
        if new_events_df.empty:
            logger.info(f"Aucun nouvel evenement pour le jeu {game_id}")
            return pd.DataFrame(), home_team, away_team, score_home, score_away

        # If a serving client is provided, proceed with sending data for prediction
        if self.serving_client:
            try:
                # Retrieve the list of features the model is expecting from the serving client
                feature_cols = self.serving_client.features

                X = new_events_df.copy()

                # One-hot encoding for XGBoost
                if 'Type of Shot' in X.columns:
                    needs_encoding = any(f.startswith('shot_') and f not in X.columns for f in feature_cols)
                    if needs_encoding:
                        logger.info("Applying One-Hot Encoding to 'Type of Shot'...")
                        X = pd.get_dummies(X, columns=['Type of Shot'], prefix='shot')

                        # Ensure ALL required shot columns exist, fill with 0 if missing
                        for f in feature_cols:
                            if f.startswith('shot_') and f not in X.columns:
                                X[f] = 0

                # Check if all required features exist in the processed data
                missing_features = [f for f in feature_cols if f not in X.columns]
                if missing_features:
                    logger.error(f"Features manquantes: {missing_features}")
                    # Return the raw events and current score, skipping prediction
                    return new_events_df, home_team, away_team, score_home, score_away

                # Only keep features needed for prediction
                X_predict = X[feature_cols].copy()

                # Clean data
                X_predict = X_predict.replace([np.inf, -np.inf], 0).fillna(0)

                logger.info(f"Envoi de {len(X_predict)} événements avec features: {feature_cols}")

                # Get the predictions from the Flask service
                predictions_df = self.serving_client.predict(X_predict)

                # Join the prediction results back to the original new events dataframe
                new_events_df = pd.concat([new_events_df,predictions_df], axis=1)

            except Exception as e:
                logger.error(f"Erreur lors de l'envoi des predictions: {e}")
                # If predictions fails, return the events without the prediction column.
                return new_events_df, home_team, away_team, score_home, score_away

        # Return the new events (now potentially augmented with predictions) and game status.
        return new_events_df, home_team, away_team, score_home, score_away