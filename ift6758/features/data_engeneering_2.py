import os
import sys
import pandas as pd
import numpy as np
from ift6758.features.data_engeneering_1 import get_multiple_season_df, get_event_distance, get_event_angle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))     # /ift6758


#Pour la caracteristique Game Seconds
def ensure_game_seconds(df):
    # Si deja une colonne Game Seconds, skip
    if 'Game Seconds' in df.columns:
        return df
    
    #Sinon, calcule a partir de period + Time
    def to_seconds(row):
        p = int(row['Period']) - 1
        if 'Time' in row and pd.notna(row['Time']):
            mm, ss = map(int, row['Time'].split(':'))
            sec_in_period = mm * 60 + ss # 60*nb de minutes + secondes
        else:
            # rien donc 0
            sec_in_period = 0
        return p * 20 * 60 + sec_in_period 
        #20 minutes par periodes, donc nb de periodes *20*60 + temps dans la periode actuelle

    df['Game Seconds'] = df.apply(to_seconds, axis=1)
    return df



"""
Informations de l'évènement précédent récupérées dans Pbp_DataFrame.py
"""



def rebound_and_angle_change(df):
    #is_rebound: True is le last_event_type a ete un tir
    df['is_rebound'] = df['last_event_type'].isin(['shot-on-goal'])

    #Changement dangle seulement is cest un rebound
    def angle_change(row):
        if not row['is_rebound']:
            return 0.0

        # Calculer l'angle de l'évènement précédent
        last_event_x = row.get('last_x')
        last_event_y = row.get('last_y')
        shot_zone = row.get('Zone')
        angle_info_list = [last_event_x, last_event_y, shot_zone]
        if None not in angle_info_list:
            last_event_angle = get_event_angle({
                'X': last_event_x,
                'Y': last_event_y,
                'Zone': shot_zone
            })
        else:
            last_event_angle = None

        if pd.isna(row.get('Angle')) or last_event_angle is None:
            return 0.0

        return 180 - row['Angle'] - last_event_angle
    
    df['angle_change'] = df.apply(angle_change, axis=1)

    # vitesse (distance/temps), protege contre la division par 0
    def speed(row):
        if pd.isna(row['distance_from_last']) or pd.isna(row['time_since_last']) or row['time_since_last'] == 0:
            return np.nan
        return row['distance_from_last'] / row['time_since_last']
    
    df['speed'] = df.apply(speed, axis=1)
    return df



"""
BONUS: power-play
"""
def add_power_play_features(df):
    # Trier les événements pour assurer l’ordre chronologique
    df.sort_values(['Game ID', 'Game Seconds'], inplace=True)

    # Nouvelle colonne pour le temps écoulé depuis le début du power-play
    df['PP_Time_Elapsed'] = 0

    for game_id, group in df.groupby('Game ID'):
        pp_start = None  # Au début du match, pas de power-play
        # Pour chaque évènement du match
        for index, row in group.iterrows():
            # S'il y a une équipe avec supériorité de joueurs
            if row['Situation'] in ['Power play', 'Short-handed']:
                # Si aucun power-play en cours
                if pp_start is None:
                    # Initialiser le power-play
                    pp_start = row['Game Seconds']  # Temps de début
                    df.at[index, 'PP_Time_Elapsed'] = 0  # Temps écoulé
                # Si power-play en cours
                else:
                    # Mettre à jour le temps écoulé
                    df.at[index, 'PP_Time_Elapsed'] = row['Game Seconds'] - pp_start
            # Si aucune équipe en supériorité de joueurs:
            # --> Soit aucune power-play en cours
            # --> Soit power-play finie
            else:
                pp_start = None  # Réinitialiser temps de début
                df.at[index, 'PP_Time_Elapsed'] = 0  # Réinitialiser temps écoulé


    # --- Mettre à jour le nombre de joueurs sur glace ---

    # situationCode: [#gardien local (0,1) | #patineurs locaux [0-5] | #patineurs visiteurs [0-5] | #gardien visiteur (0,1)]

    # Nombre de joueurs (non-gardiens) dans chaque équipe
    df['Home_Skaters'] = df['situationCode'].astype(str).str.extract(r'^.\s*([0-9])')[0].astype(float)  # Extraire le 2ème caractère numérique (on ignore les lettres parfois présentes)
    df['Away_Skaters'] = df['situationCode'].astype(str).str.extract(r'^..\s*([0-9])')[0].astype(float)  # Extraire le 3ème caractère numérique (on ignore les lettres parfois présentes)
    # Remplacer les 0 (ou NaN) par None
    df['Home_Skaters'] = df['Home_Skaters'].replace(0.0, None)
    df['Away_Skaters'] = df['Away_Skaters'].replace(0.0, None)

    # Joueurs (non-gardiens) amicaux
    # [Si l’évènement vient de l’équipe locale : joueurs amicaux = Home_Skaters, sinon = Away_Skaters]
    df['Friendly_Skaters'] = np.where(
        df['Team'] == df['Home Team'],
        df['Home_Skaters'], df['Away_Skaters']
    )

    # Joueurs (non-gardiens) adverses
    # [Si l’évènement vient de l’équipe locale : joueurs adverses = Away_Skaters, sinon = Home_Skaters]
    df['Opponent_Skaters'] = np.where(
        df['Team'] == df['Home Team'],
        df['Away_Skaters'], df['Home_Skaters']
    )

    return df

#build_full_feature_set works if everything above was fixed
def build_full_feature_set(all_seasons_list, output_filename):
    #recupere toutes les events
    seasons_df = get_multiple_season_df(all_seasons_list)
    seasons_df = ensure_game_seconds(seasons_df)
    
    #Calcul les angles et distances pour tous les evenemnts
    seasons_df['Angle'] = seasons_df.apply(get_event_angle, axis=1)
    seasons_df['Distance'] = seasons_df.apply(get_event_distance, axis=1)

    #Ajouter les colonnes binaires manquantes
    seasons_df['is_goal'] = (seasons_df['Event Type'] == 'Goal').astype(int)
    seasons_df['empty_net'] = (seasons_df['Net'] == 'Empty').fillna(False).astype(int)

    # Ajouter rebound/angle_change/speed
    seasons_df = rebound_and_angle_change(seasons_df)

    # Ajouter les caractéristiques des power-plays
    seasons_df = add_power_play_features(seasons_df)

    #selectionne et ordonne les colonnes finales
    final_cols = [
        'Game ID',
        'Game Seconds',
        'Period',
        'X',
        'Y',
        'Distance',
        'Angle',
        'Type of Shot',
        'last_event_type',
        'last_x',
        'last_y',
        'time_since_last',
        'distance_from_last',
        'is_rebound',
        'angle_change',
        'speed',
        'is_goal',
        'empty_net',
        'PP_Time_Elapsed',
        'Friendly_Skaters',
        'Opponent_Skaters'
    ]

    #certaines columns peuvent ne pas exister : intersect
    final_cols = [c for c in final_cols if c in seasons_df.columns]
    final = seasons_df[final_cols].copy()

    #Save to CSV
    final.to_csv(f'../../games_data/{output_filename}', index=False)

    return final


def apply_feature_engineering(df):
    df = ensure_game_seconds(df)

    # Calcul les angles et distances pour tous les evenemnts
    df['Angle'] = df.apply(get_event_angle, axis=1)
    df['Distance'] = df.apply(get_event_distance, axis=1)

    # Ajouter les colonnes binaires manquantes
    df['is_goal'] = (df['Event Type'] == 'Goal').astype(int)
    df['empty_net'] = (df['Net'] == 'Empty').fillna(False).astype(int)

    # Ajouter rebound/angle_change/speed
    seasons_df = rebound_and_angle_change(df)

    # Ajouter les caractéristiques des power-plays
    df = add_power_play_features(seasons_df)

    return df


if __name__ == '__main__':
    df_train = build_full_feature_set(
        ["2016-2017", "2017-2018", "2018-2019", "2019-2020"],
          'feature_dataset_2_train.csv'
          )
    df_test = build_full_feature_set(
        ['2020-2021'],
        'feature_dataset_2_test.csv'
        )