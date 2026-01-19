import os
import sys
import tempfile

from sklearn.model_selection import train_test_split

import wandb
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from dotenv import load_dotenv
load_dotenv()

from CatBoost_model import CatBoostModel
from lr_model import LRModel
from LightGBM_model import LightGBMModel

sys.path.append(os.path.abspath("../.."))
from ift6758.models.wandb_utils import WandbLogger


class TestModel:
    def __init__(self, wandb_artifact_path):
        self.wandb_artifact_path = wandb_artifact_path
        self.model = self.load_model()

        # Special handling for the Stacking model
        self.is_stacking = isinstance(self.model, dict)
        if self.is_stacking:
            self.estimators = self.model['estimators']
            self.meta_model = self.model['meta_model']

    def load_model(self):
        api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
        artifact = api.artifact(self.wandb_artifact_path)

        # Download artifact to a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        artifact_path = artifact.download(root=temp_dir.name)
        model_file_name = [f for f in os.listdir(artifact_path) if f.endswith(".pkl")]
        model_file = os.path.join(artifact_path, model_file_name[0])

        # Load model
        if model_file.endswith(".pkl"):
            model = joblib.load(model_file)
            temp_dir.cleanup()
            return model
        else:
            raise ValueError("Model file not .pkl")

    def predict(self, X):
        if self.is_stacking:
            # Manual stacking model prediction
            base_predictions = []
            for name, model in self.estimators:
                preds = model.predict_proba(X)
                base_predictions.append(preds)
            X_meta = np.column_stack(base_predictions)
            return self.meta_model.predict(X_meta)
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if self.is_stacking:
            # Manual stacking model prediction
            base_predictions = []
            for name, model in self.estimators:
                preds = model.predict_proba(X)
                base_predictions.append(preds)
            X_meta = np.column_stack(base_predictions)
            proba = self.meta_model.predict_proba(X_meta)
            # Handle multiple output dimensions
            if proba.ndim == 1:
                return proba
            return proba[:, 1]
        else:
            proba = self.model.predict_proba(X)
            # Handle multiple output dimensions
            if proba.ndim == 1:
                return proba
            return proba[:, 1]

    def get_ROC(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc, y_prob

    @staticmethod
    def get_goal_rate_by_percentile(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob':y_prob})
        df['centile'] = pd.qcut(df['y_prob'], q=100, labels=False, duplicates='drop')
        goal_rate = df.groupby('centile')['y_true'].mean()
        return goal_rate.index, goal_rate.values

    @staticmethod
    def get_cumulative_goal(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob':y_prob})
        df = df.sort_values('y_prob', ascending=False)
        df['But cumulatifs'] = df['y_true'].cumsum()
        df['Ratio cumulatifs'] = df['But cumulatifs'] / df['y_true'].sum()
        df['Proportion de tirs'] = np.arange(1, len(df) + 1) / len(df)
        return df['Proportion de tirs'], df['Ratio cumulatifs']

    @staticmethod
    def get_calibration(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        return prob_pred, prob_true


if __name__ == "__main__":

    # Load models
    models = {
        'MLP': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/MLPClassifier:v4'),
        'CatBoost': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/CatBoostClassifier:v12'),
        'LightGBM': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/LightGBM:v2'),
        'Stacking': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/StackingModel:v2'),
        # 'XGBoost': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/XGB_features_and_hp_tuned:v2'),
        'LogisticRegression': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/LogReg_Model_with_distance_and_angle:v5')
    }

    # Models features
    features = {
        'MLP': ['all'],
        'CatBoost': ['all'],
        'LightGBM': ['all'],
        'Stacking': ['all'],
        'XGBoost': ["Distance", "Angle", "shot_deflected", "shot_slap", "Opponent_Skaters", "empty_net",
                    "Friendly_Skaters", "speed", "time_since_last", "shot_wrist", "shot_wrap-around",
                    "shot_tip-in", "shot_backhand", "shot_snap", "is_rebound"],
        'LogisticRegression': ['Distance', 'Angle']
    }

    # Load data
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    y = data['is_goal']
    X = data.drop(columns=['is_goal'])

    # Data split
    _, X_valid, _, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results={}


    for model_name, model in models.items():

        # Curves data
        model_features = features.get(model_name)

        if model_features[0] == 'all':
            X_eval = X_valid
            y_eval = y_valid
        else:
            X_eval = X_valid[model_features]
            mask = X_eval.notna().all(axis=1)
            X_eval = X_eval[mask]
            y_eval = y_valid[mask]

        fpr, tpr, roc_auc, y_prob = model.get_ROC(X_eval, y_eval)

        results[model_name] = {
            'roc': (fpr, tpr, roc_auc, y_prob),
            'goal_rate': model.get_goal_rate_by_percentile(y_eval, y_prob),
            'cumulative': model.get_cumulative_goal(y_eval, y_prob),
            'calibration': model.get_calibration(y_eval, y_prob)
        }



    """
    Plot summary figures
    """

    # --- ROC ---
    roc_fig_summary, ax = plt.subplots()
    for model_name, data in results.items():
        fpr, tpr, roc_auc, _ = data['roc']
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Courbe ROC')
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


    # --- Goal Rate by Percentile ---
    goal_rate_fig_summary, ax = plt.subplots()
    for model_name, data in results.items():
        x, y = data['goal_rate']
        y_smooth = pd.Series(y).rolling(window=5, center=True, min_periods=1).mean()
        ax.plot(x, y_smooth, label=model_name)

    ax.invert_xaxis()
    ax.set_title(f'Taux de buts par centile')
    ax.set_xlabel('Centile de probabilité')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Taux de buts')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


    # --- Cumulative Goals ---
    cumulative_goals_fig_summary, ax = plt.subplots()
    for model_name, data in results.items():
        x, y = data['cumulative']
        ax.plot(x, y, label=model_name)
    ax.set_title(f'Proportion cumulée de buts')
    ax.set_xlabel('Proportion de tirs')
    ax.set_ylabel('Proportion cumulée de buts')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1 - x) * 100:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    # --- Calibration/Reliability ---
    calibration_fig_summary, ax = plt.subplots()
    for model_name, data in results.items():
        x, y = data['calibration']
        ax.plot(x, y, label=model_name)
    ax.set_title(f'Courbe de calibration/fiabilité')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence observée')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    plt.close(roc_fig_summary)
    plt.close(goal_rate_fig_summary)
    plt.close(cumulative_goals_fig_summary)
    plt.close(calibration_fig_summary)