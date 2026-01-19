import os
import sys
import tempfile
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

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        return metrics

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
        'LogReg_distance': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/LogReg_Model_with_distance:v7'),
        'LogReg_angle': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/LogReg_Model_with_angle:v5'),
        'Logreg_distance_angle': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/LogReg_Model_with_distance_and_angle:v5'),
        'XGBoost': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/XGB_features_and_hp_tuned:v2'),
        'Stacking': TestModel('IFT6758--2025-A03/IFT6758.2025-A03/StackingModel:v2')
    }

    # Models features
    features = {
        'LogReg_distance': ['Distance'],
        'LogReg_angle': ['Angle'],
        'Logreg_distance_angle': ['Distance', 'Angle'],
        'XGBoost': ["Distance", "Angle", "shot_deflected", "shot_slap", "Opponent_Skaters", "empty_net",
                    "Friendly_Skaters", "speed", "time_since_last", "shot_wrist", "shot_wrap-around",
                    "shot_tip-in", "shot_backhand", "shot_snap", "is_rebound"],
        'Stacking': ['all']
    }

    # -- Load test data --
    # Regular season
    test_data_regular = f'../../games_data/feature_dataset_2_test_regular.csv'
    df_test_regular = pd.read_csv(test_data_regular)

    X_test_regular = df_test_regular.drop(columns=['is_goal'])
    y_test_regular = df_test_regular['is_goal']

    # Playoffs
    test_data_playoffs = f'../../games_data/feature_dataset_2_test_playoff.csv'
    df_test_playoffs = pd.read_csv(test_data_playoffs)

    X_test_playoffs = df_test_playoffs.drop(columns=['is_goal'])
    y_test_playoffs = df_test_playoffs['is_goal']

    # Final test data
    test_data = {
        'regular': (X_test_regular, y_test_regular),
        'playoff': (X_test_playoffs, y_test_playoffs)
    }

    # Evaluate models and compute curves data
    metrics = {
        'regular': {},
        'playoff': {},
    }

    results = {
        'regular': {},
        'playoff': {},
    }

    for model_name, model in models.items():

        for data_name, (test_data_X, test_data_y) in test_data.items():

            # Modify test dataset according to model feature selection
            if features[model_name][0] != 'all':
                if model_name == 'XGBoost':
                    test_data_X = pd.get_dummies(test_data_X, columns=['Type of Shot'], prefix='shot')
                    missing_cols = [shot_type for shot_type in features[model_name] if shot_type not in test_data_X.columns]
                    for shot_type in missing_cols:
                        test_data_X[shot_type] = 0
                    test_data_X = test_data_X[features[model_name]]
                else:
                    test_data_X = test_data_X[features[model_name]]

            # Metrics
            model_metrics = model.evaluate(test_data_X, test_data_y)
            metrics[data_name][model_name] = model_metrics

            # Curves data
            fpr, tpr, roc_auc, y_prob = model.get_ROC(test_data_X, test_data_y)

            # Results
            results[data_name][model_name] = {
                'roc': (fpr, tpr, roc_auc, y_prob),
                'goal_rate': model.get_goal_rate_by_percentile(test_data_y, y_prob),
                'cumulative': model.get_cumulative_goal(test_data_y, y_prob),
                'calibration': model.get_calibration(test_data_y, y_prob)
            }


            """
            Plot curves individually
            """

            # --- ROC ---
            roc_fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            if data_name == 'regular':
                ax.set_title(f'Courbe ROC - {model_name} (Test - Saison régulière)')
            else:
                ax.set_title(f'Courbe ROC - {model_name} (Test - Playoffs)')
            ax.set_xlabel("Taux de faux positifs")
            ax.set_ylabel("Taux de vrais positifs")
            ax.legend()
            ax.grid(True)


            # --- Goal Rate by Percentile ---
            x_g, y_g = results[data_name][model_name]['goal_rate']
            y_smooth = pd.Series(y_g).rolling(window=5, center=True, min_periods=1).mean()

            goal_rate_fig, ax = plt.subplots()
            ax.plot(x_g, y_smooth, label=f'{model_name}')
            ax.invert_xaxis()
            if data_name == 'regular':
                ax.set_title(f'Taux de buts par centile - {model_name} (Test - Saison régulière)')
            else:
                ax.set_title(f'Taux de buts par centile - {model_name} (Test - Playoffs)')
            ax.set_xlabel('Centile de probabilité')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_ylim(0, 1)
            ax.set_ylabel('Taux de buts')
            ax.legend()
            ax.grid(True)


            # --- Cumulative Goals ---
            x_c, y_c = results[data_name][model_name]['cumulative']

            cumulative_goals_fig, ax = plt.subplots()
            ax.plot(x_c, y_c, label=f'{model_name}')
            if data_name == 'regular':
                ax.set_title(f'Proportion cumulée des buts - {model_name} (Test - Saison régulière)')
            else:
                ax.set_title(f'Proportion cumulée des buts - {model_name} (Test - Playoffs)')
            ax.set_xlabel('Proportion de tirs')
            ax.set_ylabel('Proportion cumulée de buts')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1 - x) * 100:.0f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
            ax.legend()
            ax.grid(True)


            # --- Calibration / Reliability ---
            x, y = results[data_name][model_name]['calibration']

            calibration_fig, ax = plt.subplots()
            ax.plot(x, y, label=f'{model_name}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
            if data_name == 'regular':
                ax.set_title(f'Courbe de calibration/fiabilité - {model_name} (Test - Saison régulière)')
            else:
                ax.set_title(f'Courbe de calibration/fiabilité - {model_name} (Test - Playoffs)')
            ax.set_xlabel('Probabilité prédite')
            ax.set_ylabel('Fréquence observée')
            ax.legend()
            ax.grid(True)


            # --- Log metrics and figures ---
            logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'{model_name} Test')

            # Metrics
            logger.log_metrics(model_metrics, as_summary=True)

            # Figures
            logger.log_figure(f'ROC_{model_name}_test-{data_name}', roc_fig)
            logger.log_figure(f'GoalRate_{model_name}_test-{data_name}', goal_rate_fig)
            logger.log_figure(f'CumulativeGoals_{model_name}_test-{data_name}', cumulative_goals_fig)
            logger.log_figure(f'Calibration_{model_name}_test-{data_name}', calibration_fig)

            plt.close(roc_fig)
            plt.close(goal_rate_fig)
            plt.close(cumulative_goals_fig)
            plt.close(calibration_fig)

            logger.finish()



    """
    Plot summary figures
    """

    for data_name in ['regular', 'playoff']:

        # --- ROC ---
        roc_fig_summary, ax = plt.subplots()
        for model_name, data in results[data_name].items():
            fpr, tpr, roc_auc, _ = data['roc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        if data_name == 'regular':
            ax.set_title(f'Courbe ROC - Test (Saison régulière)')
        else:
            ax.set_title(f'Courbe ROC - Test (Playoffs)')
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


        # --- Goal Rate by Percentile ---
        goal_rate_fig_summary, ax = plt.subplots()
        for model_name, data in results[data_name].items():
            x, y = data['goal_rate']
            y_smooth = pd.Series(y).rolling(window=5, center=True, min_periods=1).mean()
            ax.plot(x, y_smooth, label=model_name)

        ax.invert_xaxis()
        if data_name == 'regular':
            ax.set_title(f'Taux de buts par centile - Test (Saison régulière)')
        else:
            ax.set_title(f'Taux de buts par centile - Test (Playoffs)')
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
        for model_name, data in results[data_name].items():
            x, y = data['cumulative']
            ax.plot(x, y, label=model_name)
        if data_name == 'regular':
            ax.set_title(f'Proportion cumulée de buts - Test (Saison régulière)')
        else:
            ax.set_title(f'Proportion cumulée de buts - Test (Playoffs)')
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
        for model_name, data in results[data_name].items():
            x, y = data['calibration']
            ax.plot(x, y, label=model_name)
        if data_name == 'regular':
            ax.set_title(f'Courbe de calibration/fiabilité - Test (Saison régulière)')
        else:
            ax.set_title(f'Courbe de calibration/fiabilité - Test (Playoffs)')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Fréquence observée')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


        # --- Log figures ---
        logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'Tests Summary')

        logger.log_figure(f'ROC_summary_test-{data_name}', roc_fig_summary)
        logger.log_figure(f'GoalRate_summary_test-{data_name}', goal_rate_fig_summary)
        logger.log_figure(f'CumulativeGoals_summary_test-{data_name}', cumulative_goals_fig_summary)
        logger.log_figure(f'Calibration_summary_test-{data_name}', calibration_fig_summary)

        plt.close(roc_fig_summary)
        plt.close(goal_rate_fig_summary)
        plt.close(cumulative_goals_fig_summary)
        plt.close(calibration_fig_summary)

        logger.finish()