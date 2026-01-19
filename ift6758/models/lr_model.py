import os
import sys

sys.path.append(os.path.abspath("../.."))
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from ift6758.models.wandb_utils import WandbLogger
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.ticker as mticker

class LRModel:
    def __init__(self, project_name='IFT6758.2025-A03'):
        self.model = LogisticRegression(class_weight='balanced')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics
    
    def log_model(self, artifact_name='logreg_model', artifact_type='model', description='Logistic Regression Model'):
        self.logger.log_model_artifact(self.model, artifact_name=artifact_name, artifact_type=artifact_type, description=description)


    def get_ROC(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_prob)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        return fpr, tpr, roc_auc, y_prob
    
    @staticmethod
    def get_goal_rate_by_percentile(y_true, y_prob):
        # Calcule taux de but selon le centile
        df = pd.DataFrame({'y_true': y_true, 'y_prob':y_prob})
        df['centile'] = pd.qcut(df['y_prob'], q=100, labels=False, duplicates='drop')
        goal_rate = df.groupby('centile')['y_true'].mean()
        return goal_rate.index, goal_rate.values
    
    @staticmethod
    def get_cumulative_goal(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob':y_prob})
        df = df.sort_values('y_prob', ascending=False)
        df['But cumulatifs'] = df['y_true'].cumsum()
        df['Rato cumulatifs'] = df['But cumulatifs'] / df['y_true'].sum()
        df['Proportion de tirs'] = np.arange(1, len(df) + 1) / len(df)

        return df['Proportion de tirs'], df['Rato cumulatifs']
    
    @staticmethod
    def get_calibration(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        return prob_pred, prob_true

    def finish(self):
        """Ferme la session W&B proprement"""
        self.logger.finish()

if __name__ == "__main__":
    print("Logistic Regression Model Module")
    data = pd.read_csv('../../games_data/feature_dataset_1_train.csv')

    # Read data
    X_distance = data[['Distance']]
    X_angle = data[['Angle']]
    X_distance_angle = data[['Distance', 'Angle']]

    y = data['Goal']

    # Data split
    X_train_distance, x_valid_distance, y_train_distance, y_valid_distance = train_test_split(X_distance, y, test_size=0.2, random_state=42)
    X_train_angle, x_valid_angle, y_train_angle, y_valid_angle = train_test_split(X_angle, y, test_size=0.2, random_state=42)
    X_train_distance_angle, x_valid_distance_angle, y_train_distance_angle, y_valid_distance_angle = train_test_split(X_distance_angle, y, test_size=0.2, random_state=42)
    
    # Model creation
    model_distance = LRModel()
    model_angle = LRModel()
    model_distance_angle = LRModel()

    model_dict = {'Model_with_distance':(model_distance, x_valid_distance, y_valid_distance),
                  'Model_with_angle': (model_angle, x_valid_angle, y_valid_angle),
                  'Model_with_distance_and_angle':(model_distance_angle, x_valid_distance_angle, y_valid_distance_angle)}
    
    
    model_distance.train(X_train_distance, y_train_distance)
    model_angle.train(X_train_angle, y_train_angle)
    model_distance_angle.train(X_train_distance_angle, y_train_distance_angle)

    result = {}

    #====================================================================================
    # Plot and log fig of the differents curves
    #====================================================================================

    # Compute curve data
    for name, (model, x_valid, y_valid) in model_dict.items():
        logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'{name}_balanced')
        # ROC curve
        fpr, tpr, roc_auc, y_prob = model.get_ROC(x_valid, y_valid)
        
        result[name] = {
            'roc': (fpr, tpr, roc_auc),
            'goal_rate': LRModel.get_goal_rate_by_percentile(y_valid, y_prob),
            'cumulative': LRModel.get_cumulative_goal(y_valid, y_prob),
            'calibration': LRModel.get_calibration(y_valid, y_prob)
        }
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'{name}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f'Courbe ROC - {name}')
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.legend()
        ax.grid(True)
        logger.log_figure(f'ROC_{name}', fig)
        plt.close(fig)

        # Goal rate by percentile

        x, y = result[name]['goal_rate']
        y_smooth = pd.Series(y).rolling(window=5, center=True, min_periods=1).mean()
        fig, ax = plt.subplots()
        ax.plot(x, y_smooth, color='blue', label=f'{name}')
        ax.invert_xaxis()
        ax.set_title(f'Taux de buts par centile - {name}')
        ax.set_xlabel('Centile de probabilité')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Taux de buts')
        ax.legend()
        ax.grid(True)
        logger.log_figure(f'GoalRate_{name}', fig)
        plt.close(fig)

        # Cumulative goals

        x, y = result[name]['cumulative']
        fig, ax = plt.subplots()
        ax.plot(x, y, color='blue', label=f'{name}')
        ax.set_title(f'Proportion cumulée du buts - {name}')
        ax.set_xlabel('Propotion de tir')
        ax.set_ylabel('Proportion cumulée de buts')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1-x)*100:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.legend()
        ax.grid(True)
        logger.log_figure(f'CumulativeGoals_{name}', fig)
        plt.close(fig)

        # Calibration

        x, y = result[name]['calibration']
        fig, ax = plt.subplots()
        ax.plot(x, y, color='blue', label=f'{name}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        ax.set_title(f'Coubre de calibration - {name}')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Fréquence observée')
        ax.legend()
        ax.grid(True)
        logger.log_figure(f'Calibration_{name}', fig)
        plt.close(fig)

        # Confusion matrice

        y_pred = model.predict(x_valid)
        cm = confusion_matrix(y_valid, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"Matrice de confusion - {name}")
        logger.log_figure(f'ConfusionMatrix_{name}', fig)
        plt.close(fig)

        # Log model

        metrics = model.evaluate(x_valid, y_valid)
        logger.log_metrics(metrics, as_summary=True)
        logger.log_hyperparameters(model.model.get_params())
        logger.log_model_artifact(model, f'LogReg_{name}', 'model', f'Regression_logistic_{name}')

        logger.finish()

    # Compute random baseline
    y_ref = y_valid_distance
    random_prob = np.random.uniform(0, 1, len(y_ref))
    fpr, tpr, _ = roc_curve(y_ref, random_prob)
    result["Baseline aléatoire"] = {
        'roc': (fpr, tpr, 0.5),
        'goal_rate': LRModel.get_goal_rate_by_percentile(y_ref, random_prob),
        'cumulative': LRModel.get_cumulative_goal(y_ref, random_prob),
        'calibration': LRModel.get_calibration(y_ref, random_prob)
    }
        

    #====================================================================================
    # Plot ALL the curve on one figure
    #====================================================================================

    # ROC curve

    fig_roc, ax = plt.subplots()
    for name, data in result.items():
        fpr, tpr, auc = data['roc']
        style = '--' if 'Baseline aléatoire' in name else '-'
        color = 'gray' if 'Baseline aléatoire' in name else None
        ax.plot(fpr, tpr, style, color=color, label=name)

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Courbe ROC')
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.grid(True)
    
    plt.tight_layout()
    ax.legend()
    plt.show()


    # Taux de but par centile

    fig_goal, ax = plt.subplots()
    for name, data in result.items():
        x, y = data['goal_rate']
        y_smooth = pd.Series(y).rolling(window=5, center=True, min_periods=1).mean()
        style = '--' if 'Baseline aléatoire' in name else '-'
        color = 'gray' if 'Baseline aléatoire' in name else None
        ax.plot(x, y_smooth, style, color=color, label=name)

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


    # Proprortion cumulée de buts

    fig_cum, ax = plt.subplots()
    for name, data in result.items():
        x, y = data['cumulative']
        style = '--' if 'Baseline aléatoire' in name else '-'
        color = 'gray' if 'Baseline aléatoire' in name else None
        ax.plot(x, y, style, color=color, label=name)

    ax.set_title(f'Proportion cumulée du buts')
    ax.set_xlabel('Propotion de tir')
    ax.set_ylabel('Proportion cumulée de buts')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1-x)*100:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


    # Calibration
    
    fig_cal, ax = plt.subplots()
    for name, data in result.items():
        x, y = data['calibration']
        style = '--' if 'Baseline aléatoire' in name else '-'
        color = 'gray' if 'Baseline aléatoire' in name else None
        ax.plot(x, y, style, color=color, label=name)

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Coubre de calibration - {name}')
    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence observée')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    summary_logger = WandbLogger(project_name="IFT6758.2025-A03", run_name="LR_summary_all_models_balanced")
    summary_logger.log_figure("ROC_All_Models", fig_roc)
    summary_logger.log_figure("GoalRate_All_Models", fig_goal)
    summary_logger.log_figure("CumulativeGoals_All_Models", fig_cum)
    summary_logger.log_figure("Calibration_All_Models", fig_cal)
    summary_logger.finish()