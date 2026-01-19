import os
import sys

sys.path.append(os.path.abspath("../.."))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ift6758.models.wandb_utils import WandbLogger
import xgboost as xgb


class XGBModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        return metrics
    
    @staticmethod
    def get_goal_rate_by_percentile(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df['centile'] = pd.qcut(df['y_prob'], q=100, labels=False, duplicates='drop')
        goal_rate = df.groupby('centile')['y_true'].mean()
        return goal_rate.index, goal_rate.values
    
    @staticmethod
    def get_cumulative_goal(y_true, y_prob):
        df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        df = df.sort_values('y_prob', ascending=False)
        df['CumulativeGoals'] = df['y_true'].cumsum()
        df['CumulativeRate'] = df['CumulativeGoals'] / df['y_true'].sum()
        df['ShotProportion'] = np.arange(1, len(df)+1) / len(df)
        return df['ShotProportion'], df['CumulativeRate']
    
    @staticmethod
    def get_calibration(y_true, y_prob, n_bins=10):
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        return prob_pred, prob_true
    
if __name__ == "__main__":
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    X = data[['Distance', 'Angle']]
    y = data['is_goal']

    #Split train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    #Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    #Create model dict for loop structure
    model = XGBModel()
    model.train(X_train_scaled, y_train)

    model_dict = {'XGB_Distance_Angle': (model, X_valid_scaled, y_valid)}

    result = {}

    #Compute curve data, figures, W&B logging
    for name, (model, x_valid, y_valid) in model_dict.items():
        logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'{name}_baseline')
        # ROC curve
        y_prob = model.predict_proba(x_valid)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)
        roc_auc = np.trapz(tpr, fpr)

        result[name]  = {
            'roc': (fpr, tpr, roc_auc),
            'goal_rate': XGBModel.get_goal_rate_by_percentile(y_valid, y_prob),
            'cumulative': XGBModel.get_cumulative_goal(y_valid, y_prob),
            'calibration': XGBModel.get_calibration(y_valid, y_prob) 
        }

        #ROC figure
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

        #Goal rate by percentile
        x, y_vals = result[name]['goal_rate']
        y_smooth = pd.Series(y_vals).rolling(window=5, center=True, min_periods=1).mean()
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

        #Cumulative goals
        x, y_vals = result[name]['cumulative']
        fig, ax = plt.subplots()
        ax.plot(x, y_vals, color='blue', label=f'{name}')
        ax.set_title(f'Proportion cumulee du buts - {name}')
        ax.set_xlabel('Proportion de tir')
        ax.set_ylabel('Proportion cumulee de buts')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1-x)*100:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.legend()
        ax.grid(True)
        logger.log_figure(f'CumulativeGoals_{name}', fig)
        plt.close(fig)

        #Calibration
        x, y_vals = result[name]['calibration']
        fig, ax = plt.subplots()
        ax.plot(x, y_vals, color='blue', label=f'{name}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        ax.set_title(f'Courbe de calibration - {name}')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Fréquence observée')
        ax.legend()
        ax.grid(True)
        logger.log_figure(f'Calibration_{name}', fig)
        plt.close(fig)

        # Metrics
        metrics = model.evaluate(x_valid, y_valid)
        logger.log_metrics(metrics, as_summary=True)
        logger.finish()

    #====================================================================================
    # Random baseline
    #====================================================================================
    y_ref = y_valid
    random_prob = np.random.uniform(0, 1, len(y_ref))
    fpr, tpr, _ = roc_curve(y_ref, random_prob)
    result["Baseline aléatoire"] = {
        'roc': (fpr, tpr, 0.5),
        'goal_rate': XGBModel.get_goal_rate_by_percentile(y_ref, random_prob),
        'cumulative': XGBModel.get_cumulative_goal(y_ref, random_prob),
        'calibration': XGBModel.get_calibration(y_ref, random_prob)
    }

    #====================================================================================
    # Combined figures for blog summary
    #====================================================================================
    # ROC
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
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Goal rate by percentile
    fig_goal, ax = plt.subplots()
    for name, data in result.items():
        x, y_vals = data['goal_rate']
        y_smooth = pd.Series(y_vals).rolling(window=5, center=True, min_periods=1).mean()
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

    # Cumulative goals
    fig_cum, ax = plt.subplots()
    for name, data in result.items():
        x, y_vals = data['cumulative']
        style = '--' if 'Baseline aléatoire' in name else '-'
        color = 'gray' if 'Baseline aléatoire' in name else None
        ax.plot(x, y_vals, style, color=color, label=name)
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
        x, y_vals = data['calibration']
        style = '--' if 'Baseline aléatoire' in name else '-'
        color = 'gray' if 'Baseline aléatoire' in name else None
        ax.plot(x, y_vals, style, color=color, label=name)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Courbe de calibration')
    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence observée')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

