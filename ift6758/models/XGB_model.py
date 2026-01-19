import os
import sys
sys.path.append(os.path.abspath("../.."))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, accuracy_score, precision_score, recall_score
import xgboost as xgb
from ift6758.models.wandb_utils import WandbLogger
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve

class XGBModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
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
    

if __name__ == '__main__':
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')

    #Logging Wandb
    logger = WandbLogger(project_name="IFT6758.2025-A03", run_name='XGB_all_features_tuned')

    data['is_rebound'] = data['is_rebound'].astype(int) #make bool column into 1 (True) and 0 (False)

    data = pd.get_dummies(data, columns=['Type of Shot'], prefix='shot') #one-hot encoding because nominal and not many categories
    data['last_event_type'] = data['last_event_type'].map({'Shot': 0, 'Goal': 1}) #label encoding for this one

    #Selectionne toutes les features sauf target
    X = data.drop(columns=['is_goal'])
    y = data['is_goal']

    #Split train/valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    #Dont need standardisation for XGB

    #Definit le model
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    #Definir la grille HP
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    #Grid search avec validation croisee
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    results_df = pd.DataFrame(grid_search.cv_results_)

    # Example: visualize performance vs max_depth
    fig, ax = plt.subplots()
    for lr in sorted(results_df['param_learning_rate'].unique()):
        subset = results_df[results_df['param_learning_rate'] == lr]
        ax.plot(subset['param_max_depth'], subset['mean_test_score'], marker='o', label=f'lr={lr}')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Mean CV ROC')
    ax.set_title('Effect of max_depth and learning_rate')
    ax.legend()
    logger.log_figure('Hyperparameter_effects', fig)
    plt.close(fig)

    #Meilleur model
    best_model = XGBModel()
    best_model.model = grid_search.best_estimator_
    best_model.train(X_train, y_train)

    #Evaluer sur validation
    metrics = best_model.evaluate(X_valid, y_valid)
    print(f'Validation metrics: {metrics}')

    logger.log_metrics(metrics, as_summary=True)

    #Generer les quatres figures pour le blog
    y_prob = best_model.predict_proba(X_valid)

    # ROC
    fpr, tpr, _ = roc_curve(y_valid, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'XGB all features tuned (AUC={np.trapz(tpr, fpr):.3f})')
    ax.plot([0,1],[0,1],'k--')
    ax.set_title('ROC Curve - XGB tuned')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(True)
    logger.log_figure('ROC_XGB_tuned', fig)
    plt.close(fig)

    # Goal rate by percentile
    x, y_vals = XGBModel.get_goal_rate_by_percentile(y_valid, y_prob)
    y_smooth = pd.Series(y_vals).rolling(window=5, center=True, min_periods=1).mean()
    fig, ax = plt.subplots()
    ax.plot(x, y_smooth)
    ax.invert_xaxis()
    ax.set_title('Goal rate by percentile')
    ax.set_xlabel('Probability percentile')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel('Goal rate')
    logger.log_figure('GoalRate_XGB_tuned', fig)
    plt.close(fig)

    # Cumulative goals
    x, y_vals = XGBModel.get_cumulative_goal(y_valid, y_prob)
    fig, ax = plt.subplots()
    ax.plot(x, y_vals)
    ax.set_title('Cumulative goals - XGB tuned')
    ax.set_xlabel('Proportion of shots')
    ax.set_ylabel('Cumulative proportion of goals')
    logger.log_figure('Cumulative_XGB_tuned', fig)
    plt.close(fig)

    # Calibration
    x, y_vals = XGBModel.get_calibration(y_valid, y_prob)
    fig, ax = plt.subplots()
    ax.plot(x, y_vals)
    ax.plot([0,1],[0,1],'k--')
    ax.set_title('Calibration curve - XGB tuned')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    logger.log_figure('Calibration_XGB_tuned', fig)
    plt.close(fig)

    logger.log_model_artifact(
        model=best_model.model,
        artifact_name='XGB_all_features_tuned',
        artifact_type='model',
        description=f"Best XGB model with params: {grid_search.best_params_}"
    )

    logger.finish()