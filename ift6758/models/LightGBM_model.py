import os
import sys
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath("../.."))
from ift6758.models.wandb_utils import WandbLogger


class LightGBMModel:
    def __init__(self, n_estimators=2000, learning_rate=0.05, num_leaves=31, max_depth=7, boosting_type='gbdt',
                 reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20, bagging_fraction=0.8, feature_fraction=0.8,
                 use_logger=True):
        # Logger for Wandb run
        self.logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'LightGBM') if use_logger else None

        # HP
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.boosting_type = boosting_type
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.bagging_fraction = bagging_fraction
        self.feature_fraction = feature_fraction
        self.train_params = None

        # Model
        self.model = None
        self.best_iteration = None

        # Features
        self.label_encoders = {}
        self.numerical_features = None
        self.categorical_features = None
        self.categorical_feature_indices = None
        self.selected_features = None

        # Imputers
        self.numerical_imputer = None
        self.categorical_imputer = None


    def preprocess_data(self, X, fit=False):
        if fit:
            # Get features by type
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

            # Imputers
            self.numerical_imputer = SimpleImputer(strategy="median")
            self.categorical_imputer = SimpleImputer(strategy="constant", fill_value="unknown")

        X_processed = X.copy()

        # Imputation for numerical features
        if len(self.numerical_features) > 0:
            if fit:
                X_processed[self.numerical_features] = self.numerical_imputer.fit_transform(
                    X[self.numerical_features]
                )
            else:
                X_processed[self.numerical_features] = self.numerical_imputer.transform(
                    X[self.numerical_features]
                )

        # Imputation for categorical features
        if len(self.categorical_features) > 0:
            if fit:
                X_processed[self.categorical_features] = self.categorical_imputer.fit_transform(
                    X[self.categorical_features]
                )

                # Label encoding
                for column in self.categorical_features:
                    encoder = LabelEncoder()
                    X_processed[column] = encoder.fit_transform(X_processed[column].astype(str))
                    self.label_encoders[column] = encoder

                # Categorical features indices
                self.categorical_feature_indices = [
                    X_processed.columns.get_loc(col) for col in self.categorical_features
                ]

            else:
                X_processed[self.categorical_features] = self.categorical_imputer.transform(
                    X[self.categorical_features]
                )

                # Label encoding
                for col in self.categorical_features:
                    encoder = self.label_encoders[col]
                    # New categories (not seen in training) --> -1, else --> encoder
                    X_processed[col] = X_processed[col].astype(str).apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )

        return X_processed



    def train(self, X_train, y_train, X_val, y_val, features_selection=False):

        # Compute class imbalance weight for output (is_goal)
        class_counts = np.bincount(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1]

        # Training dataset
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.categorical_feature_indices,
            free_raw_data=False
        )

        # Validation dataset
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=self.categorical_feature_indices,
            reference=train_data,
            free_raw_data=False
        )

        # Params for different boosting types
        if self.boosting_type == 'dart':
            # DART
            params = {
                'objective': 'binary',
                'boosting_type': 'dart',
                'metric': ['auc'],

                # Architecture
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,

                # DART-specific
                'drop_rate': 0.1,
                'max_drop': 50,
                'skip_drop': 0.5,
                'uniform_drop': False,

                # Regularisation
                'min_child_samples': self.min_child_samples,
                'min_child_weight': 0.001,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,

                # Sampling
                'bagging_fraction':self.bagging_fraction,
                'bagging_freq': 5,
                'feature_fraction': self.feature_fraction,

                # Class weight
                'scale_pos_weight': scale_pos_weight,
                'is_unbalance': False,

                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            # GBDT
            params = {
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'metric': ['auc'],

                # Architecture
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,

                # Regularisation
                'min_child_samples': self.min_child_samples,
                'min_child_weight': 0.001,
                'min_split_gain': 0.0,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,

                # Sampling (bagging + feature sampling)
                'bagging_fraction': self.bagging_fraction,
                'bagging_freq': 5,
                'feature_fraction': self.feature_fraction,
                'feature_fraction_bynode': 0.8,

                # Class weight
                'scale_pos_weight': scale_pos_weight,
                'is_unbalance': False,

                'max_bin': 255,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }

        self.train_params = params

        # Early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]

        # Train model
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks  # Early stopping
        )

        # Feature selection by importance
        if features_selection:
            # Compute feature importance (importance gain)
            feature_importance = self.model.feature_importance(importance_type='gain')
            feature_names = X_train.columns.tolist()

            # Sort features by importance (most-->less)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            # Keep features contributing to 95% of total importance
            importance_sum = importance_df['importance'].cumsum()
            total = importance_df['importance'].sum()
            threshold_idx = (importance_sum / total >= 0.95).idxmax()
            self.selected_features = importance_df.iloc[:threshold_idx + 1]['feature'].tolist()

            # Re-train model with selected features
            if len(self.selected_features) < len(feature_names):

                X_train_selected = X_train[self.selected_features]
                X_val_selected = X_val[self.selected_features]

                selected_cat_features_indices = [
                    self.selected_features.index(col)
                    for col in self.categorical_features
                    if col in self.selected_features
                ]

                train_data_selected = lgb.Dataset(
                    X_train_selected,
                    label=y_train,
                    categorical_feature=selected_cat_features_indices,
                    free_raw_data=False
                )

                val_data_selected = lgb.Dataset(
                    X_val_selected,
                    label=y_val,
                    categorical_feature=selected_cat_features_indices,
                    reference=train_data_selected,
                    free_raw_data=False
                )

                self.model = lgb.train(
                    params,
                    train_data_selected,
                    num_boost_round=self.n_estimators,
                    valid_sets=[train_data_selected, val_data_selected],
                    valid_names=['train', 'valid'],
                    callbacks=callbacks
                )

                self.categorical_feature_indices = selected_cat_features_indices

        else:
            self.selected_features = X_train.columns.tolist()

        self.best_iteration = self.model.best_iteration


    def predict(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        X_selected = X_processed[self.selected_features]

        predictions = self.model.predict(X_selected)
        return (predictions > 0.5).astype(int)

    def predict_proba(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        X_selected = X_processed[self.selected_features]

        return self.model.predict(X_selected, num_iteration=self.best_iteration)

    def evaluate(self, X_valid, y_valid):
        y_pred = self.predict(X_valid)
        y_prob = self.predict_proba(X_valid)

        metrics = {
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred),
            "recall": recall_score(y_valid, y_pred),
            "f1_score": f1_score(y_valid, y_pred),
            "roc_auc": roc_auc_score(y_valid, y_prob)
        }
        return metrics


    def get_ROC(self, X_valid, y_valid):
        y_prob = self.predict_proba(X_valid)
        fpr, tpr, _ = roc_curve(y_valid, y_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc, y_prob

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
        df['But cumulatifs'] = df['y_true'].cumsum()
        df['Ratio cumulatifs'] = df['But cumulatifs'] / df['y_true'].sum()
        df['Proportion de tirs'] = np.arange(1, len(df) + 1) / len(df)
        return df['Proportion de tirs'], df['Ratio cumulatifs']

    @staticmethod
    def get_calibration(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        return prob_pred, prob_true


def bayesian_optimization(X_train, y_train, X_val, y_val, n_trials=50):

    def objective(trial):
        # Params and values to test
        params = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=500)
        }

        # Create model with suggested params
        model = LightGBMModel(
            boosting_type=params['boosting_type'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            max_depth=params['max_depth'],
            min_child_samples=params['min_child_samples'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            bagging_fraction=params['bagging_fraction'],
            feature_fraction=params['feature_fraction'],
            use_logger=False
        )

        # Process Data
        X_train_processed = model.preprocess_data(X_train, fit=True)
        X_valid_processed = model.preprocess_data(X_valid, fit=False)

        # Train
        try:
            model.train(X_train_processed, y_train, X_valid_processed, y_val, features_selection=False)

            # Evaluate on validation set
            metrics = model.evaluate(X_val, y_val)

            return metrics['roc_auc']

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    # Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study


if __name__ == "__main__":
    name = 'LightGBM'
    use_feature_selection = False

    print(name)

    # Load Dataset
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    y = data['is_goal']
    X = data.drop(columns=['is_goal'])

    # Data split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Bayesian HP optimization
    best_params, study = bayesian_optimization(
        X_train, y_train,
        X_valid, y_valid,
        n_trials=50
    )

    # Final model with optimised params
    final_model = LightGBMModel(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        num_leaves=best_params['num_leaves'],
        max_depth=best_params['max_depth'],
        min_child_samples=best_params['min_child_samples'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        bagging_fraction=best_params['bagging_fraction'],
        feature_fraction=best_params['feature_fraction'],
        boosting_type=best_params['boosting_type'],
    )

    # Process Data
    X_train_processed = final_model.preprocess_data(X_train, fit=True)
    X_valid_processed = final_model.preprocess_data(X_valid, fit=False)

    # Train
    final_model.train(
        X_train_processed, y_train,
        X_valid_processed, y_valid,
        features_selection=use_feature_selection
    )

    # Evaluate
    metrics = final_model.evaluate(X_valid, y_valid)
    print("Metrics:", metrics)

    """
    Plot the curves (individual figures)
    """

    # Get predictions
    fpr, tpr, roc_auc, y_prob = final_model.get_ROC(X_valid, y_valid)

    result = {
        name: {
            'roc': (fpr, tpr, roc_auc),
            'goal_rate': final_model.get_goal_rate_by_percentile(y_valid, y_prob),
            'cumulative': final_model.get_cumulative_goal(y_valid, y_prob),
            'calibration': final_model.get_calibration(y_valid, y_prob)
        }
    }

    # --- ROC ---
    roc_fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f'Courbe ROC - {name}')
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend()
    ax.grid(True)

    # --- Goal Rate by Percentile ---
    x_g, y_g = result[name]['goal_rate']
    y_smooth = pd.Series(y_g).rolling(window=5, center=True, min_periods=1).mean()

    goal_rate_fig, ax = plt.subplots()
    ax.plot(x_g, y_smooth, label=name)
    ax.invert_xaxis()
    ax.set_title(f'Taux de buts par centile - {name}')
    ax.set_xlabel('Centile de probabilité')
    ax.set_ylabel('Taux de buts')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # --- Cumulative Goals ---
    x_c, y_c = result[name]['cumulative']

    cumulative_goals_fig, ax = plt.subplots()
    ax.plot(x_c, y_c, label=name)
    ax.set_title(f'Proportion cumulée des buts - {name}')
    ax.set_xlabel('Proportion de tirs')
    ax.set_ylabel('Proportion de buts cumulée')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1 - x) * 100:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.legend()
    ax.grid(True)

    # --- Calibration / Reliability ---
    x, y = result[name]['calibration']

    calibration_fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', label=name)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    ax.set_title(f'Courbe de calibration/fiabilité - {name}')
    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence observée')
    ax.legend()
    ax.grid(True)

    """
    Plot the curves (unique figure)
    """

    summary_fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # --- ROC Curve ---
    fpr, tpr, roc_auc = result[name]['roc']
    axs[0, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    axs[0, 0].plot([0, 1], [0, 1], 'k--')
    axs[0, 0].set_title('Courbe ROC')
    axs[0, 0].set_xlabel("Taux de faux positifs")
    axs[0, 0].set_ylabel("Taux de vrais positifs")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- Goal Rate by Percentile ---
    x_g, y_g = result[name]['goal_rate']
    y_smooth = pd.Series(y_g).rolling(window=5, center=True, min_periods=1).mean()
    axs[0, 1].plot(x_g, y_smooth, label=name)
    axs[0, 1].invert_xaxis()
    axs[0, 1].set_title('Taux de buts par centile')
    axs[0, 1].set_xlabel('Centile')
    axs[0, 1].set_ylabel('Taux de buts')
    axs[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- Cumulative Goals ---
    x_c, y_c = result[name]['cumulative']
    axs[1, 0].plot(x_c, y_c, label=name)
    axs[1, 0].set_title('Proportion cumulée de buts')
    axs[1, 0].set_xlabel('Proportion de tirs')
    axs[1, 0].set_ylabel('Proportion cumulée de buts')
    axs[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(1 - x) * 100:.0f}"))
    axs[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Calibration / Reliability ---
    x, y = result[name]['calibration']
    axs[1, 1].plot(x, y, color='blue', label=name)
    axs[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.7)
    axs[1, 1].set_title(f'Courbe de calibration/fiabilité - {name}')
    axs[1, 1].set_xlabel('Probabilité prédite')
    axs[1, 1].set_ylabel('Fréquence observée')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


    # --- Log metrics and figures ---
    # Metrics
    final_model.logger.log_metrics(metrics, as_summary=True)
    selected_features = list(final_model.selected_features)
    final_model.logger.log_metrics({
        'n_selected_features': len(selected_features),
        'selected_features': selected_features
    })
    final_model.logger.log_hyperparameters(final_model.train_params)
    description = f'{name}: calibration, no feature selection'
    final_model.logger.log_model_artifact(final_model, f'{name}', 'model', description)

    # Figures
    final_model.logger.log_figure(f'ROC_{name}', roc_fig)
    final_model.logger.log_figure(f'GoalRate_{name}', goal_rate_fig)
    final_model.logger.log_figure(f'CumulativeGoals_{name}', cumulative_goals_fig)
    final_model.logger.log_figure(f'Calibration_{name}', calibration_fig)
    final_model.logger.log_figure(f'Summary_{name}', summary_fig)

    plt.close(roc_fig)
    plt.close(goal_rate_fig)
    plt.close(cumulative_goals_fig)
    plt.close(calibration_fig)
    plt.close(summary_fig)

    final_model.logger.finish()