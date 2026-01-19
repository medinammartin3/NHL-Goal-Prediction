import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from catboost import CatBoostClassifier, Pool

sys.path.append(os.path.abspath("../.."))
from ift6758.models.wandb_utils import WandbLogger


class CatBoostModel:
    def __init__(self, depth=6, learning_rate=0.05, iterations=1500, calibration=False):
        # Logger for Wandb run
        self.logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'CatBoostClassifier')

        # Model
        self.model = None
        self.calibrated_model = None

        # Imputers
        self.numerical_imputer = None
        self.categorical_imputer = None

        # HP
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.calibration = calibration
        self.best_params = None

        # Features
        self.cat_features_indices = None
        self.feature_names = None
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.selected_features = None


    def preprocess_data(self, X, fit=False):
        if fit:
            # Select features by type
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

        # Imputation for categorical features (no encoding needed - CatBoost handles automatically)
        if len(self.categorical_features) > 0:
            if fit:
                X_processed[self.categorical_features] = self.categorical_imputer.fit_transform(
                    X[self.categorical_features]
                )
            else:
                X_processed[self.categorical_features] = self.categorical_imputer.transform(
                    X[self.categorical_features]
                )

        return X_processed


    def grid_search(self, param_grid, X_train, y_train, cv=3):
        # Train data
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=self.cat_features_indices  # Categorical features indices
        )

        # Base model
        base_model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=3,
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            auto_class_weights='Balanced',
            early_stopping_rounds=50,
            eval_metric='AUC',
            task_type='CPU',
            thread_count=-1,
            random_seed=42,
            verbose=False
        )

        # Grid search for params optimization
        grid_result = base_model.grid_search(
            param_grid,
            X=train_pool,
            cv=cv,
            verbose=False,
            plot=False
        )

        # Update model HP with the best ones found on grid search
        self.best_params = grid_result['params']
        self.depth = self.best_params.get('depth', self.depth)
        self.learning_rate = self.best_params.get('learning_rate', self.learning_rate)
        self.iterations = self.best_params.get('iterations', self.iterations)


    def train(self, X_train, y_train, X_val, y_val, features_selection=False):
            self.numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

            self.cat_features_indices = [
                X_train.columns.get_loc(col) for col in self.categorical_features
            ]

            # Pool for training data
            train_pool = Pool(
                X_train,
                y_train,
                cat_features=self.cat_features_indices
            )

            # Pool for validation data
            val_pool = Pool(
                X_val,
                y_val,
                cat_features=self.cat_features_indices
            )

            self.model = CatBoostClassifier(
                # HP
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,

                # Regularisation
                l2_leaf_reg=3,

                # Bagging
                bootstrap_type='Bayesian',
                bagging_temperature=1,

                # Class balancing
                auto_class_weights='Balanced',

                # Early stopping
                early_stopping_rounds=50,

                # Metrics
                eval_metric='AUC',

                task_type='CPU',
                thread_count=-1,
                random_seed=42,
                verbose=False
            )

            # Fit model
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                verbose=100,
                plot=False
            )

            # Feature selection by importance
            if features_selection:
                # Compute feature importance
                feature_importance = self.model.get_feature_importance(train_pool)
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

                    train_pool_selected = Pool(
                        X_train_selected,
                        y_train,
                        cat_features=selected_cat_features_indices
                    )

                    val_pool_selected = Pool(
                        X_val_selected,
                        y_val,
                        cat_features=selected_cat_features_indices
                    )

                    self.model = CatBoostClassifier(
                        iterations=self.iterations,
                        learning_rate=self.learning_rate,
                        depth=self.depth,
                        l2_leaf_reg=3,
                        bootstrap_type='Bayesian',
                        bagging_temperature=1,
                        auto_class_weights='Balanced',
                        early_stopping_rounds=50,
                        eval_metric='AUC',
                        task_type='CPU',
                        thread_count=-1,
                        random_seed=42,
                        verbose=False
                    )

                    self.model.fit(
                        train_pool_selected,
                        eval_set=val_pool_selected,
                        use_best_model=True,
                        verbose=100,
                        plot=False
                    )

                    self.cat_features_indices = selected_cat_features_indices

            else:
                self.selected_features = X_train.columns.tolist()


            # Calibration
            if self.calibration:
                self.calibrated_model = CalibratedClassifierCV(
                    FrozenEstimator(self.model),
                    method='sigmoid',
                    cv="prefit"  # Avoid refitting the model (it's already trained)
                )

                # Fit calibrated model on validation set
                X_val_selected = X_val[self.selected_features]
                self.calibrated_model.fit(X_val_selected, y_val)



    def predict(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        X_selected = X_processed[self.selected_features]

        if self.calibration and self.calibrated_model is not None:
            return self.calibrated_model.predict(X_selected)
        else:
            return self.model.predict(X_selected)


    def predict_proba(self, X):
        X_processed = self.preprocess_data(X, fit=False)
        X_selected = X_processed[self.selected_features]

        if self.calibration and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_selected)[:, 1]
        else:
            return self.model.predict_proba(X_selected)[:, 1]


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

    name = 'CatBoostClassifier'
    use_calibration = True
    use_feature_selection = True

    print(name)

    # Load Dataset
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    y = data['is_goal']
    X = data.drop(columns=['is_goal'])

    # Data split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model
    model = CatBoostModel(depth=6, learning_rate=0.05, iterations=1500, calibration=use_calibration)

    # Process Data
    X_train_processed = model.preprocess_data(X_train, fit=True)
    X_valid_processed = model.preprocess_data(X_valid, fit=False)

    # Grid Search for HP tuning
    grid = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [1000, 1500, 2000],
    }
    model.cat_features_indices = [
        X_train_processed.columns.get_loc(column) for column in model.categorical_features
    ]
    model.grid_search(grid, X_train_processed, y_train, cv=3)

    # Train model with the best HP found
    model.train(X_train_processed, y_train, X_valid_processed, y_valid, features_selection=use_feature_selection)
    
    # Evaluate
    metrics = model.evaluate(X_valid, y_valid)
    print("Metrics:", metrics)


    """
    Plot the curves (individual figures)
    """

    # Get predictions
    fpr, tpr, roc_auc, y_prob = model.get_ROC(X_valid, y_valid)

    result = {
        name: {
            'roc': (fpr, tpr, roc_auc),
            'goal_rate': model.get_goal_rate_by_percentile(y_valid, y_prob),
            'cumulative': model.get_cumulative_goal(y_valid, y_prob),
            'calibration': model.get_calibration(y_valid, y_prob)
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
    model.logger.log_metrics(metrics, as_summary=True)
    selected_features = list(model.selected_features)
    model.logger.log_metrics({
        'n_selected_features': len(selected_features),
        'selected_features': selected_features
    })
    if model.calibration and model.calibrated_model is not None:
        model.logger.log_hyperparameters(model.calibrated_model.get_params())
    else:
        model.logger.log_hyperparameters(model.model.get_params())
    description = f'{name}: feature selection, calibration'
    model.logger.log_model_artifact(model, f'{name}', 'model', description)

    # Figures
    model.logger.log_figure(f'ROC_{name}', roc_fig)
    model.logger.log_figure(f'GoalRate_{name}', goal_rate_fig)
    model.logger.log_figure(f'CumulativeGoals_{name}', cumulative_goals_fig)
    model.logger.log_figure(f'Calibration_{name}', calibration_fig)
    model.logger.log_figure(f'Summary_{name}', summary_fig)

    plt.close(roc_fig)
    plt.close(goal_rate_fig)
    plt.close(cumulative_goals_fig)
    plt.close(calibration_fig)
    plt.close(summary_fig)

    model.logger.finish()