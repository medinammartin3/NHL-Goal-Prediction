import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath("../.."))
from ift6758.models.wandb_utils import WandbLogger


class MLPModel:
    def __init__(self, hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=0.001):

        # Logger for Wandb run
        self.logger = WandbLogger(project_name="IFT6758.2025-A03", run_name=f'MLPClassifier')

        # MLP base model
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=500,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            random_state=42,
        )

        self.pipeline = None

    def train(self, X_train, y_train):
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

        # Scaling and imputing (missing values treatment) for numerical features
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Scaling and imputing (missing values treatment) for categorical features
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Apply the transformations to the features
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        # Imbalanced pipeline for SMOTE over-sampling
        self.pipeline = ImbPipeline([
            ("preprocess", preprocessor),   # Preprocess the data with the transformations
            ("smote", SMOTE(random_state=42)),   # SMOTE
            ("classifier", self.model)    # Model
        ])

        # Train
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

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

    name = 'MLPClassifier'

    print(name)

    # Load Dataset
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    y = data['is_goal']
    X = data.drop(columns=['is_goal'])

    # Data split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    # ----- Cross Validation for Hyperparameter Tuning ----

    # Base model
    base_model = MLPClassifier(
        max_iter=400,
        random_state=42,
        early_stopping=True,
    )

    # Features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Scaling and imputing transformers
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])
    #Preprocessor applying the transformations
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    # SMOTE with preprocessing
    pipeline = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", base_model)
    ])
    # Params to test with grid search
    param_grid = {
        "classifier__hidden_layer_sizes": [(128, 64, 32), (256, 128, 64), (128, 64, 31, 16), (256, 128, 64, 32)],
        "classifier__alpha": [1e-5, 1e-4, 1e-3],
        "classifier__learning_rate_init": [0.001, 0.005, 0.01],
        "classifier__activation": ["tanh"],
        "classifier__learning_rate": ["adaptive"]
    }
    # Grid search for HP optimizations
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    best_pipeline = grid.best_estimator_  # Best pipeline found

    # Calibration of best pipeline
    best_mlp = best_pipeline.named_steps["classifier"]
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(best_mlp),
        method="isotonic",
        cv="prefit"  # Avoid fitting again the model (it's already trained)
    )
    # Process validation data for calibrated model fitting
    X_valid_preprocessed = best_pipeline.named_steps['preprocess'].transform(X_valid)
    calibrated_model.fit(X_valid_preprocessed, y_valid)

    # Replace pipeline for calibrated one
    final_pipeline = best_pipeline
    final_pipeline.steps[-1] = ("classifier", calibrated_model)

    # Final trained model
    final_model = MLPModel()
    final_model.pipeline = final_pipeline

    # Evaluate model
    metrics = final_model.evaluate(X_valid, y_valid)


    """
    Plot the curves (individual figures)
    """

    # Get predictions
    fpr, tpr, roc_auc, y_prob = final_model.get_ROC(X_valid, y_valid)

    result = {
        name: {
            'roc': (fpr, tpr, roc_auc, y_prob),
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
    fpr, tpr, roc_auc, _ = result[name]['roc']
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
    calibrated_clf = final_pipeline.named_steps["classifier"]
    final_model.logger.log_hyperparameters(best_mlp.get_params())
    description = f'{name} pipeline with calibration'
    final_model.logger.log_model_artifact(final_model.pipeline, f'{name}', 'model', description)

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