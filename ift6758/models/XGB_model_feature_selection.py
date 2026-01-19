import os
import sys
sys.path.append(os.path.abspath("../.."))
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, roc_curve, auc, accuracy_score, 
    precision_score, recall_score, roc_auc_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.calibration import calibration_curve
from ift6758.models.wandb_utils import WandbLogger
import joblib
from scipy.stats import spearmanr

def goal_rate_by_percentile(y_true, y_prob, q=100):
    df = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(y_prob)})
    # Bin predictions into percentiles
    df['centile'] = pd.qcut(df['p'], q=q, labels=False, duplicates='drop')
    #Compute goal rate per percentile
    gr = df.groupby('centile')['y'].mean()
    return gr.index.values, gr.values

def cumulative_goals(y_true, y_prob):
    df = pd.DataFrame({'y': np.asarray(y_true), 'p': np.asarray(y_prob)})
    #Sort by predicted probability descending 
    df = df.sort_values('p', ascending=False).reset_index(drop=True)
    #Cumulative goals scored as we include more shots
    df['cum_goals'] = df['y'].cumsum()
    total_goals = df['y'].sum() if df['y'].sum() > 0 else 1.0
    df['cum_rate'] = df['cum_goals'] / total_goals
    #Proportion of shots included
    df['shot_prop'] = (np.arange(len(df)) + 1) / len(df)
    return df['shot_prop'].values, df['cum_rate'].values

#Feature selection functions

def remove_highly_correlated_features(X, threshold=0.90):
    # Remove features with correlation > threshold using Spearman correlation.
    
    print(f"\n[Correlation Filter] Starting with {X.shape[1]} features...")
    
    # Calculate Spearman correlation 
    # For each pair, calculate correlation on non-NaN overlap
    corr_matrix = X.corr(method='spearman').abs()
    
    # Find pairs with high correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    keep_cols = [col for col in X.columns if col not in to_drop]
    
    print(f"[Correlation Filter] Removed {len(to_drop)} features: {to_drop}")
    print(f"[Correlation Filter] Keeping {len(keep_cols)} features")
    
    return keep_cols, corr_matrix

def xgb_feature_importance_selection(X_train, y_train, X_valid, y_valid, 
                                     threshold_percentile=50, scale_pos_weight=None):
    #Use XGBoost's built-in feature importance to select features.
    print(f"\n[XGB Feature Importance] Training initial model...")
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Get feature importances (gain-based)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Calculate threshold
    threshold_value = np.percentile(importances, threshold_percentile)
    selected_features = importance_df[importance_df['importance'] >= threshold_value]['feature'].tolist()
    
    print(f"[XGB Feature Importance] Importance threshold: {threshold_value:.4f}")
    print(f"[XGB Feature Importance] Selected {len(selected_features)}/{len(X_train.columns)} features")
    print(f"[XGB Feature Importance] Top 5: {selected_features[:5]}")
    
    return selected_features, model, importance_df

#Pas utilise finalement
def recursive_feature_elimination_xgb(X_train, y_train, n_features_to_select=10,
                                     scale_pos_weight=None, cv_folds=3):
    # Custom RFE for XGBoost with cross-validation.
    print(f"\n[RFE] Starting with {X_train.shape[1]} features, targeting {n_features_to_select}...")
    
    current_features = X_train.columns.tolist()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    feature_ranking = []
    
    while len(current_features) > n_features_to_select:
        # Train model with current features
        X_curr = X_train[current_features]
        
        # Cross-validation to get stable feature importances
        importances_per_fold = []
        scores_per_fold = []
        
        for train_idx, val_idx in cv.split(X_curr, y_train):
            X_fold_train = X_curr.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_curr.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )

            model.fit(X_fold_train, y_fold_train)
            importances_per_fold.append(model.feature_importances_)

            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            score = roc_auc_score(y_fold_val, y_pred_proba)
            scores_per_fold.append(score)

        # Average importance across folds
        avg_importances = np.mean(importances_per_fold, axis=0)
        avg_score = np.mean(scores_per_fold)
        
        # Find least important feature
        least_important_idx = np.argmin(avg_importances)
        least_important_feature = current_features[least_important_idx]
        
        feature_ranking.append({
            'feature': least_important_feature,
            'rank': len(current_features),
            'importance': avg_importances[least_important_idx],
            f'cv_ROC': avg_score
        })
        
        # Remove least important feature
        current_features.remove(least_important_feature)
        
        if len(current_features) % 5 == 0:
            print(f"[RFE] {len(current_features)} features remaining, ROC={avg_score:.4f}")
    
    print(f"[RFE] Final {len(current_features)} features selected")
    return current_features, pd.DataFrame(feature_ranking)


def nested_cv_feature_selection(X, y, feature_selection_method='importance', 
                                n_outer=5, n_inner=3):
    # Nested cross-validation to evaluate feature selection stability.
    print(f"\n[Nested CV] Evaluating feature selection stability...")
    
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    feature_counts = pd.Series(0, index=X.columns) #count how many times each feature is selected across folds
    
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y.iloc[train_idx]
        
        # Calculate scale_pos_weight for this fold
        pos = (y_train_outer == 1).sum()
        neg = (y_train_outer == 0).sum()
        scale_pos_weight = neg / max(1, pos) #compute class imbalance per fold
        
        # Feature selection on this fold
        if feature_selection_method == 'importance':
            selected, _, _ = xgb_feature_importance_selection(
                X_train_outer, y_train_outer, 
                X.iloc[val_idx], y.iloc[val_idx],
                threshold_percentile=40,
                scale_pos_weight=scale_pos_weight
            )
        
        feature_counts[selected] += 1
        print(f"[Nested CV] Fold {fold}: {len(selected)} features selected")
    
    # Calculate stability (proportion of folds selecting each feature)
    stability = feature_counts / n_outer
    
    return stability.sort_values(ascending=False)

# Hyperparamter tuning
def grid_search_xgb_manual(X_train, y_train, cv_folds=5, scale_pos_weight=None):
    """
    Manual grid search with cross-validation for XGBoost.
    """
    print(f"\n[Grid Search] Starting hyperparameter search...")
    
    param_grid = {
        'n_estimators': [100, 300, 400],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.5, 1],
        'min_child_weight': [1, 3, 5]
    }
    
    # Generate parameter combinations (sample to keep it manageable)
    from itertools import product
    import random
    
    # All combinations
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    
    best_score = -np.inf
    best_params = None
    results = []
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    #evaluate each parameter combination
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        params['scale_pos_weight'] = scale_pos_weight
        params['random_state'] = 42
        params['eval_metric'] = 'logloss'
        
        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train)

            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

            score = roc_auc_score(y_fold_val, y_pred_proba)
            cv_scores.append(score)
        
        #Compute average performance
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        results.append({
            'params': params,
            'mean_ROC': mean_score,
            'std_ROC': std_score
        })
        
        #Track best configuration and score
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        
        if i % 10 == 0:
            print(f"[Grid Search] Completed {i}/{len(combinations)} combinations...")
    
    print(f"[Grid Search] Best CV ROC: {best_score:.4f}")
    print(f"[Grid Search] Best params: {best_params}")
    
    return best_params, best_score, pd.DataFrame(results)


if __name__ == '__main__':
    logger = WandbLogger(project_name="IFT6758.2025-A03", run_name='XGB_feature_selection_hp')

    #Load an prepare data
    data = pd.read_csv('../../games_data/feature_dataset_2_train.csv')
    data['is_rebound'] = data['is_rebound'].astype(int)
    data = pd.get_dummies(data, columns=['Type of Shot'], prefix='shot')
    data = pd.get_dummies(data, columns=['last_event_type'])
    data = data.dropna(subset=['is_goal'])

    #Split X/y
    X = data.drop(columns=['is_goal'])
    y = data['is_goal'].astype(int)

    # Train/valid split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Compute class imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(1, pos)

    #Correlation based filter
    corr_matrix_initial = X_train.corr(method='spearman').abs()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix_initial, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Absolute Correlation'})
    ax.set_title('Initial Correlation Matrix (Spearman)', fontsize=14)
    plt.tight_layout()
    logger.log_figure('01_correlation_initial', fig)
    plt.close(fig)
    
    # Remove highly correlated features
    keep_after_corr, corr_matrix = remove_highly_correlated_features(X_train, threshold=0.90)
    X_train_filtered = X_train[keep_after_corr]
    X_valid_filtered = X_valid[keep_after_corr]
    
    logger.log_metrics({'n_features_after_correlation': len(keep_after_corr)})
    
    # Plot after correlation filtering
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(X_train_filtered.corr(method='spearman').abs(), cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix After Filtering', fontsize=14)
    plt.tight_layout()
    logger.log_figure('02_correlation_filtered', fig)
    plt.close(fig)

    # XGB feature importance selection
    selected_features, initial_model, importance_df = xgb_feature_importance_selection(
        X_train_filtered, y_train,
        X_valid_filtered, y_valid,
        threshold_percentile=30,  # Keep top 70% of features
        scale_pos_weight=scale_pos_weight
    )
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = importance_df.head(20)
    ax.barh(range(len(top_20)), top_20['importance'])
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'])
    ax.set_xlabel('Importance (Gain)')
    ax.set_title('Top 20 Feature Importances (XGBoost)')
    ax.invert_yaxis()
    plt.tight_layout()
    logger.log_figure('03_feature_importance', fig)
    plt.close(fig)
    
    # Update datasets
    X_train_selected = X_train_filtered[selected_features]
    X_valid_selected = X_valid_filtered[selected_features]
    
    logger.log_metrics({'n_features_after_importance': len(selected_features)})
    
    #Nested CV for stability
    stability_scores = nested_cv_feature_selection(
        X_train_filtered, y_train,
        feature_selection_method='importance',
        n_outer=5
    )
    
    # Plot stability scores
    fig, ax = plt.subplots(figsize=(10, 8))
    top_stable = stability_scores.head(25)
    ax.barh(range(len(top_stable)), top_stable.values)
    ax.set_yticks(range(len(top_stable)))
    ax.set_yticklabels(top_stable.index)
    ax.set_xlabel('Selection Frequency (proportion of folds)')
    ax.set_title('Top 25 Most Stable Features Across CV Folds')
    ax.invert_yaxis()
    ax.axvline(x=0.6, color='r', linestyle='--', label='Stability threshold (0.6)')
    ax.legend()
    plt.tight_layout()
    logger.log_figure('04_feature_stability', fig)
    plt.close(fig)
    
    # Select only stable features (present in >= 60% of folds)
    stable_features = stability_scores[stability_scores >= 0.6].index.tolist()
    
    if len(stable_features) < 8:  # Ensure minimum number of features
        stable_features = stability_scores.head(10).index.tolist()
    
    print(f"Stable features selected: {stable_features}")
    
    X_train_stable = X_train_filtered[stable_features]
    X_valid_stable = X_valid_filtered[stable_features]
    
    logger.log_metrics({
        'n_features_stable': len(stable_features),
        'stable_features': stable_features
    })

    #Hyperparameter tuning for stable features
    best_params, best_cv_score, search_results = grid_search_xgb_manual(
        X_train_stable, y_train,
        cv_folds=5,
        scale_pos_weight=scale_pos_weight
    )
    
    logger.log_metrics({
        'best_cv_ROC': float(best_cv_score),
        **{f'best_param_{k}': v for k, v in best_params.items() if k != 'random_state'}
    })
    
    # Plot hyperparameter effects
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Effect of max_depth
    ax = axes[0, 0]
    for lr in search_results['params'].apply(lambda x: x['learning_rate']).unique():
        subset = search_results[search_results['params'].apply(lambda x: x['learning_rate']) == lr]
        depths = subset['params'].apply(lambda x: x['max_depth'])
        ax.plot(depths, subset['mean_ROC'], marker='o', label=f'lr={lr}')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Mean CV ROC')
    ax.set_title('Effect of max_depth and learning_rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect of n_estimators
    ax = axes[0, 1]
    for md in search_results['params'].apply(lambda x: x['max_depth']).unique():
        subset = search_results[search_results['params'].apply(lambda x: x['max_depth']) == md]
        n_est = subset['params'].apply(lambda x: x['n_estimators'])
        ax.plot(n_est, subset['mean_ROC'], marker='o', label=f'depth={md}')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Mean CV ROC')
    ax.set_title('Effect of n_estimators and max_depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect of subsample
    ax = axes[1, 0]
    for ct in search_results['params'].apply(lambda x: x['colsample_bytree']).unique():
        subset = search_results[search_results['params'].apply(lambda x: x['colsample_bytree']) == ct]
        ss = subset['params'].apply(lambda x: x['subsample'])
        ax.plot(ss, subset['mean_ROC'], marker='o', label=f'colsample={ct}')
    ax.set_xlabel('subsample')
    ax.set_ylabel('Mean CV ROC')
    ax.set_title('Effect of subsample and colsample_bytree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution of CV scores
    ax = axes[1, 1]
    ax.hist(search_results['mean_ROC'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(best_cv_score, color='r', linestyle='--', linewidth=2, label=f'Best: {best_cv_score:.4f}')
    ax.set_xlabel('Mean CV ROC Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Hyperparameter Configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.log_figure('05_hyperparameter_effects', fig)
    plt.close(fig)

    #Train final model
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_stable, y_train)
    
    # Predictions
    y_valid_prob = final_model.predict_proba(X_valid_stable)[:, 1]
    y_valid_pred = final_model.predict(X_valid_stable)
    
    # Calculate metrics
    accuracy = accuracy_score(y_valid, y_valid_pred)
    precision = precision_score(y_valid, y_valid_pred)
    recall = recall_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    roc_auc = roc_auc_score(y_valid, y_valid_prob)
    
    print(f"\nFinal Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    logger.log_metrics({
        'final_accuracy': float(accuracy),
        'final_precision': float(precision),
        'final_recall': float(recall),
        'final_f1': float(f1),
        'final_roc_auc': float(roc_auc)
    }, as_summary=True)

    #Performance figures
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'XGB (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Final XGB Model with Feature Selection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    logger.log_figure('06_ROC', fig)
    plt.close(fig)
    
    # Goal Rate by Percentile
    x_cent, y_cent = goal_rate_by_percentile(y_valid, y_valid_prob, q=100)
    y_smooth = pd.Series(y_cent).rolling(window=5, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_cent, y_smooth, marker='o', linewidth=2, markersize=4)
    ax.invert_xaxis()
    ax.set_xlabel('Shot Probability Model Percentile', fontsize=12)
    ax.set_ylabel('Goal Rate', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title('Goal Rate by Shot Probability Percentile', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    logger.log_figure('07_goal_rate_percentile', fig)
    plt.close(fig)
    
    # Cumulative Goals
    x_cum, y_cum = cumulative_goals(y_valid, y_valid_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_cum, y_cum, linewidth=2)
    ax.set_xlabel('Proportion of Shots', fontsize=12)
    ax.set_ylabel('Cumulative Proportion of Goals', fontsize=12)
    ax.set_title('Cumulative % of Goals vs Cumulative % of Shots', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    logger.log_figure('08_cumulative_goals', fig)
    plt.close(fig)
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_valid, y_valid_prob, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='XGB Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    logger.log_figure('09_calibration', fig)
    plt.close(fig)

    logger.log_model_artifact(
        model=final_model,
        artifact_name='XGB_features_and_hp_tuned',
        artifact_type='model'
    )

    logger.finish()
