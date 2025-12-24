"""
Nested Cross-Validation with Oversampling Inside Folds

This module implements proper nested CV where oversampling (SMOTE/ROS) is applied
INSIDE each fold to prevent data leakage. This gives unbiased performance estimates.

Structure:
- Outer loop (10-fold): For performance estimation
- Inner loop (5-fold): For hyperparameter tuning
- Oversampling applied fresh in each fold via imblearn Pipeline
"""

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import numpy as np


def create_pipeline(classifier, oversampler='smote'):
    """
    Create pipeline with scaling + oversampling + classifier.
    Oversampling is applied fresh in each CV fold (no data leakage).

    Args:
        classifier: sklearn classifier instance
        oversampler: 'smote' or 'ros' (random oversampling)
    """
    if oversampler == 'smote':
        sampler = SMOTE(random_state=42)
    elif oversampler == 'ros':
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError(f"Unknown oversampler: {oversampler}")

    return ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('sampler', sampler),
        ('classifier', classifier)
    ])


def nested_cv_with_tuning(X, y, classifier, param_grid, oversampler='smote',
                          outer_folds=10, inner_folds=5):
    """
    Nested CV: outer loop for evaluation, inner loop for hyperparameter tuning.
    Oversampling is applied fresh inside each fold (no data leakage).

    Args:
        X: Features (numpy array or DataFrame)
        y: Target (numpy array or Series)
        classifier: sklearn classifier instance (unfit)
        param_grid: dict with keys prefixed by 'classifier__'
        oversampler: 'smote' or 'ros'
        outer_folds: Number of folds for performance estimation
        inner_folds: Number of folds for hyperparameter tuning

    Returns:
        dict with mean scores, std, and per-fold scores
    """
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)

    pipeline = create_pipeline(classifier, oversampler)

    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring='f1',
        n_jobs=-1,
        refit=True
    )

    # Outer CV for unbiased performance estimation
    # Using cross_validate to get multiple metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(
        grid_search, X, y,
        cv=outer_cv,
        scoring=scoring,
        return_train_score=True,  # Also get train scores for overfitting detection
        n_jobs=-1
    )

    return {
        # Test metrics (outer fold validation sets)
        'test_accuracy_mean': cv_results['test_accuracy'].mean(),
        'test_accuracy_std': cv_results['test_accuracy'].std(),
        'test_precision_mean': cv_results['test_precision'].mean(),
        'test_precision_std': cv_results['test_precision'].std(),
        'test_recall_mean': cv_results['test_recall'].mean(),
        'test_recall_std': cv_results['test_recall'].std(),
        'test_f1_mean': cv_results['test_f1'].mean(),
        'test_f1_std': cv_results['test_f1'].std(),

        # Train metrics (for overfitting detection)
        'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        'train_f1_mean': cv_results['train_f1'].mean(),

        # Per-fold scores for detailed analysis
        'fold_f1_scores': cv_results['test_f1'].tolist()
    }


def get_models_and_param_grids():
    """
    Returns all models with their parameter grids.
    Parameter names are prefixed with 'classifier__' for pipeline compatibility.
    """
    return {
        'Logistic Regression': {
            'classifier': LogisticRegression(max_iter=2000, random_state=42),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1, 5, 10],
                'classifier__solver': ['lbfgs', 'liblinear'],
                'classifier__penalty': ['l2']
            }
        },
        'KNN': {
            'classifier': KNeighborsClassifier(),
            'param_grid': {
                'classifier__n_neighbors': [3, 5, 7, 9, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__p': [1, 2]
            }
        },
        'SVM': {
            'classifier': SVC(probability=True, random_state=42),
            'param_grid': {
                'classifier__C': [0.1, 1, 5, 10],
                'classifier__gamma': ['scale', 0.1, 0.01],
                'classifier__kernel': ['rbf']
            }
        },
        'Decision Tree': {
            'classifier': DecisionTreeClassifier(random_state=42),
            'param_grid': {
                'classifier__max_depth': [3, 5, 7, 10, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'classifier': RandomForestClassifier(random_state=42),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        },
        'AdaBoost': {
            'classifier': AdaBoostClassifier(random_state=42),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 1]
            }
        },
        'Gaussian NB': {
            'classifier': GaussianNB(),
            'param_grid': {
                'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        },
        'XGBoost': {
            'classifier': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 1.0]
            }
        }
    }


def run_all_nested_cv(X, y, oversampler='smote'):
    """
    Run nested CV for all classical models.

    Args:
        X: Features (numpy array or DataFrame)
        y: Target (numpy array or Series)
        oversampler: 'smote' or 'ros'

    Returns:
        dict with results for each model
    """
    models_config = get_models_and_param_grids()
    results = {}

    sampler_name = "SMOTE" if oversampler == 'smote' else "Random Oversampling"
    print(f"\n{'='*60}")
    print(f"NESTED CV WITH {sampler_name} INSIDE FOLDS")
    print(f"{'='*60}")
    print("Outer CV: 10-fold (performance estimation)")
    print("Inner CV: 5-fold (hyperparameter tuning)")
    print(f"{'='*60}\n")

    for name, config in models_config.items():
        print(f"Running nested CV for {name}...", end=" ", flush=True)

        result = nested_cv_with_tuning(
            X, y,
            config['classifier'],
            config['param_grid'],
            oversampler=oversampler
        )

        results[name] = result
        print(f"F1: {result['test_f1_mean']:.4f} (+/- {result['test_f1_std']:.4f})")

    return results


def print_nested_cv_summary(results):
    """Print a formatted summary of nested CV results."""
    print("\n" + "="*80)
    print("NESTED CV RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Test F1':>12} {'Std':>10} {'Train F1':>12} {'Overfit?':>10}")
    print("-"*80)

    for name, metrics in results.items():
        test_f1 = metrics['test_f1_mean']
        test_std = metrics['test_f1_std']
        train_f1 = metrics['train_f1_mean']

        # Check for overfitting (train >> test)
        gap = train_f1 - test_f1
        overfit = "Yes" if gap > 0.15 else "No"

        print(f"{name:<25} {test_f1:>12.4f} {test_std:>10.4f} {train_f1:>12.4f} {overfit:>10}")

    print("="*80)
