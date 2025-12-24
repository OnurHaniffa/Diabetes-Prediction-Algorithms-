"""
Training script using Nested CV with SMOTE inside folds.

This is the CORRECT way to apply SMOTE with cross-validation:
- SMOTE is applied fresh inside each CV fold
- Prevents data leakage from synthetic samples
- Provides unbiased, realistic performance estimates

Compare these results with classical_training_tuned_smote.py to see
the difference between correct (nested) and problematic (outside-fold) approaches.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SRC.preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features
)

from SRC.nested_cv_models import run_all_nested_cv, print_nested_cv_summary
from SRC.util_results import save_results


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_nested_cv_smote():
    print("\n" + "="*60)
    print("NESTED CV TRAINING WITH SMOTE INSIDE FOLDS")
    print("="*60)

    # ===== DATA LOADING AND PREPROCESSING =====
    print("\n===== LOADING DATA =====")
    df = load_data("Data/pima.csv")

    print("\n===== IMPUTING MISSING VALUES =====")
    df = impute_missing_values(df)

    print("\n===== REMOVING OUTLIERS =====")
    df = remove_outliers(df)

    print("\n===== FEATURE SELECTION =====")
    df = select_features(df, target="Outcome", threshold=0.20)

    # Extract features and target (NO split - nested CV handles this)
    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    print(f"\nDataset shape: {X.shape}")
    print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")

    # ===== RUN NESTED CV WITH SMOTE =====
    results = run_all_nested_cv(X, y, oversampler='smote')

    # ===== PRINT SUMMARY =====
    print_nested_cv_summary(results)

    # ===== SAVE RESULTS =====
    ensure_dir("Results/nested_cv_metrics")

    # Add metadata to results
    final_results = {
        "methodology": {
            "description": "Nested CV with SMOTE applied inside each fold",
            "outer_cv": "10-fold stratified (for performance estimation)",
            "inner_cv": "5-fold stratified (for hyperparameter tuning)",
            "oversampling": "SMOTE applied fresh in each fold (no data leakage)",
            "scoring": "F1 score for tuning, multiple metrics for evaluation"
        },
        "models": results
    }

    save_results(final_results, "metrics_nested_cv_smote.json")
    print(f"\nResults saved to: Results/metrics_nested_cv_smote.json")

    print("\n" + "="*60)
    print("NESTED CV TRAINING COMPLETE (SMOTE)")
    print("="*60)

    return results


if __name__ == "__main__":
    run_nested_cv_smote()
