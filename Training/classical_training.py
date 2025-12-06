import sys, os
import joblib
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from SRC.preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features,
    split_data,
    scale_data
)

from SRC.classical_models import (
    train_all_models,
    evaluate_models_cv
)

from SRC.util_results import save_results, save_models


def run_training():

    print("\n===== LOADING DATA =====")
    df = load_data("Data/pima.csv")

    print("\n===== IMPUTING MISSING VALUES =====")
    df = impute_missing_values(df)

    print("\n===== REMOVING OUTLIERS =====")
    df = remove_outliers(df)

    print("\n===== FEATURE SELECTION =====")
    df = select_features(df, target="Outcome", threshold=0.20)

    print("\n===== SPLITTING DATA =====")
    x_train, x_test, y_train, y_test = split_data(df)

    print("Train size:", x_train.shape)
    print("Test size:", x_test.shape)

    print("\n===== SCALING DATA =====")
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    print("\n===== TRAINING MODELS (Train/Test Split) =====")
    split_results, trained_models = train_all_models(x_train_scaled, y_train, x_test_scaled, y_test)

    print("\n===== TRAIN/TEST RESULTS =====")
    for model_name, metrics in split_results.items():
        print(f"\n--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # FULL DF for cross-validation
    x = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    print("\n===== 10-FOLD CROSS VALIDATION =====")
    cv_results = evaluate_models_cv(x, y, k=10)

    print("\n===== CROSS VALIDATION RESULTS =====")
    for model_name, metrics in cv_results.items():
        print(f"\n--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # SAVE METRICS
    all_results = {
        "train_test": split_results,
        "cross_validation": cv_results
    }

    save_results(all_results, "metrics_classical.json")

    # SAVE MODELS
    save_models(trained_models)

    print("\n===== TRAINING COMPLETE =====")


if __name__ == "__main__":
    run_training()
