import os, sys, joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SRC.preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features,
    split_data,
    scale_data
)

from SRC.classical_models_tuned import (
    get_all_tuned_models_with_ros,
    train_tuned_models,
    evaluate_tuned_models_cv
)

from SRC.util_results import save_results


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_training_tuned_ros():

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

    print("\n===== SCALING DATA =====")
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    print("\n===== RANDOM OVERSAMPLING + GRID SEARCH TUNING =====")
    tuned_models, X_train_ros, y_train_ros = get_all_tuned_models_with_ros(
        x_train_scaled, y_train
    )

    print("\n===== TRAINING TUNED MODELS ON OVERSAMPLED DATA =====")
    tuned_results, trained_models = train_tuned_models(
        tuned_models, X_train_ros, y_train_ros, x_test_scaled, y_test
    )

    print("\n===== TUNED + ROS RESULTS (Train/Test) =====")
    for model_name, metrics in tuned_results.items():
        print(f"\n--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # === 10-FOLD CROSS VALIDATION (on ROS'd training data) ===
    print("\n===== 10-FOLD CROSS VALIDATION =====")
    cv_results = evaluate_tuned_models_cv(tuned_models, X_train_ros, y_train_ros, k=10)

    print("\n===== TUNED + ROS RESULTS (Cross Validation) =====")
    for model_name, metrics in cv_results.items():
        print(f"\n--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # === SAVE RESULTS ===
    all_results = {
        "train_test": tuned_results,
        "cross_validation": cv_results
    }
    save_results(all_results, "metrics_classical_tuned_ros.json")

    # === SAVE MODELS ===
    ensure_dir("Models/tuned_ros")
    for name, model in trained_models.items():
        path = f"Models/tuned_ros/{name.replace(' ', '_').lower()}_tuned_ros.pkl"
        joblib.dump(model, path)
        print(f"Saved model: {path}")

    print("\n===== TUNED + ROS TRAINING COMPLETE =====")


if __name__ == "__main__":
    run_training_tuned_ros()
