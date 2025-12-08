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
    get_all_tuned_models,
    train_tuned_models
)

from SRC.util_results import save_results


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_training_tuned():

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

    print("\n===== GRID SEARCH TUNING =====")
    tuned_models = get_all_tuned_models(x_train_scaled, y_train)

    print("\n===== TRAINING TUNED MODELS =====")
    tuned_results, trained_models = train_tuned_models(
        tuned_models, x_train_scaled, y_train, x_test_scaled, y_test
    )

    print("\n===== TUNED RESULTS =====")
    for model_name, metrics in tuned_results.items():
        print(f"\n--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # === SAVE RESULTS ===
    save_results(tuned_results, "metrics_classical_tuned.json")

    # === SAVE MODELS ===
    ensure_dir("Models/Tuned")
    for name, model in trained_models.items():
        path = f"Models/Tuned/{name.replace(' ', '_').lower()}_tune.pkl"
        joblib.dump(model, path)
        print(f"Saved model: {path}")

    print("\n===== TUNED TRAINING COMPLETE =====")


if __name__ == "__main__":
    run_training_tuned()
