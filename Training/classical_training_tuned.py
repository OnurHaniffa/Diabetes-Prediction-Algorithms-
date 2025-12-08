import os, sys, json, joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==== IMPORT PREPROCESSING ====
from SRC.preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features
)

from SRC.preprocessing_balanced import balanced_split_and_scale

# ==== IMPORT TUNING AND TRAINING ====
from SRC.classical_models_tuned import (
    get_all_tuned_models,
    train_tuned_models
)

# ==== IMPORT SAVE FUNCTION ====
from SRC.util_results import save_results


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_training_classical_tuned():

    print("\n===== LOADING DATA =====")
    df = load_data("Data/pima.csv")

    print("\n===== IMPUTING MISSING VALUES =====")
    df = impute_missing_values(df)

    print("\n===== REMOVING OUTLIERS =====")
    df = remove_outliers(df)

    print("\n===== FEATURE SELECTION =====")
    df = select_features(df, target="Outcome", threshold=0.20)

    print("\n===== BALANCED SPLIT + SMOTE + SCALING =====")
    x_train_bal, x_test_scaled, y_train_bal, y_test = balanced_split_and_scale(df)

    print("\n===== TUNING ALL MODELS (GridSearchCV) =====")
    tuned_models = get_all_tuned_models(x_train_bal, y_train_bal)

    print("\n===== TRAINING TUNED MODELS ON TRAINING DATA =====")
    tuned_results, trained_tuned_models = train_tuned_models(
        tuned_models, 
        x_train_bal, y_train_bal,
        x_test_scaled, y_test
    )

    print("\n===== EVALUATION RESULTS (TEST SET) =====")
    for model_name, metrics in tuned_results.items():
        print(f"\n--- {model_name} ---")
        for m, v in metrics.items():
            print(f"{m}: {v:.4f}")

    # ===== SAVE METRICS =====
    ensure_dir("Results")
    save_results(tuned_results, "metrics_classical_tuned.json")

    # ===== SAVE TUNED MODELS =====
    ensure_dir("Models/tuned")

    for name, model in trained_tuned_models.items():
        filename = f"Models/tuned/{name.replace(' ', '_').lower()}_tuned.pkl"
        joblib.dump(model, filename)
        print(f"Saved model → {filename}")

    print("\n===== TRAINING COMPLETE (TUNED MODELS) =====")


if __name__ == "__main__":
    run_training_classical_tuned()
