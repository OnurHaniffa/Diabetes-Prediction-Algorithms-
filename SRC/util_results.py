import json
import os
import joblib

def save_results(results: dict, filename: str):
    """
    Saves a dictionary of results into the Results/ folder as a JSON file.
    """
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to {filepath}")




def save_models(models: dict, folder: str = "Models/classical"):
    """
    Saves trained model objects into Models/classical/ as .pkl files.
    """
    os.makedirs(folder, exist_ok=True)

    for model_name, model_object in models.items():
        filename = model_name.replace(" ", "_").lower() + ".pkl"
        path = os.path.join(folder, filename)

        joblib.dump(model_object, path)
        print(f"Saved model: {path}")

