# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a diabetes prediction ML project replicating methods from a research article using the Pima Indians Diabetes dataset. It compares classical ML models (with and without hyperparameter tuning) and neural networks.

## Commands

### Environment Setup
```bash
./setup.sh                          # Creates .venv, installs dependencies from requirements.txt
source .venv/bin/activate           # Activate virtual environment
```

### Running Training Scripts
All training scripts must be run from the project root directory:
```bash
python Training/classical_training.py           # Train untuned classical models
python Training/classical_training_tuned.py     # Train with GridSearchCV hyperparameter tuning
python Training/classical_training_tuned_smote.py  # Train tuned models with SMOTE balancing
```

### Testing
```bash
python test_pipeline.py             # Test the preprocessing pipeline
```

## Architecture

### Directory Structure
- **SRC/**: Core ML code (preprocessing, models, utilities)
- **Training/**: Training scripts that orchestrate the pipeline
- **Models/**: Saved trained model files (.pkl)
- **Results/**: JSON files with evaluation metrics
- **Data/**: Contains `pima.csv` dataset
- **Notebooks/**: Jupyter notebooks for exploration

### Data Pipeline Flow
1. `load_data()` - Load CSV
2. `impute_missing_values()` - Replace zeros with mean for Glucose, BloodPressure, SkinThickness, Insulin, BMI
3. `remove_outliers()` - IQR-based removal (3*IQR rule) for Insulin and SkinThickness
4. `select_features()` - Keep features with |correlation| >= 0.20 to Outcome
5. `split_data()` - 85/15 train/test split, stratified, random_state=42
6. `scale_data()` - MinMaxScaler (fit on train only)

### Model Types
**Classical Models** (`SRC/classical_models.py`):
- Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest, AdaBoost, Gaussian NB

**Tuned Models** (`SRC/classical_models_tuned.py`):
- Same models with GridSearchCV hyperparameter optimization (F1 scoring, 5-fold CV)

**Neural Networks** (`SRC/neural_networks.py`):
- 1-hidden-layer ANN: 5 neurons (ReLU) → 1 output (sigmoid), SGD lr=0.01

### Class Imbalance Handling
`SRC/preprocessing_balanced.py` provides:
- `balanced_split_and_scale()` - Applies SMOTE after scaling, only on training data
- `apply_random_oversampling()` - Alternative random oversampling

### Utility Functions
`SRC/util_results.py`:
- `save_results(dict, filename)` - Saves metrics to Results/ as JSON
- `save_models(dict, folder)` - Saves trained models as .pkl files

## Key Implementation Notes

- Training scripts use `sys.path.append()` to import from SRC/ - run from project root
- Preprocessing imports in `SRC/` use relative imports (e.g., `from preprocessing import ...`)
- All models use `random_state=42` for reproducibility
- Evaluation includes both train/test split metrics and 10-fold cross-validation
