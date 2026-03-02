# Diabetes Prediction — ML Research

Replication and extension of a peer-reviewed study ([ICT Express, 2021](https://doi.org/10.1016/j.icte.2021.02.004)) on diabetes prediction using the Pima Indians Diabetes Dataset. Systematically compares 8 classical ML algorithms and 1 neural network with rigorous evaluation methodology.

## Methodology

- **8 ML Algorithms:** Logistic Regression, KNN, SVM, Decision Tree, Random Forest, AdaBoost, Naive Bayes, XGBoost
- **1 Neural Network:** Keras Sequential (SGD optimizer)
- **4 Training Variants Per Algorithm:** Untuned, GridSearchCV-tuned, Tuned + SMOTE, Tuned + Random Oversampling
- **32+ Model Configurations** evaluated systematically
- **Nested Cross-Validation:** 10-fold outer, 5-fold inner — SMOTE applied inside each fold via `imblearn.Pipeline` to prevent data leakage
- **Overfitting Detection:** Flags models where train-test accuracy gap exceeds 0.15
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC AUC

## Project Structure

```
├── Data/           # Pima Indians Diabetes Dataset
├── Models/         # Saved trained models (.pkl)
├── Notebooks/      # Jupyter notebooks for exploration and analysis
├── Results/        # Evaluation outputs and comparison tables
├── SRC/            # Source package (preprocessing, models, training, results)
├── Training/       # Training scripts and pipeline
├── main.py         # Entry point
└── test_pipeline.py # Pipeline tests
```

## Tech Stack

Python, scikit-learn, TensorFlow/Keras, XGBoost, imbalanced-learn, Pandas, NumPy, Matplotlib, Seaborn

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Key Technical Decisions

1. **Data leakage prevention:** SMOTE is applied inside each cross-validation fold (not before splitting), using `imblearn.Pipeline`. This is a common methodological error in ML research that we explicitly avoid.
2. **Nested CV:** Inner loop for hyperparameter tuning, outer loop for unbiased performance estimation.
3. **Systematic comparison:** Every algorithm is evaluated under 4 conditions to isolate the effects of tuning and class imbalance handling.

## Reference

> Diabetes prediction using machine learning algorithms. *ICT Express*, 2021. [DOI: 10.1016/j.icte.2021.02.004](https://doi.org/10.1016/j.icte.2021.02.004)

## Built By

**Onur Haniffa** — [LinkedIn](https://linkedin.com/in/onurhaniffa) · [GitHub](https://github.com/OnurHaniffa)
