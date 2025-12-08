from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def tune_logistic(X, y):  # Logistic Regression (balanced class weights + regularization tuning)
    model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")

    param_grid = {
        "C": [0.01, 0.1, 1, 5, 10],
        "solver": ["lbfgs", "liblinear"],
        "penalty": ["l2"]
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest Logistic Regression:", grid.best_params_)
    return grid.best_estimator_



def tune_knn(X, y):     # K-Nearest Neighbors (tuning neighbors, weights, p)
    model = KNeighborsClassifier()

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13],
        "weights": ["uniform", "distance"],
        "p": [1, 2]   # Manhattan, Euclidean
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest KNN:", grid.best_params_)
    return grid.best_estimator_



def tune_svm(X, y):     # Support Vector Machine (balanced class weights + C and gamma tuning)
    model = SVC(probability=True, random_state=42, class_weight="balanced")

    param_grid = {
        "C": [0.1, 1, 5, 10, 20],
        "gamma": ["scale", 0.1, 0.01, 0.001],
        "kernel": ["rbf"]
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest SVM:", grid.best_params_)
    return grid.best_estimator_


def tune_decision_tree(X, y):       # Decision Tree (balanced class weights + depth and split tuning)
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest Decision Tree:", grid.best_params_)
    return grid.best_estimator_



def tune_random_forest(X, y):       # Random Forest (balanced, more controlled depth and estimators)
    model = RandomForestClassifier(random_state=42, class_weight="balanced")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest Random Forest:", grid.best_params_)
    return grid.best_estimator_



def tune_adaboost(X, y):        # AdaBoost (tuning n_estimators and learning_rate)
    model = AdaBoostClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 1, 2]
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest AdaBoost:", grid.best_params_)
    return grid.best_estimator_


def tune_gaussian_nb(X, y):     # Gaussian Naive Bayes (tuning var_smoothing)
    model = GaussianNB()

    param_grid = {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X, y)

    print("\nBest Gaussian NB:", grid.best_params_)
    return grid.best_estimator_


def get_all_tuned_models(X, y):         #Returns a dictionary of all TUNED classical models used in the article. Each model is TUNED here using GridSearchCV.


    tuned_models = {
        "Logistic Regression": tune_logistic(X, y),
        "KNN": tune_knn(X, y),
        "SVM": tune_svm(X, y),
        "Decision Tree": tune_decision_tree(X, y),
        "Random Forest": tune_random_forest(X, y),
        "AdaBoost": tune_adaboost(X, y),
        "Gaussian NB": tune_gaussian_nb(X, y),
    }

    return tuned_models


def train_tuned_models(models, x_train, y_train, x_test, y_test):
    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }

        trained[name] = model

    return results, trained
