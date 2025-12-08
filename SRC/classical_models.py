from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np


def get_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=42)

def get_knn():
    return KNeighborsClassifier(n_neighbors=7)

def get_svm():
    return SVC(kernel="rbf", probability=True, random_state=42)     # Using RBF kernel beacuse its the default for WEKA

def get_decision_tree():
    return DecisionTreeClassifier(random_state=42)

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def get_adaboost():
    return AdaBoostClassifier(random_state=42)

def get_gaussian_nb():
    return GaussianNB()

def get_all_models():    #Returns a dictionary of all classical models used in the article. Each model is UNTRAINED. Training happens in Training/classical_training.py
   

    return {
        "Logistic Regression": get_logistic_regression(),
        "KNN": get_knn(),
        "SVM": get_svm(),
        "Decision Tree": get_decision_tree(),
        "Random Forest": get_random_forest(),
        "AdaBoost": get_adaboost(),
        "Gaussian NB": get_gaussian_nb(),
    }


def train_all_models(x_train, y_train, x_test, y_test):
    models = get_all_models()
    results = {}
    trained_models = {}   # New dictionary to hold trained model objects

    for name, model in models.items():
        model.fit(x_train, y_train)       # Trained here, function will be called from training/classical_training.py
        preds = model.predict(x_test)
       
       
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds)
        }
        results[name]=metrics
        trained_models[name] = model    #  Save trained model object here

   
    return results, trained_models   # Return both results and trained models

def evaluate_models_cv(x, y, k=10):     # 10 k-fold cross validation evaluation of all models  
   
    models = get_all_models()
    results = {}

    for name, model in models.items():
        accuracy = cross_val_score(model, x, y, cv=k, scoring="accuracy").mean()
        precision = cross_val_score(model, x, y, cv=k, scoring="precision").mean()
        recall = cross_val_score(model, x, y, cv=k, scoring="recall").mean()
        f1 = cross_val_score(model, x, y, cv=k, scoring="f1").mean()

        results[name] = {
            "cv_accuracy": accuracy,
            "cv_precision": precision,
            "cv_recall": recall,
            "cv_f1": f1
        }

    return results
