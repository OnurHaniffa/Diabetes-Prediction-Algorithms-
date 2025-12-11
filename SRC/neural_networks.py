import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Use your existing preprocessing functions
from preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features,
    split_data,
    scale_data
)


def prepare_diabetes_data(path):
    """
    Runs the SAME preprocessing pipeline as your classical models,
    so the ANN is fully consistent with the article + your code.
    """

    print("\n=== 1) LOADING RAW DATA ===")
    df = load_data(path)

    print("\n=== 2) IMPUTING MISSING VALUES ===")
    df = impute_missing_values(df)

    print("\n=== 3) REMOVING OUTLIERS (IQR, 3*IQR rule) ===")
    df = remove_outliers(df)

    print("\n=== 4) FEATURE SELECTION (corr >= 0.20, article-style) ===")
    df = select_features(df, target="Outcome", threshold=0.20)
    # After this, df has 5 input features + Outcome

    print("\n=== 5) TRAIN / TEST SPLIT (85% / 15%) ===")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n=== 6) MIN-MAX SCALING (0–1, article-style) ===")
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def build_article_ann_one_hidden(input_dim: int):
    """
    Builds the 1-hidden-layer ANN described in the article:

    - Input: 5 features
    - Hidden layer: 5 neurons, ReLU
    - Output layer: 1 neuron, sigmoid
    - Optimizer: SGD with learning rate 0.01
    """

    model = Sequential()

    # Input + hidden layer (5 neurons, ReLU)
    model.add(
        Dense(
            units=5,               # number of neurons in hidden layer
            activation="relu",     # article: ReLU in hidden layer
            input_dim=input_dim    # number of input features
        )
    )

    # Output layer (1 neuron, sigmoid for binary classification)
    model.add(
        Dense(
            units=1,
            activation="sigmoid"   # outputs probability between 0 and 1
        )
    )

    # Optimizer: Stochastic Gradient Descent, learning_rate=0.01 (article choice)
    optimizer = SGD(learning_rate=0.01)

    # Compile = tell Keras how to train:
    # - loss: binary cross entropy (standard for binary classification)
    # - metrics: track accuracy while training
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_article_ann_one_hidden(path: str, epochs: int = 200, batch_size: int = 16):
    """
    Full pipeline:
    - prepares data
    - builds article-style ANN (1 hidden layer, ReLU)
    - trains it
    - evaluates on test set
    """

    # 1) Data pipeline
    X_train, X_test, y_train, y_test = prepare_diabetes_data(path)

    input_dim = X_train.shape[1]  # should be 5 (Glucose, BMI, Insulin, Preg, Age)

    # 2) Build model
    print("\n=== BUILDING ARTICLE-STYLE ANN (1 hidden layer, ReLU) ===")
    model = build_article_ann_one_hidden(input_dim)

    print("\nModel summary:")
    model.summary()

    # 3) Train model (fit)
    print(f"\n=== TRAINING ANN for {epochs} epochs, batch_size={batch_size} ===")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 4) Evaluate on test set
    print("\n=== EVALUATION ON TEST SET ===")
    y_prob = model.predict(X_test).ravel()          # probabilities in [0,1]
    y_pred = (y_prob >= 0.5).astype(int)           # convert to class 0/1

    acc = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", acc)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Return model + history + metrics in case you want to save/log
    results = {
        "accuracy": acc,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "history": history.history
    }

    return model, results
