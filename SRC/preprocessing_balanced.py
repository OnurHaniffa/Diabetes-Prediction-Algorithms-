
from SRC.preprocessing import (
    impute_missing_values,
    remove_outliers,
    select_features,
    split_data,
    scale_data
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import numpy as np



def balanced_split_and_scale(df):       #same as classical but with new funtion for SMOTE balancing 

    # Split (same as classical)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Scale first
    scaler = MinMaxScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Apply SMOTE ONLY to training set
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

    print("Before SMOTE:", x_train.shape, y_train.shape)
    print("After SMOTE:", x_train_balanced.shape, y_train_balanced.shape)

    return x_train_balanced, x_test_scaled, y_train_balanced, y_test


def apply_random_oversampling(X_train_scaled, y_train):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

    print("\n=== RANDOM OVERSAMPLING APPLIED ===")
    print(f"Before: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print(f"After:  0={sum(y_resampled==0)}, 1={sum(y_resampled==1)}")

    return X_resampled, y_resampled
