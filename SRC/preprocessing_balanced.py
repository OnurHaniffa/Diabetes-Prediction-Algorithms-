from SRC.preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


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
