from SRC.preprocessing import (
    load_data,
    impute_missing_values,
    remove_outliers,
    select_features,
    split_data,
    scale_data
)

def main():

    print("\n===== LOADING DATA =====")
    df = load_data("Data/pima.csv")

    print("\n===== IMPUTING MISSING VALUES =====")
    df = impute_missing_values(df)

    print("\n===== REMOVING OUTLIERS =====")
    df = remove_outliers(df)

    print("\n===== SELECTING FEATURES =====")
    df = select_features(df, target='Outcome', threshold=0.20)
    print("Selected dataframe shape:", df.shape)

    print("\n===== SPLITTING DATA =====")
    x_train, x_test, y_train, y_test = split_data(df)
    print("Train/Test sizes:", x_train.shape, x_test.shape)

    print("\n===== SCALING DATA =====")
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    print("Scaled train shape:", x_train_scaled.shape)
    print("Scaled test shape:", x_test_scaled.shape)

    print("\n===== PREPROCESSING COMPLETE =====")

if __name__ == "__main__":
    main()
