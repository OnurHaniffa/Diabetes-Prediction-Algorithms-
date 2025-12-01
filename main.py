def main():
    print("Diabetes replication project initialized!")

if __name__ == "__main__":
    main()

from src.preprocessing.preprocessing import *

# Load data
df = load_data("data/diabetes.csv")

# Missing values
df = impute_missing_values(df)

# Outlier removal
df = remove_outliers(df)

# Train/test split
X_train, X_test, y_train, y_test = split_data(df)

# Scaling
X_train_scaled, X_test_scaled = scale_data(X_train, X_Test)

print("Everything ran successfully!")
