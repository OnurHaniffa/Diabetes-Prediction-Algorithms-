import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path):            #This funtion is to load data from a CSV file
   
    '''Loading the Pima 
    Diabetes Daataset from a CSV file.'''

    df=pd.read_csv(path)
    print('Data loaded successfully with shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print(df.head())
    return df


def impute_missing_values(df):     #This function is to handle missing values in the dataset


    print('Zero counts per column:\n', (df == 0).sum())
    cols_with_missing=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']       #Columns where zero indicates missing values
    for col in cols_with_missing:
        df[col]=df[col].replace(0,np.nan)     #Replace zeros with NaN

    for col in cols_with_missing:
        df[col]=df[col].fillna(df[col].mean())      #Impute NaN with mean values

    print('Missing values after imputation:\n', df.isna().sum())
    return df


def remove_outliers(df):

    df_clean = df.copy()

    cols = ['Insulin', 'SkinThickness']   # columns known to have crazy values
    rows_to_drop = set()

    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 3 * IQR   # 3*IQR: only extreme values
        upper = Q3 + 3 * IQR

        idx = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].index

        print(f"Extreme outliers in {col}: {len(idx)}")
        rows_to_drop.update(idx)

    print("Total unique rows to remove:", len(rows_to_drop))

    df_clean = df_clean.drop(index=rows_to_drop).reset_index(drop=True)
    print("Shape after outlier removal:", df_clean.shape)

    return df_clean

def select_features(df, target='Outcome',threshold=0.20):   #This function is to select specific features from the dataset
    corr=df.corr()[target].drop(target)  
    print('Feature correlations with target:\n', corr)

    selected_features = corr[corr.abs() > threshold].index.tolist()

    print(f"\nSelected features (|correlation| >= {threshold}):")
    print(selected_features)
    return df[selected_features + [target]]  #Return dataframe with selected features and target column
    

    
def split_data(df):             #This function is to split the dataset into training and testing sets     
    
    x=df.drop('Outcome',axis=1)
    y=df['Outcome']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42,stratify=y)       #random state makes it reproducible by keeping the same split every time,stratify maintains class distribution
    
    print('x train shape:', x_train.shape, 'x test shape:', x_test.shape)
    return x_train, x_test, y_train, y_test


def scale_data(x_train,x_test):        #This function is to scale the features using Min-Max Scaling
    scaler=MinMaxScaler()
    scaler.fit(x_train)        #Fit only on training data, this is done to prevent data leakage. It learns the parameter needed for scaling.
   
    x_train_scaled=scaler.transform(x_train)
    x_test_scaled=scaler.transform(x_test)      #Transform both training and testing data. This uses the parameters learned from training data fitting.
    
    x_train_scaled=pd.DataFrame(x_train_scaled,columns=x_train.columns)
    x_test_scaled=pd.DataFrame(x_test_scaled,columns=x_test.columns)

    print('Data scaling samle:\n', x_train_scaled.head())
    return x_train_scaled, x_test_scaled




