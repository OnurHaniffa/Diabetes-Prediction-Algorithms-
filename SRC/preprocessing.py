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

    '''Imputing missing values in the dataset.'''

    print('Zero counts per column:\n', (df == 0).sum())
    cols_with_missing=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']       #Columns where zero indicates missing values
    for col in cols_with_missing:
        df[col]=df[col].replace(0,np.nan)     #Replace zeros with NaN

    for col in cols_with_missing:
        df[col]=df[col].fillna(df[col].mean())      #Impute NaN with median values

    print('Missing values after imputation:\n', df.isna().sum())
    return df


def remove_outliers(df):   #This function is to remove outliers from the dataset using the IQR method 
   
    col_for_outliers=['Glucose','BloodPressure','SkinThickness','Insulin','BMI',]
    for col in col_for_outliers:
        Q1=df[col].quantile(0.25)
        Q2=df[col].quantile(0.75)
        IQR=Q2-Q1
        Lower_Bound=Q1-1.5*IQR          #Calculating Lower and Upper bounds
        Upper_Bound=Q2+1.5*IQR

        df=df[(df[col]<= Upper_Bound) & (df[col]>=Lower_Bound)]     #Removing outliers
        print(f'Outliers removed from {col}. New shape: {df.shape}')
    return df


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




