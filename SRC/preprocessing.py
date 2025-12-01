import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(path):            #This funtion is to load data from a CSV file
   
    '''Loading the Pima 
    Diabetes Daataset from a CSV file.'''

    df.read_csv(path)
    print('Data loaded successfully with shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print(df.head())
    return df




def impute_missing_values(df):     #This function is to handle missing values in the dataset

    '''Imputing missing values in the dataset.'''

    print('Zero counts per column:\n', (df == 0).sum())
    cols_with_missing=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
