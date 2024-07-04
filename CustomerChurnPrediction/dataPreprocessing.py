import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def dataPreprocessing(df : pd.DataFrame):
    """
    Preprocesses the input DataFrame for customer churn prediction.

    This function performs the following preprocessing steps:
    1. Drops the 'customerID' column as it is not useful for classification.
    2. Replaces specific values in several columns to simplify the data:
        - 'No internet service' is replaced with 'No' in columns related to internet services.
        - 'No phone service' is replaced with 'No' in the 'MultipleLines' column.
        - 'Month-to-month' is replaced with 'Month' in the 'Contract' column.
    3. Creates a new 'tenure_group' column by binning the 'tenure' column into 6 categories.
    4. Converts the 'TotalCharges' column to numeric, handling any non-numeric values by replacing them with the mean.
    5. Encodes categorical variables using Label Encoding.
    6. Drops any remaining missing values.
    7. Separates the features (X) from the target variable (y).

    Parameters:
    df (pd.DataFrame): The input DataFrame containing customer data.

    Returns:
    tuple: A tuple containing two elements:
        - X (pd.DataFrame): The preprocessed feature matrix.
        - y (pd.Series): The target variable (Churn).
    """

    # customer id won't be useful for classification 
    df.drop('customerID' , axis=1 , inplace=True)

    # no internet service and no phone service were already there
    df['OnlineSecurity']=df['OnlineSecurity'].replace('No internet service','No')
    df['OnlineBackup']=df['OnlineBackup'].replace('No internet service','No')
    df['DeviceProtection']=df['DeviceProtection'].replace('No internet service','No')
    df['TechSupport']=df['TechSupport'].replace('No internet service','No')
    df['StreamingTV']=df['StreamingTV'].replace('No internet service','No')
    df['StreamingMovies']=df['StreamingMovies'].replace('No internet service','No')
    df['MultipleLines']=df['MultipleLines'].replace('No phone service','No')
    df['Contract']=df['Contract'].replace('Month-to-month','Month')

    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 60, 72 , np.inf], labels=[1, 2, 3, 4, 5, 6])



    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    df.dropna(inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return (X , y)