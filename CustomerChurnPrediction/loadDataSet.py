import pandas as pd


def loadDataSet(path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    """ Load dataset input: path : string , output : pandas dataframe"""
    return pd.read_csv(path)