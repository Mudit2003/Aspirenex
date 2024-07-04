import pandas as pd


def loadDataSet(path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:

    return pd.read_csv(path)