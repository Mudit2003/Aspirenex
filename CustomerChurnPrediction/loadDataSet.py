import pandas as pd


def loadDataSet(path = r'D:\thrash\others\Aspirenex\CustomerChurnPrediction\WA_Fn-UseC_-Telco-Customer-Churn.csv') -> pd.DataFrame:
    # print(path)
    return pd.read_csv(path)