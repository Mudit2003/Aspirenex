import pandas as pd


import os


def loadDataset(path):
    train_csv = pd.read_csv(os.path.join(path, 'written_name_train_v2.csv'))
    validation_csv = pd.read_csv(os.path.join(path, 'written_name_validation_v2.csv'))
    test_csv = pd.read_csv(os.path.join(path, 'written_name_test_v2.csv'))

    return train_csv , validation_csv , test_csv