from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def featureSelection(X , y):

    scaler = MinMaxScaler()
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 , random_state=42)

    X_train = scaler.fit_transform(X_train)

    # select k best feature selection method 
    select_k_best = SelectKBest(chi2, k=10)
    X_train_new = select_k_best.fit_transform(X_train, y_train)


    selected_features = select_k_best.get_support(indices=True)
    # transform the test data accordingly , note that we did not want the test data to affect the calculation of K best features 
    # this is to ensure a generalize model else overfitting will prevail 
    X_test_new = X_test.iloc[:selected_features]

    return (X_train_new , X_test_new , y_train , y_test)