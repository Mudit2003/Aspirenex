from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def featureSelection(X, y):
    """
    Perform feature selection using SelectKBest and chi-squared test.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.

    Returns
    -------
    tuple
        Tuple containing the selected features and split test data:
        - X_train_new : array-like of shape (n_samples, k_features)
          Selected training features.
        - X_test_new : array-like of shape (n_samples, k_features)
          Selected test features.
        - y_train : array-like of shape (n_samples,)
          Training target values.
        - y_test : array-like of shape (n_samples,)
          Test target values.

    Notes
    -----
    This function scales the input data using MinMaxScaler, performs train-test split,
    applies SelectKBest feature selection using chi-squared test with k=10 features,
    and returns the selected features along with the split training and test data.
    """
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train = scaler.fit_transform(X_train)

    # Select k best feature selection method
    select_k_best = SelectKBest(chi2, k=10)
    X_train_new = select_k_best.fit_transform(X_train, y_train)

    selected_features = select_k_best.get_support(indices=True)
    X_test_new = X_test.iloc[:, selected_features]

    return X_train_new, X_test_new, y_train, y_test
