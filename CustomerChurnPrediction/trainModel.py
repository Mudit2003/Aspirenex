from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

def trainModel(X_train, X_test, y_train, y_test):
    """
    Trains a Random Forest classifier on the given training data and evaluates it on the test data.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The training input samples.
    X_test : array-like of shape (n_samples, n_features)
        The test input samples.
    y_train : array-like of shape (n_samples,)
        The target values for training.
    y_test : array-like of shape (n_samples,)
        The target values for testing.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        The trained Random Forest classifier instance.

    Notes
    -----
    This function fits a Random Forest classifier on the training data and evaluates it using
    classification report, confusion matrix, accuracy score, and ROC AUC score on the test data.
    It also computes cross-validation scores using 10-fold cross-validation on the test data.
    """
    rf = RandomForestClassifier(random_state=42)

    # Fit the model
    rf.fit(X_train, y_train)

    # Random Forest evaluation
    y_pred_rf = rf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

    # Cross-validation scores
    rf_cv_scores = cross_val_score(rf, X_test, y_test, cv=10)

    return rf
