from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score




def trainModel(X_train , X_test , y_train , y_test):

    rf = RandomForestClassifier(random_state=42)

    # Fit the models
    rf.fit(X_train, y_train)

    # Random Forest evaluation
    y_pred_rf = rf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))

    rf_cv_scores = cross_val_score(rf, X_test, y_test, cv=10)


    return rf