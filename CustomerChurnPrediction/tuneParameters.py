from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV


def tuneParameters(rf , X_train , y_train , X_test , y_test):

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Parameters from Grid Search:", grid_search.best_params_)
    best_rf = grid_search.best_estimator_

    # evaluation
    y_pred_best_rf = best_rf.predict(X_test)
    print("Best Random Forest Classification Report:\n", classification_report(y_test, y_pred_best_rf))
    print("Best Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_rf))
    print("Best Random Forest Accuracy Score:", accuracy_score(y_test, y_pred_best_rf))
    print("Best Random Forest ROC AUC Score:", roc_auc_score(y_test, y_pred_best_rf))