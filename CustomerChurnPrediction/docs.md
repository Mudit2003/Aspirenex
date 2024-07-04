# Customer Churn Prediction Project Documentation

## Overview
This project aims to predict customer churn using machine learning techniques. Churn prediction helps businesses understand which customers are likely to leave, allowing proactive measures to retain them.

## Modules

### 1. **Data Preprocessing Module**

#### `dataPreprocessing(df: pd.DataFrame) -> tuple`
Preprocesses the input DataFrame for customer churn prediction.

**Steps:**
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

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame containing customer data.

**Returns:**
- `tuple`: A tuple containing two elements:
  - `X (pd.DataFrame)`: The preprocessed feature matrix.
  - `y (pd.Series)`: The target variable (Churn).

---

### 2. **Feature Selection Module**

#### `featureSelection(X: pd.DataFrame, y: pd.Series) -> tuple`
Performs feature selection using SelectKBest and chi-squared test.

**Steps:**
1. Splits the data into training and test sets.
2. Scales the training data using MinMaxScaler.
3. Applies SelectKBest feature selection with chi-squared test to select k=10 best features.
4. Transforms the test data accordingly to keep consistency in feature selection.

**Parameters:**
- `X (pd.DataFrame)`: The feature matrix.
- `y (pd.Series)`: The target variable.

**Returns:**
- `tuple`: A tuple containing four elements:
  - `X_train_new (pd.DataFrame)`: Selected training features.
  - `X_test_new (pd.DataFrame)`: Selected test features.
  - `y_train (pd.Series)`: Training target values.
  - `y_test (pd.Series)`: Test target values.

---

### 3. **Model Training Module**

#### `trainModel(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> RandomForestClassifier`
Trains a Random Forest classifier on the selected features.

**Steps:**
1. Initializes a Random Forest classifier with default parameters.
2. Fits the model to the training data (`X_train`, `y_train`).
3. Evaluates the model performance on the test data (`X_test`, `y_test`) using:
   - Classification Report
   - Confusion Matrix
   - Accuracy Score
   - ROC AUC Score

**Parameters:**
- `X_train (pd.DataFrame)`: Selected training features.
- `X_test (pd.DataFrame)`: Selected test features.
- `y_train (pd.Series)`: Training target values.
- `y_test (pd.Series)`: Test target values.

**Returns:**
- `RandomForestClassifier`: Trained Random Forest classifier instance.

---

### 4. **Parameter Tuning Module**

#### `tuneParameters(rf: RandomForestClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None`
Tunes the hyperparameters of the Random Forest classifier using Grid Search.

**Steps:**
1. Defines a parameter grid for Grid Search with:
   - 'n_estimators': [100, 200, 300]
   - 'max_depth': [None, 10, 20, 30]
   - 'min_samples_split': [2, 5, 10]
   - 'min_samples_leaf': [1, 2, 4]
2. Performs Grid Search Cross Validation (CV=5) to find the best hyperparameters.
3. Prints the best parameters found by Grid Search.
4. Evaluates the best model on the test data (`X_test`, `y_test`) using:
   - Classification Report
   - Confusion Matrix
   - Accuracy Score
   - ROC AUC Score

**Parameters:**
- `rf (RandomForestClassifier)`: Initialized Random Forest classifier.
- `X_train (pd.DataFrame)`: Selected training features.
- `X_test (pd.DataFrame)`: Selected test features.
- `y_train (pd.Series)`: Training target values.
- `y_test (pd.Series)`: Test target values.

**Returns:**
- `None`

---

## Usage

```python
import warnings
warnings.filterwarnings("ignore")

from dataPreprocessing import dataPreprocessing as dp
from featureSelection import featureSelection as fs
from loadDataSet import loadDataSet as ld
from trainModel import trainModel as tm
from tuneParameters import tuneParameters as tp

def main():
    # Load and preprocess dataset
    X, y = dp(ld())

    # Perform feature selection
    X_train_new, X_test_new, y_train, y_test = fs(X, y)

    # Train a Random Forest model
    rf = tm(X_train_new, X_test_new, y_train, y_test)

    # Tune model parameters
    tp(rf, X_train_new, X_test_new, y_train, y_test)

if __name__ == "__main__":
    main()
