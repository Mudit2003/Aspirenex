# Handwritten Text Recognition Model Training and Prediction

This script performs the following steps:
1. Loads and cleans the dataset.
2. Preprocesses the images and labels.
3. Extracts the set of unique characters in the training data.
4. Builds the OCR model using a pre-trained ResNet50 base.
5. Trains the model on the training data.
6. Makes predictions using the trained model.
7. Decodes the predictions and tests the model on the test data.

## Functions

### `loadDataset`
- **Description:** Loads the dataset from the specified source.
- **Returns:** DataFrame containing the loaded dataset.

### `cleanData`
- **Description:** Cleans the dataset by removing NaN values and unreadable entries.
- **Parameters:** DataFrame (input dataset)
- **Returns:** Tuple of cleaned training, test, and validation datasets.

### `imagePreprocessing`
- **Description:** Preprocesses the images and labels for training, validation, and test datasets.
- **Parameters:** Training, test, and validation CSV files.
- **Returns:** Preprocessed data for training, test, and validation.

### `tokenizing`
- **Description:** Tokenizes the labels in the training data.
- **Parameters:** Training dataset.
- **Returns:** Tokenized labels.

### `buildModel`
- **Description:** Builds the OCR model using a pre-trained ResNet50 base.
- **Parameters:** Set of unique characters in the training data.
- **Returns:** Compiled OCR model.

### `trainModel`
- **Description:** Trains the OCR model on the training data.
- **Parameters:** OCR model, training data, validation data, training and validation label lengths.
- **Returns:** Trained OCR model.

### `predict`
- **Description:** Makes predictions using the trained OCR model.
- **Parameters:** Trained OCR model.
- **Returns:** None.

### `decode_predictions`
- **Description:** Decodes the predictions made by the OCR model.
- **Parameters:** Predictions made by the model.
- **Returns:** Decoded text predictions.

### `testPrediction`
- **Description:** Tests the model on the test data and evaluates its performance.
- **Parameters:** Trained OCR model, test data, test labels.
- **Returns:** None.

