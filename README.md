
# Handwritten Text Recognition Model Training and Prediction

This script performs the following steps:
1. Loads and cleans the dataset.
2. Preprocesses the images and labels.
3. Extracts the set of unique characters in the training data.
4. Builds the OCR model using a pre-trained ResNet50 base.
5. Trains the model on the training data.
6. Makes predictions using the trained model.
7. Decodes the predictions and tests the model on th test data.

Functions
---------
loadDataset : function
    Loads the dataset from the specified source.
cleanData : function
    Cleans the dataset by removing NaN values and unreadable entries.
imagePreprocessing : function
    Preprocesses the images and labels for training, validation, and test datasets.
tokenizing : function
    Tokenizes the labels in the training data.
buildModel : function
    Builds the OCR model using a pre-trained ResNet50 base.
trainModel : function
    Trains the OCR model on the training data.
predict : function
    Makes predictions using the trained model.
decode_predictions : function
    Decodes the predictions made by the model.
testPrediction : function
    Tests the model on the test data and evaluates its performance.

