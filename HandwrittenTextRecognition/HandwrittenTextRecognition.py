from loadDataset import loadDataset
from model import buildModel
from predict import predict, decode_predictions, testPrediction
from preprocess import cleanData, imagePreprocessing, tokenizing
from training import trainModel


train_csv , test_csv , validation_csv = cleanData(loadDataset)

x_train , y_train , train_label_len, train_input_len , x_test , y_test , x_val , y_val , valid_label_len , valid_input_len = imagePreprocessing(train_csv , test_csv)

characters = set(char for label in train_csv['IDENTITY'].values for char in label)

model = buildModel(characters)

model = trainModel(model , x_train ,y_train , train_input_len , train_label_len , x_val , y_val , valid_input_len , valid_label_len)

predict(model)
print(decode_predictions(model))
testPrediction(model , x_test , y_test)
