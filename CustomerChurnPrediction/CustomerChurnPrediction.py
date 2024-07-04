import warnings
warnings.filterwarnings("ignore")
from dataPreprocessing import dataPreprocessing as dp 
from featureSelection import featureSelection as fs
from loadDataSet import loadDataSet as ld 
from trainModel import trainModel as tm
from tuneParameters import tuneParameters as tp

X, y = dp(ld())
X_train_new , X_test_new , y_train , y_test = fs(X, y)
rf = tm(X_train_new , X_test_new , y_train , y_test)

tp(rf, X_train_new , X_test_new , y_train , y_test)






    









