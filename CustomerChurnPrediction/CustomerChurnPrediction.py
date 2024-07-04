import warnings
warnings.filterwarnings("ignore")

import dataPreprocessing as dp 
import featureSelection as fs
import loadDataSet as ld 
import trainModel as tm
import tuneParameters as tp

tp(tm(fs(dp(ld()))))







    









