#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
__file__
    Submission.py
__author__
    Xu Xiaoming< humixu0208@gmail.com >
"""

import numpy as np
import pandas as pd
import time,datetime
from datetime import datetime
from Preprocessing import builddata
from Train import trainmodel
from Evaluate import eveluate

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras import backend as K


class Submission(object):

    def __init__(self, model, test_features):
        self.model = model
        self.test_features = test_features

    def submission(self, path):
        submission = pd.DataFrame()
        submission['id'] = self.test_features.index
        submission['proba'] = pd.DataFrame(self.model.predict(test_features.values))
        submission.to_csv(path, index=False)
        print('write the result to local!')



###################
## Main Function ##
###################

if __name__ == "__main__":
    localtime = time.asctime(time.localtime())
    print("程序开始运行时间：" + str(localtime))

    train_data_path = "./DATA/DATA_v2/stock_train_data_20170910.csv"
    test_data_path = "./DATA/DATA_v2/stock_test_data_20170910.csv"

    builddata = builddata(train_data_path,test_data_path)
    train_features, train_label, train_weight, val_features, val_label, val_weight = builddata.split_train_data()
    test_features = builddata.process_test_data()

    trainDNN = trainmodel()
    model = trainDNN.dnn_model()
    model = trainDNN.model_fit(model, train_features, train_label, train_weight)

    eveluate = eveluate(model, train_features, train_label, val_features, val_label)
    eveluate.train_performance()
    eveluate.val_performance()

    Submission = Submission(model, test_features)
    Submission.submission('./submission/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

    localtime = time.asctime(time.localtime())
    print("程序结束运行时间：" + str(localtime))