#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
__file__
    Evaluate.py
__author__
    Xu Xiaoming< humixu0208@gmail.com >
"""

import numpy as np
import pandas as pd
import time,datetime
from Preprocessing import builddata
from Train import trainmodel

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers



##############
## Eveluate ##
##############

class eveluate(object):

    train_features = np.array([])
    train_label = np.array([])
    val_features = np.array([])
    val_label = np.array([])

    def __init__(self, model, train_features, train_label, val_features, val_label):
        self.model = model
        self.train_features = train_features.values
        self.train_label = train_label.values
        self.val_features = val_features.values
        self.val_label = val_label.values

    def train_performance(self):
        train_loss, train_accuracy = self.model.evaluate(self.train_features, self.train_label, verbose = 0)
        print("Train Accuracy = {:.2f}".format(train_accuracy))
        print("Train Loss = {:.2f}".format(train_loss))

    def val_performance(self):
        val_loss, val_accuracy = self.model.evaluate(self.val_features, self.val_label, verbose = 0)
        print("Val Accuracy = {:.2f}".format(val_accuracy))
        print("Val Loss = {:.2f}".format(val_loss))



###################
## Main Function ##
###################
if __name__ == "__main__":
    localtime = time.asctime(time.localtime())
    print("程序开始运行时间：" + str(localtime))

    train_data_path = "./DATA/DATA_V3/stock_train_data_20170916.csv"
    test_data_path = "./DATA/DATA_V3/stock_test_data_20170916.csv"

    builddata = builddata(train_data_path,test_data_path)
    train_features, train_label, train_weight, val_features, val_label, val_weight = builddata.split_train_data_normalization()
    test_features = builddata.process_test_data_normalization()

    trainDNN = trainmodel()
    model = trainDNN.dnn_model()
    model = trainDNN.model_fit(model, train_features, train_label, train_weight)

    eveluate = eveluate(model, train_features, train_label, val_features, val_label)
    eveluate.train_performance()
    eveluate.val_performance()

    localtime = time.asctime(time.localtime())
    print("程序结束运行时间：" + str(localtime))