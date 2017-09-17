#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
__file__
    Train.py
__author__
    Xu Xiaoming< humixu0208@gmail.com >
"""

import numpy as np
import pandas as pd
import time,datetime
from Preprocessing import builddata

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras import backend as K



#################
## Train Model ##
#################

class trainmodel(object):

    def dnn_model(self):
        # define the model
        model = Sequential([
            Dense(256, input_dim=116),
            Activation('relu'),
            Dropout(0.5),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(1),
            Activation('sigmoid')
        ])

        # Another way to define your optimizer
        #rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # We add metrics to get more results you want to see
        model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def model_fit(self, model, train_features, train_label, train_weight):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, verbose=1, patience=5)
        model.fit(train_features.values, train_label, sample_weight = train_weight, callbacks = [early_stopping],
                  epochs = 2, batch_size = 100)

        return(model)



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

    train_loss, train_accuracy = model.evaluate(train_features.values, train_label, verbose = 0)
    print("Train Accuracy = {:.2f}".format(train_accuracy))

    val_loss, val_accuracy = model.evaluate(val_features.values, val_label, verbose = 0)
    print("Val Accuracy = {:.2f}".format(val_accuracy))

    localtime = time.asctime(time.localtime())
    print("程序结束运行时间：" + str(localtime))











