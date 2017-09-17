#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
__file__
    Preprocessing.py
__author__
    Xu Xiaoming< humixu0208@gmail.com >
"""

import numpy as np
import pandas as pd
import time,datetime



class builddata(object):
    train_data_path = ""
    test_data_path = ""

    def __init__(self, train_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def split_train_data(self):
        # Read raw data
        df_train = pd.read_csv(self.train_data_path)
        df_train.set_index('id', inplace=True)
        df_train['group'] = df_train['group'].astype(str)
        df_train = pd.get_dummies(df_train, columns=['group'])
        print ("Read raw data is completed, Dimension of df_train is {}".format(df_train.shape))

        # Split df_train to train_data and val_data
        train_data = df_train[~df_train.era.isin([19,20])]
        val_data = df_train[df_train.era.isin([19,20])]
        print ("Dimension of train_data is {}".format(train_data.shape))
        print ("Dimension of val_data is {}".format(val_data.shape))

        # Data processing
        train_features = train_data.copy()
        train_features.drop(labels=['weight','label','era'], axis=1, inplace=True)
        train_label = train_data.label
        train_weight = train_data.weight

        val_features = val_data.copy()
        val_features.drop(labels=['weight','label','era'], axis=1, inplace=True)
        val_label = val_data.label
        val_weight = val_data.weight

        return(train_features,train_label,train_weight,val_features,val_label,val_weight)

    def process_test_data(self):
        # Read raw data
        df_test = pd.read_csv(self.test_data_path)
        df_test.set_index('id', inplace=True)
        df_test['group'] = df_test['group'].astype(str)
        test_features = pd.get_dummies(df_test, columns=['group'])
        print ("Dimension of test_features is {}".format(test_features.shape))

        return(test_features)



if __name__ == "__main__":
    localtime = time.asctime(time.localtime())
    print("程序开始运行时间：" + str(localtime))

    train_data_path = "./DATA/DATA_v2/stock_train_data_20170910.csv"
    test_data_path = "./DATA/DATA_v2/stock_test_data_20170910.csv"

    builddata = builddata(train_data_path,test_data_path)
    train_features, train_label, train_weight, val_features, val_label, val_weight = builddata.split_train_data()
    test_features = builddata.process_test_data()

    localtime = time.asctime(time.localtime())
    print("程序结束运行时间：" + str(localtime))
    exit()



