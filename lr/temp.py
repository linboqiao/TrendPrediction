# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import linear_model as lineM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation as cvf
from sklearn.preprocessing import OneHotEncoder as ohe


def clean_out(dat,p):
    data=dat.iloc[:,np.array(dat.dtypes!='object')]
    object_dat=dat.iloc[:,np.array(dat.dtypes=='object')]
    high=data.quantile(p)
    low=data.quantile(1-p)
    return pd.concat([data[(data<=high) &(data>=low)],object_dat],axis=1).dropna()


def one_hot_parse(var_name,data):
    v=data
    for i in var_name:
        one_hot_var=pd.get_dummies(v[i])
        var_name_1=list(one_hot_var.columns)
        for j in range(0,len(var_name_1)):
            var_name_1[j]=i+str(var_name_1[j])
        one_hot_var.columns=var_name_1
        v=pd.concat([v.drop(i,axis=1),one_hot_var],axis=1)
    return v

a=pd.read_csv(r'D:/python_project/stock_train_data_20170910.csv')
print a.dtypes
a.describe().T
#plt.hist(a['label'],200)
#print a.count()
#a=clean_out(a,0.98)
print a
#a=one_hot_parse(['group','era'],a)
print a
logistic=lineM.LogisticRegression()
y=a['label']
x=a[a.columns[a.columns!='label']]
test_data=a[a['era']>=19]
train_data=a[a['era']<19]
#test_data=a[(a['era19']>0)|(a['era20']>0)]
#train_data=a[(a['era19']<=0)&(a['era20']<=0)]
test_x=test_data[test_data.columns[test_data.columns!='label']]
test_y=test_data['label']
train_x=train_data[train_data.columns[train_data.columns!='label']]
train_y=train_data['label']
a=logistic.fit(train_x,train_y)
a.predict(test_x)
print cvf.cross_val_score(logistic,train_x,train_y,cv=5)