# -*- coding: utf-8 -*-


from sklearn import linear_model as lineM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation as cvf
from sklearn.preprocessing import OneHotEncoder as ohe


###########
###function
###########
##去离群点
def clean_out(dat,p):
    data=dat.iloc[:,np.array(dat.dtypes!='object')]
    object_dat=dat.iloc[:,np.array(dat.dtypes=='object')]
    high=data.quantile(p)
    low=data.quantile(1-p)
    return pd.concat([data[(data<=high) &(data>=low)],object_dat],axis=1).dropna()

##哑变量编码
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

##循环编码
def get_circle(var_name,data,n):
    circle=data['var_name']%n
    pd.concat([data,circle],axis=1)
    return data

##woe编码
class woe_encoder(object):
    
    def __init__(self,data,quantile,yname):
        self.data=data
        self.start=pd.quantile(quantile)
        self.have_encode=False
        self.y=data[yname]
        self.y_0,self.y_1=data.groupby(data[yname]).count()
        
    def woe_value(self,var,var_name,var_q):
        var_new=var
        seq_v,seq_name=self.encoder(pd.DataFrame({var_name:var}),var_q)
        vi=0
        wait_count=pd.DataFrame(seq_v,self.y)
        for i in range(0,len(seq_v)):
            count_0,count1=
        
    def encoder(self,var_b,var_q):
        b=tuple(var_b)
        var=pd.Series(b)
        jug=[var<=var_q[0]]*len(var_q)
        jug_name=[' ']*(len(var_q)-1)
        for i in range(0,(len(var_q)-1)):
            if(i==0):
                jug[i]=var>var_q[i]
                jug_name[i]='%s' % var_q[i]
            else:
                jug[i]=(var>var_q[i])&(var<=var_q[i+1])
                jug_name[i]='%s-%s' % (var_q[i],var_q[i+1])
        print jug
        for j in range(0,len(jug)):
            var[jug[j]]=j
        return [var,jug_name]
    
                

##权重重抽样
def get_weight_data(,)

#########
##xgboost
#########


###########
###__main__
###########

##参数
lr_paramters={}
lr_paramters['penalty']='l1'
#lr_paramters['']
logistic=lineM.LogisticRegression(lr_paramters)
a=pd.read_csv(r'D:/python_project/stock_train_data_20170910.csv')
test_data_final=pd.read.csv(r'D:/python_project/stock_test_data_20170910.csv')
weight=a['weight']
a=a.drop('weight')
print a.dtypes
a.describe().T
#plt.hist(a['label'],200)
#print a.count()
#a=clean_out(a,0.98)


for cn in range(1,21):
    a=get_circle('era',a,cn)    
    #print a
    #a=one_hot_parse(['group','era'],a)
    #print a
    y=a['label']
    x=a[a.columns[a.columns!='label']]
    test_data=a[a['era']>=19]
    train_data=a[a['era']<19]
    #  test_data=a[(a['era19']>0)|(a['era20']>0)]
    #train_data=a[(a['era19']<=0)&(a['era20']<=0)]
    test_x=test_data[test_data.columns[test_data.columns!='label']]
    test_y=test_data['label']
    train_x=train_data[train_data.columns[train_data.columns!='label']]
    train_y=train_data['label']
    a=logistic.fit(train_x,train_y)
    a.predict(test_x)
    print cvf.cross_val_score(logistic,train_x,train_y,cv=5)