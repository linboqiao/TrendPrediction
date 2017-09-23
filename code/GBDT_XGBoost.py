
# coding: utf-8

# In[ ]:

#! /usr/bin/python
from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np
import xgboost as xgb
import time


# In[ ]:

# load data

data1 = np.loadtxt('../data/stock_train_data_20170901.csv',delimiter=',',skiprows=1)
data2 = np.loadtxt('../data/stock_train_data_20170910.csv',delimiter=',',skiprows=1)
data3 = np.loadtxt('../data/stock_train_data_20170916.csv',delimiter=',',skiprows=1)
data  = np.append(data1, data2, axis=0)
data  = np.append(data, data3, axis=0)
print('data shape:', data.shape)

# data preprocessing
np.random.shuffle(data)
data_X = data[:,1:89]
data_Y = data[:,90]
weight_samples = data[:,89]
group_samples = data[:,91]
era_samples = data[:,92]
scaler = preprocessing.StandardScaler().fit(data_X)
data_X = scaler.transform(data_X)
data_cv = xgb.DMatrix(data_X, data_Y, weight = weight_samples)
print('data_X shape:', data_X.shape)
print('data_Y shape:', data_Y.shape)


# In[ ]:


# setup parameters for xgboost
param = {}
# logistic regression for binary classification. Output probability.
param['objective'] = 'binary:logistic' 
param['metrics'] = {'logloss'}
param['eta'] = 0.1          # step size of each boosting step
param['max_depth'] = 8       # maximum depth of the tree
param['silent'] = 1
param['nthread'] = 7
param['seed'] = 0
param['nrounds'] = 10
#param['eval_metric'] = "auc"
# https://rdrr.io/cran/xgboost/man/xgb.train.html
# https://www.cnblogs.com/haobang008/p/5909207.html

start = time.time();
print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
res = xgb.cv(param, data_cv,      
             nfold=10,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(3)])
end = time.time();
print (res)
print('time elapse:', end- start);


# In[ ]:




# In[ ]:

# model training

per_train = 0.9
# data preprocessing
np.random.shuffle(data)
# rebuild the data with era (time)

data_X = data[:,1:89]
data_Y = data[:,90]
weight_samples = data[:,89]
group_samples = data[:,91].reshape(-1,1)
era_samples = data[:,92]

#data_X = np.append(data_X, group_samples, axis= 1)
scaler = preprocessing.StandardScaler().fit(data_X)
data_X = scaler.transform(data_X)
data_cv = xgb.DMatrix(data_X, data_Y, weight = weight_samples)
print('data_X shape:', data_X.shape)
print('data_Y shape:', data_Y.shape)

# more work needed for traing set selection...
test_X = data_X[int(data_X.shape[0] * per_train):]
test_Y = data_Y[int(data_X.shape[0] * per_train):]
weight_test = weight_samples[int(data_X.shape[0] * per_train):]
train_X = data_X[0:int(data_X.shape[0] * per_train)]
train_Y = data_Y[0:int(data_X.shape[0] * per_train)]
weight_train = weight_samples[0:int(data_X.shape[0] * per_train)]
print("train_X",train_X.shape)
print("test_X",test_X.shape)

xg_train = xgb.DMatrix( train_X, label=train_Y, weight = weight_train)
xg_test = xgb.DMatrix(test_X, label=test_Y, weight = weight_test)
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]

num_round = 10
start = time.time();
bst = xgb.train(param, xg_train, num_round, watchlist);
end = time.time();
print('time elapse train:', end- start);
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
start = time.time();
y_pred = bst.predict( xg_test )
end = time.time();
print('time elapse predict:', end- start);
print(y_pred)
print('error:', np.sum(test_Y == (y_pred > 0)).astype(float) / test_Y.shape[0])


# In[ ]:




# In[ ]:

data_to_Predict = np.loadtxt('../data/stock_test_data_20170916.csv',delimiter=',',skiprows=1)
ids = data_to_Predict[:,0]
test_X = data_to_Predict[:,1:-1]
test_X = xgb.DMatrix(test_X)

start = time.time();
yprob = bst.predict( test_X )
end = time.time();
print('time elapse predict:', end- start);

data_pred = np.concatenate((data_to_Predict[:,0].reshape(yprob.shape[0],-1), yprob.reshape(yprob.shape[0],-1)), axis=1)

f = open('./data_pred.csv', 'w')
f.write('id,proba\n')
for i in range(data_pred.shape[0]):
    s = '%d,%.5f\n'%(data_pred[i,0], data_pred[i,1])
    f.write(s)
f.close()


# In[ ]:



