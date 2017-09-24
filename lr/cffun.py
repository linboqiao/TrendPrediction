
import numpy as np
import pandas as pd
from pandas import Series, DataFrame, read_csv, read_table
from scipy import stats
from math import log
from sklearn import metrics
import math
import heapq
from scipy import stats
from sklearn.utils.multiclass import type_of_target
from scipy.interpolate import UnivariateSpline


MIN_NUM = -100000000
MAX_NUM = 100000000
WOE_MIN = -20
WOE_MAX = 20

__version__='0.2'
# modify
# 2016.12.05  update iv_list and iv_value


def ks_value(x, y,event=1,segNum=10):
    ks_res, ks = ks_list(x, y, segNum=10, event=1)
    return ks


def ks_value1(x, y,event=1,segNum=10):
    ks_res, ks = ks_list_1(x, y, segNum=10, event=1)
    return ks

def ks_list1(x,y, segNum=10,event=1):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    event_list = []
    non_event_list = []
    g_num = []
    b_num = []
    iv = 0
    start_point = stats.scoreatpercentile(x, 0)
    for i in range(segNum):
        end_point = stats.scoreatpercentile(x, (i + 1) * (100/segNum))
        if i < (segNum-1):
            y1 = y[np.where((x >= start_point) & (x<end_point))[0]]
        else:
            y1 = y[np.where((x >= start_point) & (x<=end_point))[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        event_list.append(rate_event)
        non_event_list.append(rate_non_event)
    ks = (np.abs(np.array(non_event_list) - np.array(event_list))).max()
    ks_res = DataFrame({'x':event_list,'y':non_event_list,'g_num':g_num, 'b_num':b_num})
    return ks_res, ks

def ks_list(x,y, segNum=10,event=1):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    ylab = list(set(y))
    if len(ylab)!=2:
        raise Exception('hiahia,your label is wrong number')
    ylab = dict(zip(ylab,[1,0]))
    df = DataFrame({'x':x,'y':y})
    df = df.reindex(np.random.permutation(df.index))
    df['y'] = df['y'].replace(ylab)
    df = df.sort_values(['x'],ascending=1)
    num_total = len(df)
    event_total = sum(df['y'])
    non_event_total = num_total - event_total
    df.index = range(num_total)
    dfindex = [int(num_total*i/float(segNum))-1 for i in range(segNum+1)]
    dfindex[0] = 0
    event_list = []
    non_event_list = []
    g_num = []
    b_num = []
    iv = 0
    points_cut = []
    start_point = stats.scoreatpercentile(x, 0)
    for i in range(1,segNum+1):
        event_count = sum(df['y'][0:(dfindex[i]+1)])
        rindex = dfindex[i]
        points_cut.append(str(df['x'][rindex]))
        non_event_count = dfindex[i]+ 1 - event_count
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        event_list.append(rate_event)
        non_event_list.append(rate_non_event)
    ks = (np.abs(np.array(non_event_list) - np.array(event_list))).max()
    ks_res = DataFrame({'b_ratio':event_list,'g_ratio':non_event_list,'g_num':g_num, 'b_num':b_num,'point':points_cut})
    ks_res = ks_res.reindex(columns =['g_num','b_num','g_ratio','b_ratio', 'point'])
    return ks_res, ks

def ks_value_s(x, y,segNum=10):
    ks_res, ks = ks_list(x, y, segNum=10, event=1)
    return ks

def ks_list_s(x,y,segNum=10):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    #ylab = list(set(y))
    #if len(ylab)!=3:
    #    raise Exception('hiahia,your label is wrong number')
    #ylab = dict(zip(ylab,[1,0]))
    df = DataFrame({'x':x,'y':y})
    #df['y'] = df['y'].replace(ylab)
    df = df.sort_values(['x'],ascending=1)
    #df['y'] = df['y'].replace(ylab)
    df = df.sort_values(['x'],ascending=1)
    num_total = len(df)
    #event_total = sum(df['y'])
    #non_event_total = num_total - event_total
    event_total = sum(df['y']==1)
    non_event_total = sum(df['y']==0)
    df.index = range(num_total)
    dfindex = [int(num_total*i/float(segNum))-1 for i in range(segNum+1)]
    dfindex[0] = 0
    event_list = []
    non_event_list = []
    g_num = []
    b_num = []
    iv = 0
    start_point = stats.scoreatpercentile(x, 0)
    for i in range(1,segNum+1):
        event_count = sum((df['y'][0:(dfindex[i]+1)])==1)
        non_event_count = sum((df['y'][0:(dfindex[i]+1)])==0)
        #non_event_count = dfindex[i]+ 1 - event_count
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        event_list.append(rate_event)
        non_event_list.append(rate_non_event)
    ks = (np.abs(np.array(non_event_list) - np.array(event_list))).max()

    direction=1

    ks_list=np.abs(np.array(non_event_list) - np.array(event_list))

    best_ks_index=heapq.nlargest(1, range(len(ks_list)), ks_list.take)[0]*0.1

    if non_event_list[int(best_ks_index*10)]>event_list[int(best_ks_index*10)]: direction=-1

    ks_res = DataFrame({'b_ratio':event_list,'g_ratio':non_event_list,'g_num':g_num, 'b_num':b_num})
    ks_res = ks_res.reindex(columns =['g_num','b_num','g_ratio','b_ratio'])
    return ks_res, ks, best_ks_index,direction

def check_target_binary(y):
    '''
    check if the target variable is binary, raise error if not.
    :param y:
    :return:
    '''
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('Label type must be binary')

def count_binary(a, event=1):
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count

def PSI_value(pre_train, pre_test, seg_num=10):
    pre_train=Series(pre_train)
    pre_test=Series(pre_test)
    df_len = pre_train.shape[0]
    tr_tb = 1.0*pre_train.value_counts()/df_len
    if len(tr_tb) >= 10 :
        tr_fb = list(tr_tb[tr_tb>=0.1].index)
        if tr_fb != []:
            tr_factor = pre_train[pre_train.isin(tr_fb)]
            tr_numeric= pre_train[~(pre_train.isin(tr_fb))]
            te_factor = pre_test[pre_test.isin(tr_fb)]
            te_numeric= pre_test[~(pre_test.isin(tr_fb))]
            factor_part_psi=PSI_value(tr_factor,te_factor)
            num=int(round((len(tr_numeric)/float(df_len))*10))
            numeric_part_psi=PSI_value(tr_numeric,te_numeric,num)
            return(pd.concat([factor_part_psi,numeric_part_psi]))
        else:
            pre_train_sort = pre_train.order()
            pre_test_sort = pre_test.order()
            seg = [i/10.0 for i in range(0,seg_num+1)]
            n=len(pre_train_sort)
            pre_train_sort.index = range(n)
            m=len(pre_test_sort)
            pre_test_sort.index = range(m)
            train_seg_cnt = [0]*seg_num
            train_seg_pct = [0.0]*seg_num
            train_seg_range = [0]*(seg_num+1)
            rangestr = ['0']*seg_num
            for i in range(0,(len(seg)-1)):
                from_index=int(round(n*seg[i]))
                to_index=int(round(n*seg[i+1]))
                train_seg_cnt[i]=len(pre_train_sort[from_index:to_index])
                train_seg_pct[i]=train_seg_cnt[i]/float(n) 
                train_seg_range[i+1]=pre_train_sort[(to_index-1)]
                rangestr[i]="["+str(train_seg_range[i])+","+str(train_seg_range[i+1])+"]"
            train_seg_range[0]=MIN_NUM
            train_seg_range[seg_num]=MAX_NUM
            test_seg_cnt=[0]*seg_num
            test_seg_pct=[0]*seg_num
            psi_value=[0]*seg_num
            for k in range(0,(len(train_seg_range)-1)):
                test_seg_cnt[k] = len(pre_test_sort[((pre_test_sort>=train_seg_range[k]) & (pre_test_sort<train_seg_range[k+1]))])
                test_seg_pct[k]=test_seg_cnt[k]/float(len(pre_test_sort))
                if (test_seg_pct[k]==0):
                    psi_value[k]=None
                else :
                    psi_value[k]=float(train_seg_pct[k]- test_seg_pct[k])*log(train_seg_pct[k]/float(test_seg_pct[k]))

            return(DataFrame({"rangestr":rangestr,"train_seg_cnt":train_seg_cnt,"train_seg_pct":train_seg_pct, 
                                "test_seg_cnt":test_seg_cnt,"test_seg_pct":test_seg_pct,"psi_value":psi_value}))
    else:
        pre_train_levels = list(tr_tb.index)
        n_levels=len(pre_train_levels)
        train_seg_cnt = [0]*n_levels
        train_seg_pct=[0.0]*n_levels
        test_seg_cnt=[0]*n_levels
        test_seg_pct=[0.0]*n_levels
        psi_value=[0.0]*n_levels
        rangestr=[0]*n_levels
        for j in range(0,n_levels):
            rangestr[j]=pre_train_levels[j]
            train_seg_cnt[j]=sum(pre_train==pre_train_levels[j])
            train_seg_pct[j]=train_seg_cnt[j]/float(len(pre_train))
            test_seg_cnt[j]=sum(pre_test==pre_train_levels[j])
            test_seg_pct[j]=test_seg_cnt[j]/float(len(pre_test))
            if ((test_seg_pct[j]!=0) and (train_seg_pct[j]!=0)):
                psi_value[j]=float(train_seg_pct[j] - test_seg_pct[j])*log(train_seg_pct[j]/float(test_seg_pct[j]))
            else :
                psi_value[j]=None
        return(DataFrame({"rangestr":rangestr,"train_seg_cnt":train_seg_cnt,"train_seg_pct":train_seg_pct,"test_seg_cnt":test_seg_cnt,"test_seg_pct":test_seg_pct,"psi_value":psi_value}))

def PSI_sum(pre_train, pre_test, seg_num=10):
    return PSI_value(pre_train, pre_test, seg_num=10).psi_value.sum()

def predict_auc(pred, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1.)
    return metrics.auc(fpr, tpr)

def auc_value(pred, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1.)
    return metrics.auc(fpr, tpr)


def iv_value_1(X,y,event=1):
    X = np.array(X)
    y = np.array(y)
    res_woe, res_iv, x_cuts,xzeor = woeArray(X,y,event=event)
    #res_iv = Series(res_iv,index=X.columns)
    return res_iv[0]

def iv_list_1(X,y,event=1):
    X = np.array(X)
    y = np.array(y)
    res_woe, res_iv, x_cuts,xzeor = woeArray(X,y,event=event)
    #res_iv = Series(res_iv,index=X.columns)
    return res_woe


def woeArray(X, y, event):
    '''
    Calculate woe of each feature category and information value
    :param X: 2-D numpy array explanatory features which should be discreted already
    :param y: 1-D numpy array target variable which should be binary
    :param event: value of binary stands for the event to predict
    :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
             numpy array of information value of each feature
    '''
    check_target_binary(y)
    X1, x_cuts = feature_discretion(X)

    res_woe = []
    res_iv = []
    zero_cnt = []
    if len(X1.shape) == 1:
        ncol = 1
    else:
        ncol = X1.shape[-1]
    for i in range(0, ncol):
        if ncol == 1:
            x = X1
        else:
            x = X1.ix[:, i]
        woe_dict, iv1 = woe_single_x(x, y, event=event)
        res_woe.append(woe_dict)
        res_iv.append(iv1)
        zl = len(x[x==0])
        zero_cnt.append(zl)
    return res_woe, res_iv, x_cuts, zero_cnt


def woe_single_x(x, y, event):
    '''
    calculate woe and information for a single feature
    :param x: 1-D numpy starnds for single feature
    :param y: 1-D numpy array target variable
    :param event: value of binary stands for the event to predict
    :return: dictionary contains woe values for categories of this feature
             information value of this feature
    '''
    check_target_binary(y)

    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        if rate_event == 0:
            woe1 = WOE_MIN
        elif rate_non_event == 0:
            woe1 = WOE_MAX
        else:
            woe1 = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv


def woe_replace(X, woe_arr):
    '''
    replace the explanatory feature categories with its woe value
    :param X: 2-D numpy array explanatory features which should be discreted already
    :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
    :return: the new numpy array in which woe values filled
    '''
    if X.shape[-1] != woe_arr.shape[-1]:
        raise ValueError('WOE dict array length must be equal with features length')

    res = np.copy(X).astype(float)
    idx = 0
    for woe_dict in woe_arr:
        for k in woe_dict.keys():
            woe = woe_dict[k]
            res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
        idx += 1

    return res


def combined_iv(X, y, masks, event=1):
    '''
    calcute the information vlaue of combination features
    :param X: 2-D numpy array explanatory features which should be discreted already
    :param y: 1-D numpy array target variable
    :param masks: 1-D numpy array of masks stands for which features are included in combination,
                  e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
    :param event: value of binary stands for the event to predict
    :return: woe dictionary and information value of combined features
    '''
    if masks.shape[-1] != X.shape[-1]:
        raise ValueError('Masks array length must be equal with features length')

    x = X[:, np.where(masks == 1)[0]]
    tmp = []
    for i in range(x.shape[0]):
        tmp.append(combine(x[i, :]))

    dumy = np.array(tmp)
    # dumy_labels = np.unique(dumy)
    woe, iv = woe_single_x(dumy, y, event)
    return woe, iv


def combine(list):
    res = ''
    for item in list:
        res += str(item)
    return res


def count_binary(a, event=1):
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count


def check_target_binary(y):
    '''
    check if the target variable is binary, raise error if not.
    :param y:
    :return:
    '''
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('Label type must be binary')


def feature_discretion(X):
    '''
    Discrete the continuous features of input data X, and keep other features unchanged.
    :param X : numpy array
    :return: the numpy array in which all continuous features are discreted
    '''
    temp = []
    x_cuts = []
    if len(X.shape) == 1:
        ncol = 1
    else:
        ncol = X.shape[-1]
    for iX in range(0, ncol):
        if ncol == 1:
            x = X
        else:
            x = X.ix[:, iX]
        #x_type = type_of_target(x)
        x_len = len(np.unique(x))
        #if x_type == 'continuous':
        if x_len > 11:
            x, x_cut = discrete(x)
            temp.append(x)
            x_cuts.append(x_cut)
        else:
            temp.append(list(x))
            x_cuts.append(np.unique(x)) 
    #print temp
    return DataFrame(temp).T, x_cuts


def discrete(x):
    '''
    Discrete the input 1-D numpy array using 5 equal percentiles
    :param x: 1-D numpy array
    :return: discreted 1-D numpy array
    '''
    # res = np.array([0] * x.shape[-1], dtype=int)
    # res[np.isnan(x)] = -1
    # for i in range(10):
    #     point1 = stats.scoreatpercentile(x, i * 10)
    #     point2 = stats.scoreatpercentile(x, (i + 1) * 10)
    #     x1 = x[np.where((x >= point1) & (x <= point2))]
    #     mask = np.in1d(x, x1)
    #     res[mask] = (i + 1)
    x = list(x)
    try:
        X_qcut = pd.qcut(x, 10)
        res = list(X_qcut.labels)
        x_cut = list(X_qcut.levels)
    except:
        xS = Series(x)
        xlab = [xS.quantile(i) for i in [i/10.0 for i in range(0,11,1)]]
        xlab_u = sorted(pd.unique(xlab))
        p_cut = pd.cut(x,xlab_u,right=True,include_lowest=True)
        #x_cut = list(p_cut.levels)
        x_cut = list(p_cut.categories)
        res = list(p_cut.labels)
    return res, x_cut


def odds_value_7s(x, y, list1, segNum=100,event=0):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    x_list = []
    y_list = []
    g_num = []
    b_num = []
    odds_list=[]
    log_list=[]
    score_list=[]
    iv = 0
    last_odds = 0.0
    point_list = list1
    for j,i in enumerate(point_list[1:]):
        start_point = stats.scoreatpercentile(x,  point_list[j])
        end_point = stats.scoreatpercentile(x, i)
        y1 = y[np.where((x >= start_point) & (x<=end_point))[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        if event_count==0:
            event_count = 1
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * non_event_count/ event_count
        pointx = (start_point+end_point)/2.0
        x_list.append(pointx)
        y_list.append(rate_event)
        if rate_event < last_odds:
            odds = last_odds + 0.01
        else:
            odds = rate_event
        last_odds = odds
        odds_list.append(odds)
        log_list.append(log(odds))
        score = 600+20*log(odds)/log(2)
        score_list.append(score)
    odds_res = DataFrame({'x':x_list,'oddsx':y_list,'good':g_num,'bad':b_num,'odds':odds_list,'logodds':log_list,'score':score_list})
    odds_res = odds_res.sort('x',ascending=True)
    return odds_res



def odds_value_s(x, y, list1, segNum=100,event=1):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    x_list = []
    y_list = []
    g_num = []
    b_num = []
    odds_list=[]
    log_list=[]
    score_list=[]
    iv = 0
    last_odds = 0.0
    point_list = list1
    for j,i in enumerate(point_list[1:]):
        start_point = stats.scoreatpercentile(x,  point_list[j])
        end_point = stats.scoreatpercentile(x, i)
        y1 = y[np.where((x >= start_point) & (x<=end_point))[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        if event_count==0:
            event_count = 1
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * non_event_count/ event_count
        pointx = (start_point+end_point)/2.0
        x_list.append(pointx)
        y_list.append(rate_event)
        if rate_event < last_odds:
            odds = last_odds + 0.01
        else:
            odds = rate_event
        last_odds = odds
        odds_list.append(odds)
        log_list.append(log(odds))
        score = 600+20*log(odds)/log(2)
        score_list.append(score)
    odds_res = DataFrame({'x':x_list,'oddsx':y_list,'good':g_num,'bad':b_num,'odds':odds_list,'logodds':log_list,'score':score_list})
    odds_res = odds_res.sort('x',ascending=True)
    return odds_res

def odds_value1(x, y, segNum=100,event=1):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    event_total, non_event_total = count_binary(y, event=event)
    #theta = (0.01/0.99) * non_event_total/ event_total
    x_labels = np.unique(x)
    x_list = []
    y_list = []
    g_num = []
    b_num = []
    odds_list=[]
    log_list=[]
    score_list=[]
    iv = 0
    last_odds = 0.0
    point_list = list(np.arange(0,3,0.1)) + list(np.arange(3,20,1)) + list(np.arange(20,100,2)) + [100]
    for j,i in enumerate(point_list[1:]):
        start_point = stats.scoreatpercentile(x,  point_list[j])
        end_point = stats.scoreatpercentile(x, i)
        y1 = y[np.where((x >= start_point) & (x<=end_point))[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        if event_count==0:
            event_count = 1
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * non_event_count/ event_count
        pointx = (start_point+end_point)/2.0
        x_list.append(pointx)
        y_list.append(rate_event)
        if rate_event < last_odds:
            odds = last_odds + 0.01
        else:
            odds = rate_event
        last_odds = odds
        odds_list.append(odds)
        log_list.append(log(odds))
        score = 600+20*log(odds)/log(2)
        score_list.append(score)
    odds_res = DataFrame({'x':x_list,'oddsx':y_list,'good':g_num,'bad':b_num,'odds':odds_list,'logodds':log_list,'score':score_list})
    odds_res = odds_res.sort('x',ascending=True)
    return odds_res

def odds_value(x, y, segNum=100,event=1):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    event_total, non_event_total = count_binary(y, event=event)
    theta = (0.05/0.95) * non_event_total/ event_total
    print '*'*50
    print 'theta: %s' % (theta)
    x_labels = np.unique(x)
    x_list = []
    y_list = []
    g_num = []
    b_num = []
    odds_list=[]
    log_list=[]
    score_list=[]
    iv = 0
    last_odds = 0.0
    point_list = list(np.arange(0,3,0.1)) + list(np.arange(3,20,1)) + list(np.arange(20,100,2)) + [100]
    for j,i in enumerate(point_list[1:]):
        start_point = stats.scoreatpercentile(x,  point_list[j])
        end_point = stats.scoreatpercentile(x, i)
        y1 = y[np.where((x >= start_point) & (x<=end_point))[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        if event_count==0:
            event_count = 1
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * non_event_count/(theta* event_count)
        pointx = (start_point+end_point)/2.0
        x_list.append(pointx)
        y_list.append(rate_event)
        if rate_event < last_odds:
            odds = last_odds + 0.01
        else:
            odds = rate_event
        last_odds = odds
        odds_list.append(odds)
        log_list.append(log(odds))
        score = 600+20*log(odds)/log(2)
        score_list.append(score)
    odds_res = DataFrame({'x':x_list,'oddsx':y_list,'good':g_num,'bad':b_num,'odds':odds_list,'logodds':log_list,'score':score_list})
    odds_res = odds_res.sort('x',ascending=True)
    return odds_res

def odds_value8s(x, y, list1,event=0):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    event_total, non_event_total = count_binary(y, event=event)
    theta = (0.05/0.95) * non_event_total/ event_total
    print '*'*50
    print 'theta: %s' % (theta)
    x_labels = np.unique(x)
    x_list = []
    y_list = []
    g_num = []
    b_num = []
    odds_list=[]
    log_list=[]
    score_list=[]
    iv = 0
    last_odds = 0.0
    point_list = list1
    for j,i in enumerate(point_list[1:]):
        start_point = stats.scoreatpercentile(x,  point_list[j])
        end_point = stats.scoreatpercentile(x, i)
        y1 = y[np.where((x >= start_point) & (x<=end_point))[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        if event_count==0:
            event_count = 1
        g_num.append(non_event_count)
        b_num.append(event_count)
        rate_event = 1.0 * non_event_count/(theta* event_count)
        pointx = (start_point+end_point)/2.0
        x_list.append(pointx)
        y_list.append(rate_event)
        if rate_event < last_odds:
            odds = last_odds + 0.01
        else:
            odds = rate_event
        last_odds = odds
        odds_list.append(odds)
        log_list.append(log(odds))
        score = 600+20*log(odds)/log(2)
        score_list.append(score)
    odds_res = DataFrame({'x':x_list,'oddsx':y_list,'good':g_num,'bad':b_num,'odds':odds_list,'logodds':log_list,'score':score_list})
    odds_res = odds_res.sort('x',ascending=True)
    return odds_res

def odds_spline(oddsDf):
    x = oddsDf.x
    y = oddsDf.logodds
    s = UnivariateSpline(x, y, s=1)
    return s


def psi_list(pre_train, pre_test, seg_num=10):
    pre_train=Series(pre_train)
    pre_test=Series(pre_test)
    tr_len = pre_train.shape[0]
    te_len = pre_test.shape[0]
    tr_tb = 1.0*pre_train.value_counts()/tr_len
    if len(tr_tb) >= 10 :
        pre_train_sort = pre_train.sort_values()
        #pre_test_sort = pre_test.sort_values()
        pre_test_sort = pre_test
        n=len(pre_train_sort)
        pre_train_sort.index = range(n)
        m=len(pre_test_sort)
        pre_test_sort.index = range(m)
        seg_value = []
        for i in range(1,seg_num):
            _index = int(round(1.0*i*tr_len/seg_num))
            seg_value.append(pre_train_sort[_index])
        seg_value = sorted(list(set(seg_value)), key=int)
        seg_len = len(seg_value)+1
        train_seg_cnt = [0]*seg_len
        train_seg_pct = [0.0]*seg_len
        train_seg_range = [0]*seg_len
        test_seg_cnt = [0]*seg_len
        test_seg_pct = [0.0]*seg_len
        test_seg_range = [0]*seg_len
        p_list=[0]*seg_len
        rangestr = ['0']*seg_len
        for i in range(len(seg_value)):
            if i == 0:
                train_seg_cnt[i] = sum(pre_train_sort<=seg_value[i])
                train_seg_pct[i] = train_seg_cnt[i]*1.0/tr_len
                rangestr[i] = "(-inf,"+str(seg_value[i])+"]"
                test_seg_cnt[i] = sum(pre_test_sort<=seg_value[i])
                test_seg_pct[i] = test_seg_cnt[i]*1.0/te_len
                p_list[i]=float(train_seg_pct[i]- test_seg_pct[i])*log(train_seg_pct[i]/float(test_seg_pct[i]))
                continue
            if i == (len(seg_value)-1):
                train_seg_cnt[i+1] = sum(pre_train_sort>seg_value[i])
                train_seg_pct[i+1] = train_seg_cnt[i+1]*1.0/tr_len
                test_seg_cnt[i+1] = sum(pre_test_sort>seg_value[i])
                test_seg_pct[i+1] = test_seg_cnt[i+1]*1.0/te_len
                rangestr[i+1]="("+str(seg_value[i])+", inf)"
                p_list[i+1]=float(train_seg_pct[i+1]- test_seg_pct[i+1])*log(train_seg_pct[i+1]/float(test_seg_pct[i+1]))
            train_seg_cnt[i] = sum((pre_train_sort>seg_value[i-1]) & (pre_train_sort<=seg_value[i]))
            train_seg_pct[i] = train_seg_cnt[i]*1.0/tr_len
            test_seg_cnt[i] = sum((pre_test_sort>seg_value[i-1]) & (pre_test_sort<=seg_value[i]))
            test_seg_pct[i] = test_seg_cnt[i]*1.0/te_len
            rangestr[i]="("+str(seg_value[i-1])+","+str(seg_value[i])+"]"
            p_list[i]=float(train_seg_pct[i]- test_seg_pct[i])*log(train_seg_pct[i]/float(test_seg_pct[i]))
        return(DataFrame({"rangestr":rangestr,"train_seg_cnt":train_seg_cnt,"train_seg_pct":train_seg_pct, 
                                "test_seg_cnt":test_seg_cnt,"test_seg_pct":test_seg_pct,"psi_value":p_list}))
    else:
        pre_train_levels = list(tr_tb.index)
        n_levels=len(pre_train_levels)
        train_seg_cnt = [0]*n_levels
        train_seg_pct=[0.0]*n_levels
        test_seg_cnt=[0]*n_levels
        test_seg_pct=[0.0]*n_levels
        p_list=[0.0]*n_levels
        rangestr=[0]*n_levels
        for j in range(0,n_levels):
            rangestr[j]=pre_train_levels[j]
            train_seg_cnt[j]=sum(pre_train==pre_train_levels[j])
            train_seg_pct[j]=train_seg_cnt[j]/float(len(pre_train))
            test_seg_cnt[j]=sum(pre_test==pre_train_levels[j])
            test_seg_pct[j]=test_seg_cnt[j]/float(len(pre_test))
            if ((test_seg_pct[j]!=0) and (train_seg_pct[j]!=0)):
                p_list[j]=float(train_seg_pct[j] - test_seg_pct[j])*log(train_seg_pct[j]/float(test_seg_pct[j]))
            else :
                p_list[j]=None
        return(DataFrame({"rangestr":rangestr,"train_seg_cnt":train_seg_cnt,"train_seg_pct":train_seg_pct,"test_seg_cnt":test_seg_cnt,"test_seg_pct":test_seg_pct,"psi_value":p_list}))

def psi_value(pre_train, pre_test, seg_num=10):
    return psi_list(pre_train, pre_test, seg_num=10).psi_value.sum()


def _bucket_woe(x):
    t_bad = x['bad']
    t_good = x['good']
    t_bad = 0.5 if t_bad == 0 else t_bad
    t_good = 0.5 if t_good == 0 else t_good
    return np.log(t_bad / t_good)


def iv_list(X, y, seg_num=10):
    df = DataFrame({'X':X,'y':y})
    df_len = df.shape[0]
    tr_tb = 1.0*df.X.value_counts()
    if len(tr_tb) >= 10 :
        df = df.sort_values(['X'])
        #pre_test_sort = pre_test.sort_values()
        xlab = [df["X"].quantile(1.0*i/seg_num) for i in range(0,(seg_num+1))]
        xlab = sorted(pd.unique(xlab))
        cuts, bins = pd.cut(df["X"],xlab,right=True,include_lowest=True,retbins=True, labels=False)
        df["labels"] = cuts.astype(str)
    else:     
        df["labels"] = df.X.astype(str)
    results = df.groupby("labels").agg({'y': [np.count_nonzero, np.size],'X':[np.max, np.min]}, as_index=False)
    result = DataFrame({'obs':results.y['size'], 'bad':results.y['count_nonzero'], 'amin':results.X['amin'], 'amax':results.X['amax']})
    result['good'] = result['obs'] - result['bad']
    t_bad = np.maximum(result['bad'].sum(), 0.5)
    t_good = np.maximum(result['good'].sum(), 0.5)
    result['woe'] = result.apply(_bucket_woe, axis=1) + np.log(t_good / t_bad)
    result['iv'] = (result['bad'] / t_bad - result['good'] / t_good) * result['woe']
    result = result.reindex(columns=['obs','amin','amax','bad','good','woe','iv'])
    return result

def iv_value(X, y, seg_num=10):
    result = iv_list(X, y, seg_num)
    return result.iv.sum()