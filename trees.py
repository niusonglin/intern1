#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:42:47 2017

@author: ushimatsubayashi
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#%%
RB=pd.read_csv('RB.csv',index_col='DateTime')
RB.index=pd.to_datetime(RB.index)
RB=RB.drop('SEC_NAME',axis=1)
RB=RB.drop('WINDCODE',axis=1)
column=RB.columns
#统计分析
from statsmodels.graphics.tsaplots import *
plot_acf(rbret['Today'][rbret.index<start_test],use_vlines=True,lags=30)#自相关偏自相关系数
plot_pacf(rbret['Today'][rbret.index<start_test],use_vlines=True,lags=30)
#%%
#cci与指标之间的关系
rbret['Today'][rbret.index<start_test].plot(linestyle='dashed')
(cci[cci.index<start_test]/100).plot()   #  在测试集上，cci与收益率是有一定的相关性的。
plt.legend()
#%%
#roc与指标之间的关系
(roc[roc.index<start_test]*100).plot(linestyle='dashed')  
rbret['Today'][rbret.index<start_test].plot()   
plt.legend()
#%%
#emv与指标之间的关系
(emv[emv.index<start_test]/10000).plot(c='k',linestyle='dashed')   
rbret['Today'][rbret.index<start_test].plot() 
plt.legend()
#%%
for i in range(len(RB.columns)):
    print(RB.iloc[:,i].describe())

ax1=plt.subplot(2,3,1)
ax1.hist(RB['OPEN'],bins=9,color='r')
ax1.set_title('OPEN')

ax2=plt.subplot(2,3,2)
ax2.hist(RB['HIGH'],bins=9)
ax2.set_title('HIGH')

ax3=plt.subplot(2,3,3)
ax3.hist(RB['LOW'],bins=9,color='g')
ax3.set_title('LOW')

ax4=plt.subplot(2,3,4)
ax4.hist(RB['CLOSE'],bins=9,color='y')
ax4.set_xlabel('CLOSE')

ax5=plt.subplot(2,3,5)
ax5.hist(RB['VOLUME'],bins=6,color='c')
ax5.set_xlabel('VOLUME')

ax6=plt.subplot(2,3,6)
ax6.hist(RB['AMT'],bins=9,color='b')
ax6.set_xlabel('AMT')
#%%
def create_series(symbol):
    RBret= pd.DataFrame(index=symbol.index)
    RBret["Today"] = symbol["CLOSE"].pct_change()*100.0
    for i,x in enumerate(RBret["Today"]):
        if (abs(x) < 0.0001):
            RBret["Today"][i] = 0.0001
    RBret["Direction"] = np.sign(RBret["Today"])
    RBret=RBret.dropna(axis=0,how='any')
    return RBret
#%%
def CCI(symbol, ndays):
    TP = (symbol['HIGH'] + symbol['LOW'] + symbol['CLOSE']) / 3 #中价
    cci = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),name = 'CCI')
    return cci
    #中价与中价的n日内移动平均的差除以n日内中价的平均绝对偏差。0.015为计算系数
def ROC(symbol,ndays):
    N = symbol['CLOSE'].diff(ndays)
    D = symbol['CLOSE'].shift(ndays)
    roc= pd.Series(N/D,name='Rate of Change')
    return roc
def EMV(symbol, ndays): 
    dm = ((symbol['HIGH'] + symbol['LOW'])/2) - ((symbol['HIGH'].shift(1) + symbol['LOW'].shift(1))/2)
    br = (symbol['VOLUME'] / 100000000) / ((symbol['HIGH'] - symbol['LOW']))
    emv = dm / br 
    emv_ma = pd.Series(pd.rolling_mean(emv, ndays), name = 'EVM')  
    return emv_ma
#%%创建x的训练集和测试集
cci=CCI(RB,20)
roc=ROC(RB,5)
emv=EMV(RB,14)
x=pd.concat([RB,cci,roc,emv],axis=1)
start_test = '2016-06-01'
x_train=x[x.index<start_test]
x_test=x[x.index>=start_test]
#y的测试集和训练集
rbret=create_series(RB)
y=rbret["Direction"]
start_test = '2016-06-01'
y_test = y[y.index >= start_test]
y_train=y[y.index < start_test]
y_train=y_train.dropna(axis=0,how='any')
x_train=x_train.dropna(axis=0,how='any')
y_train=y[x_train.index]
y_train
#%%
models = [
              ("RF", RandomForestClassifier(
              	n_estimators=100, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0)
              )]
for m in models:
        
        # Train each of the models on the training set
        m[1].fit(x_train, y_train)

        # Make an array of predictions on the test set
        pred = m[1].predict(x_test)

        # Output the hit-rate and the confusion matrix for each model
        print( (m[0], m[1].score(x_test, y_test)))
        print('confusion_matrix')
        print(  
              
              confusion_matrix(pred, y_test))
#%%
importances = models[0][1].feature_importances_
std = np.std([tree.feature_importances_ for tree in models[0][1].estimators_],
             axis=0)
indices = np.argsort(importances)
print("Feature ranking:")

for f in range(x_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_test.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_test.shape[1]), indices)
plt.xlim([-1, x_test.shape[1]])
plt.show()
