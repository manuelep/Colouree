# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:53:10 2019

@author: Colouree
"""
import os
from os.path import join, dirname
FRENNS_NAME= "vkingsol_frennsdevelopment"
CLEARSIGHT_NAME="clearsight_development"
HOST = "144.76.137.232"
USER= "vkingsol_demo"
PSWD= "gUj3z5?9"


import csv, sys, os, pytz, datetime, json, re
import warnings
#import settings, database
#from pyramid.arima import auto_arima
from random import random
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import mysql.connector
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA 
#import config
#import matplotlib.pyplot as plt   

mysql = mysql.connector.connect(
  host=HOST,
  user=USER,
  passwd=PSWD,
  database = FRENNS_NAME

      )
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
# univariate multi-step vector-output stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import warnings
frn_id='FRN100000757'
df=pd.read_sql("select DISTINCT name from syncinvoice where frenns_id = '{0}'".format(frn_id), con=mysql)
name=df['name'].values.astype(str)
#yyy=[]
for i in name:
    qr=pd.read_sql("""SELECT  syncinvoice_id, date(issue_date) as issue_date, date(due_date) as due_date, date(collection_date) as pay_date, COALESCE(DATEDIFF(date(collection_date),date(issue_date)),DATEDIFF(date(NOW()),date(issue_date))) as after_issue_days FROM `syncinvoice` where frenns_id = '{0}' and name = "{1}" order by issue_date desc """.format(frn_id,i),con=mysql)
    ser=qr[['issue_date','after_issue_days']]
    ser.to_csv("series.csv")
    series=pd.read_csv("series.csv",header=0,parse_dates=[0],index_col=1,squeeze=True).drop(['Unnamed: 0'],axis=1)
    series=series.astype('float32')
    os.remove("series.csv")
    days=series['after_issue_days']
    days=np.array(days)
##    print(np.array(days))
##    print("#######################################################################")
#    n_steps_in, n_steps_out = 3, 2
#    # split into samples
#    if len(days)>6:
#        X, y = split_sequence(days, n_steps_in, n_steps_out)
#        print(X)
#        # reshape from [samples, timesteps] into [samples, timesteps, features]
#        n_features = 1
#        X = X.reshape((X.shape[0], X.shape[1], n_features))
#        # define model
#        model = Sequential()
#        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
#        model.add(LSTM(100, activation='relu'))
#        model.add(Dense(n_steps_out))
#        model.compile(optimizer='adam', loss='mse')
#        # fit model
#        model.fit(X, y, epochs=50, verbose=0)
#        # demonstrate prediction
#        x_input = array([70, 80, 90])
#        x_input = x_input.reshape((1, n_steps_in, n_features))
#        yhat = model.predict(x_input, verbose=0)
#        yyy.append(yhat)
#    print(yhat)
#    print("#######################################################################")    
#######################################################################
        #######################################################################
##########      Outlier Detection in a multivariate Dataset     ############
#######################################################################
#
#import numpy as np 
#from scipy import stats 
#import matplotlib.pyplot as plt 
#import matplotlib.font_manager 
#import pyod
#from pyod.models.knn import KNN  
#from pyod.utils.data import generate_data, get_outliers_inliers 
#dt=pd.read_sql("select amount,outstanding_amount,vat_amount from ML where frenns_id = '{0}'".format(frn_id), con=mysql)
#days=dt['amount']
#days=np.array(days)
#
#dt=np.array(dt)
#mod=pyod.models.ocsvm.OCSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, contamination=0.1)
##mod=pyod.models.knn.KNN(contamination=0.1, n_neighbors=5, method='largest', radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1)
##mod1=pyod.models.pca.PCA(n_components=None, n_selected_components=None, contamination=0.1, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, weighted=True, standardization=True)
#
##outlier=mod.fit_predict(X=dt, y=None)
#outlier=mod.fit_predict(X=dt, y=None)
#
##print(mod.threshold_)#.std())
##outlier=pd.DataFrame(data=outlier)
#    # values    # 1st column as index)
#import collections
##a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
#counter=collections.Counter(outlier)
#ccc=list(counter.values())
#minimum=min(ccc)
#if minimum<0.03*len(outlier):
#    classs=1
#else:
#    classs=0
######################################################################################{}

import itertools
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
final_param=[]
final_param_seasonal=[]
final_aic=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(days,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            final_param.append(param)
            final_param_seasonal.append(param_seasonal)
            final_aic.append(results.aic)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

index=int(final_aic.index(max(final_aic)))
model=sm.tsa.statespace.SARIMAX(days,
                                        order=final_param[index],
                                        seasonal_order=final_param_seasonal[index],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
results = model.fit()
forecast=results.forecast( steps=1) 
predict=results.predict(start=len(days))

#########################################################################################

#outlier_grouping=outlier.groupby([0]).agg([0])
## generating a random dataset with two features 
#X_train, y_train = generate_data(n_train = 300, train_only = True, 
#                                                   n_features = 2) 
#  
## Setting the percentage of outliers 
#outlier_fraction = 0.1
#  
## Storing the outliers and inliners in different numpy arrays 
#X_outliers, X_inliers = get_outliers_inliers(X_train, y_train) 
#n_inliers = len(X_inliers) 
#n_outliers = len(X_outliers) 
#  
## Seperating the two features 
#f1 = X_train[:, [0]].reshape(-1, 1) 
#f2 = X_train[:, [1]].reshape(-1, 1) 




## define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
## choose a number of time steps
#n_steps_in, n_steps_out = 3, 2
## split into samples
#X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
#print(X)
## reshape from [samples, timesteps] into [samples, timesteps, features]
#n_features = 2
#X = X.reshape((X.shape[0], X.shape[1], n_features))
## define model
#model = Sequential()
#model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
#model.add(LSTM(100, activation='relu'))
#model.add(Dense(n_steps_out))
#model.compile(optimizer='adam', loss='mse')
## fit model
#model.fit(X, y, epochs=50, verbose=0)
## demonstrate prediction
#x_input = array([70, 80, 90])
#x_input = x_input.reshape((1, n_steps_in, n_features))
#yhat = model.predict(x_input, verbose=0)
#print(yhat)




mysql.close()