# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:50:19 2019

@author: Colouree
"""

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error

import pandas as pd 
import numpy as np

prop_eval=pd.read_csv('property_evaluation.csv',index_col=[0],encoding='latin-1')
X=prop_eval[['bathrooms','distance','floor','has360',
 'has3DTour','hasLift','hasPlan','hasVideo','newDevelopment','numPhotos',
 'rooms','size','parkingSpace','education','health','food_drink']].astype('float32')
x_names=X.columns
y=prop_eval[['priceByArea']]

X_train, X_test, y_train, y_test = train_test_split(X, y)


regressor = GradientBoostingRegressor(
    max_depth=3,
    n_estimators=6,
    learning_rate=1.0
)
regressor.fit(X_train, y_train)


#errors = [mean_squared_error(y_test, y_pred) for y_pred in regressor.staged_predict(X_test)]
#best_n_estimators = np.argmin(errors)
#
#best_regressor = GradientBoostingRegressor(
#    max_depth=4,
#    n_estimators=best_n_estimators if best_n_estimators>0 else best_n_estimators+1,
#    learning_rate=1.0
#)
#
#
#best_regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse=mean_absolute_error(y_test, y_pred)

