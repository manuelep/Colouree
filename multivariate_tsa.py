#import required packages
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR


#read the data
df = pd.read_csv("AirQualityUCI.csv",parse_dates=[['Date', 'Time']])#

#check the dtypes
# print(df.head(10))
# print(df.dtypes)
# df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d/%m/%Y %H.%M.%S')
data = df.drop(['Date_Time'], axis=1)
# print(data.head(10))
data.index = df.Date_Time
# print(data.head(10))
#missing value treatment
cols = data.columns
print(cols)
for j in cols:
    for i in range(0,len(data)):
       if data[j][i] == -200:
           data[j][i] = data[j][i-1]
from sklearn.metrics import mean_squared_error
#checking stationarity
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
johan_test_temp = data.drop([ 'CO(GT)'], axis=1)
coint_johansen(johan_test_temp,-1,1).eig
#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]
import math
#check rmse
import numpy as np
# for i in cols:
#     print('rmse value for', i, 'is : ', math.sqrt(mean_squared_error(np.asarray(pred[i]), np.asarray(valid[i]))))
#make final predictions
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print((yhat))
print(np.shape(yhat))

