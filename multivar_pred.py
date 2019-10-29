import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import scale
from TFANN import ANNR


# dc_listings = pd.read_csv('listings.csv')
# print(dc_listings.shape)
# dc_listings.head()

df = pd.read_csv("AirQualityUCI.csv",parse_dates=[['Date', 'Time']]).iloc[:,3:5]#

#check the dtypes
# print(df.head(10))
# print(df.dtypes)
# df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%d/%m/%Y %H.%M.%S')
# data = df.drop(['Date_Time'], axis=1)

# df.index = df.Date_Time
# df=df.drop(['Date_Time'], axis=1)
print(df.head(10))
########################################################################################

# data.index = df.Date_Time
# # print(data.head(10))
# #missing value treatment
# cols = data.columns

#reads data from the file and ceates a matrix with only the dates and the prices 
# stock_data = np.loadtxt('ZBH_5y.csv', delimiter=",", skiprows=1, usecols=(1, 4))
#scales the data to smaller values
stock_data=scale(df)
#gets the price and dates from the matrix
prices = stock_data[:, 1].reshape(-1, 1)
dates = stock_data[:, 0].reshape(-1, 1)
#creates a plot of the data and then displays it
print(dates)
plt.plot(dates[:, 0], prices[:, 0])
plt.show()


#Number of neurons in the input, output, and hidden layers
input2 = 1
output2 = 1
hidden2 = 50
#array of layers, 3 hidden and 1 output, along with the tanh activation function 
#array of layers, 3 hidden and 1 output, along with the tanh activation function 
layers = [('F', hidden2), ('AF', 'tanh'), ('F', hidden2), ('AF', 'tanh'), ('F', hidden2), ('AF', 'tanh'), ('F', output2)]
#construct the model and dictate params
mlpr = ANNR([input2], layers, batchSize = 256, maxIter = 20000, tol = 0.1, reg = 1e-4, verbose = True)
holdDays = 5
totalDays = len(dates)
#fit the model to the data "Learning"
mlpr.fit(dates, prices)
#Predict the stock price using the model
pricePredict = mlpr.predict(dates)
#Display the predicted reuslts agains the actual data
plt.plot(dates, prices)
plt.plot(dates, pricePredict, c='#5aa9ab')
plt.show()




