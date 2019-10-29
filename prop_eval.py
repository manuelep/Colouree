
import pandas as pd
import numpy as np
import re
import json
import requests
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#df=pd.read_csv('idealista.csv')
#df1 = pd.DataFrame()
#df1['latitude']=0
#df1['longitude']=0
#k=0
#
#for i in df['json']:
#    i = json.loads(i)
#    lat=df.loc[k,'latitude']
#    lon=df.loc[k,'longitude']
#    df1=df1.append(i,ignore_index=True)
#    df1.loc[k,'latitude']=lat
#    df1.loc[k,'longitude']=lon
#    k+=1
#
#k=0
#df1['floor'].replace('[^0-9]', 0, regex=True,inplace=True)
#df1['floor'].fillna(0,inplace=True)
#df1['parkingSpace'].fillna(0,inplace=True)
#df1['parkingSpace'].astype(str).replace('[^0]', 1, regex=True,inplace=True)
#for j in df1['parkingSpace']:
#    if str(j)=='0':
#       df1.loc[k,'parkingSpace']=0
#    else:
#       df1.loc[k,'parkingSpace']=1  
#    k+=1
#    
##################       OVERPASS QUERY            ##########################
##import overpy
##api = overpy.Overpass()
##result = api.query("""<osm-script output="json" timeout="60"><union into="_"><query into="_" type="node"><has-kv k="qwerty" modv="not" regv="."/><bbox-query e="8.95385742188" n="44.4151452431" s="44.3925796184" w="8.92227172852"/></query></union><print e="" from="_" geometry="skeleton" limit="" mode="body" n="" order="id" s="" w=""/><recurse from="_" into="_" type="down"/><print e="" from="_" geometry="skeleton" limit="" mode="meta" n="" order="id" s="" w=""/></osm-script>""")
##lll=result.nodes
##-------------------------------------------------------------------######
#     
#k=0
#all_tags=[]
#jj=True
#for i in tqdm(range(0,len(df1['latitude']))):
##with tqdm(total=k) as pbar:
#    while jj:
#        try:
#            lat=df1.loc[k,'latitude']
#            lon=df1.loc[k,'longitude']
#            req=requests.get('http://overpass-api.de/api/interpreter?data=[out:json];(node[%22amenity%22](around:200,'+str(lat)+','+str(lon)+'););out;%3E;')
#            req=req.json()['elements']
#            tags=[]
#            for l in range(0,len(req)):
#                 tag=req[l]['tags']['amenity']
#                 tags.append(tag)
#            all_tags.append(tags)
#            k+=1
#            break
#        except:
#            time.sleep(8)
#            jj=True

################################################################################
#############           MULTIVARIATE PREDICTIONS     ###########################
#import statsmodels.api as sm # import statsmodels 
#X = df1["RM"] ## X usually means our input variables (or independent variables)
#y = target["MEDV"] ## Y usually means our output/dependent variable
#X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
#
## Note the difference in argument order
#model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
#predictions = model.predict(X)
#
## Print out the statistics
#model.summary()
################################################################################
#-----------------------------------------------------------------------------#















###############################################################################
            ####    end OF part One                 #####################
###############################################################################
def get_nearest_tags(tag,df1,nearest_words):
    df1=df1.astype('float32')
#    current_tag=df1.loc[tag].sort_values(ascending=False)[1:nearest_words+1]
#    current_tag=cr_df.loc[cr_df.index.values.astype(str)[0]].sort_values(ascending=False)[1:nearest_words+1]
#    df2=pd.DataFrame(current_tag,columns=tag,index=current_tag.index)
    return df1.loc[tag].sort_values(ascending=False)[1:nearest_words+1]

edu=0
heal=0
food=0
bsns=0
public=0
resid=0
leis=0
liv=0
trans=0
df2=pd.read_csv('idealista_data.csv',index_col=0)

df2['education']=0
df2['health']=0
df2['food_drink']=0
df2['business']=0
df2['public_spaces']=0
df2['residential']=0
df2['leisure']=0
df2['living']=0
df2['transportation']=0

tags_corr=pd.read_csv(r'C:\Users\Colouree\Desktop\Colouree\only_amenity_tags_corr.csv',index_col=0)
new_tags=[]
all_tags=pd.read_csv(r'C:\Users\Colouree\Desktop\Colouree\idealista_tags.csv')
education=get_nearest_tags('education',tags_corr,7)
education=[x.strip() for x in list(education.index)]
health=get_nearest_tags('health',tags_corr,7)
health=[x.strip() for x in list(health.index)]
food_drink=get_nearest_tags('food drink',tags_corr,7)
food_drink=[x.strip() for x in list(food_drink.index)]
business=get_nearest_tags('business',tags_corr,7)
business=[x.strip() for x in list(business.index)]
public_spaces=get_nearest_tags('public spaces',tags_corr,7)
public_spaces=[x.strip() for x in list(public_spaces.index)]
residential=get_nearest_tags('residential',tags_corr,7)
residential=[x.strip() for x in list(residential.index)]
leisure=get_nearest_tags('leisure',tags_corr,7)
leisure=[x.strip() for x in list(leisure.index)]
living=get_nearest_tags('living',tags_corr,7)
living=[x.strip() for x in list(living.index)]
transportation=get_nearest_tags('transportation',tags_corr,7)
transportation=[x.strip() for x in list(transportation.index)]

for i in range(0,len(all_tags['tags'])):
    tag1=all_tags.loc[i,'tags'].split(',')
    tag1=[x.replace(' ','' ) for x in tag1]
    tag1=[x.replace('_',' ') for x in tag1]
    new_tags.append(tag1)
    edu=[x for x in tag1 if x in education]
    heal=[x for x in tag1 if x in health]
    food=[x for x in tag1 if x in food_drink]
    bsns=[x for x in tag1 if x in business]
    public=[x for x in tag1 if x in public_spaces]
    resid=[x for x in tag1 if x in residential]
    leis=[x for x in tag1 if x in leisure]
    liv=[x for x in tag1 if x in living]
    trans=[x for x in tag1 if x in transportation]
    edu=len(edu)
    heal=len(heal)
    food=len(food)
    bsns=len(bsns)
    public=len(public)
    resid=len(resid)
    leis=len(leis)
    liv=len(liv)
    trans=len(trans)
    df2.loc[i,'education']=edu
    df2.loc[i,'health']=heal
    df2.loc[i,'food_drink']=food
    df2.loc[i,'business']=bsns
    df2.loc[i,'public_spaces']=public
    df2.loc[i,'residential']=resid
    df2.loc[i,'leisure']=leis
    df2.loc[i,'living']=liv
    df2.loc[i,'transportation']=trans
df2['status'].fillna(0,inplace=True)
#df2['status'].replace('good')
numeric_attr_names=[
 'education',
 'health',
 'food_drink',
 'business',
 'public_spaces',
 'residential',
 'leisure']#'latitude','longitude', 'exterior',,'living','transportation'
""" 'bathrooms',
 'distance',
 'floor',
 'has360',
 'has3DTour',
 'hasLift',
 'hasPlan',
 'hasVideo',
 'newDevelopment',
 'numPhotos',
 'rooms',
 'showAddress',
 'size',
 'parkingSpace']"""

numeric_target=['priceByArea']
numeric_target=df2[numeric_target]# 'price','priceByArea',

#categorical_attr_names=['frenns_id','unique_frenns_id','paid','type','name', 'address', 'postcode', 'city', 'contact_person', 'email','invoice_number', 'currency','invoiceId', 'customerId', 'updateId','pay_date','issue_date', 'due_date', 'collection_date', 'creation_date', 'last_updated']
#numeric_attr = df2[numeric_attr_names].astype('float32') + 1e-7
#numeric_attr = numeric_attr.apply(np.log)
#ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())
#ori_dataset_numeric_attr.fillna(0,inplace=True)
#ori_dataset_categ_transformed = pd.get_dummies(df[categorical_attr_names])
#ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)
#print(ori_subset_transformed.head(10))
from sklearn import preprocessing
X=df2[numeric_attr_names].astype('float32')
y=numeric_target.astype('float32')
X[np.isnan(X)] = 0
y[np.isnan(y)] = 0
X=preprocessing.normalize(X)
#y=preprocessing.normalize(y)



#from sklearn.preprocessing import StandardScaler
#numeric=df2[numeric_attr_names].astype('float32')
#scalerX = StandardScaler().fit(numeric)
#X = scalerX.transform(numeric)
#
#
#numeric_target=numeric_target.astype('float32')
#scalery = StandardScaler().fit(numeric_target)
#y = scalery.transform(numeric_target)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# split a univariate sequence into samples
#def split_sequence(sequence, n_steps):
#	X, y = list(), list()
#	for i in range(len(sequence)):
#		# find the end of this pattern
#		end_ix = i + n_steps
#		# check if we are beyond the sequence
#		if end_ix > len(sequence)-1:
#			break
#		# gather input and output parts of the pattern
#		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#		X.append(seq_x)
#		y.append(seq_y)
#	return array(X), array(y)
#
## define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
## choose a number of time steps
#n_steps = 3
## split into samples
#X, y = split_sequence(raw_seq, n_steps)



#model = Sequential([
#    Dense(128, activation='relu', input_shape=(np.shape(X_train)[1],)),
#    Dense(64, activation='relu'),
#    Dense(32, activation='relu'),
#    Dense(1, activation='relu'),
#])
#model.compile(optimizer='adam',
#              loss='mean_squared_error',
#              metrics=['accuracy'])
#
#hist = model.fit(X_train, Y_train,
#          batch_size=32, epochs=100
#         )
#
#accuracy=model.evaluate(X_test, Y_test)[1]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import matplotlib.pyplot as plt

X=pd.DataFrame(data=X[0:,0:],
                index=[i for i in range(X.shape[0])],
                columns=[str(i) for i in range(X.shape[1])])

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
#X=X[['3','4','6']]
#X=X[['2','17','9','1','12','16','20']]

X1=X
################################################################################
#       feature importance                  ##############################
################################################################################
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model1 = ExtraTreesClassifier()
model1.fit(X1,y)
print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model1.feature_importances_, index=X1.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
###########     Correlation map         ##################################
import seaborn as sns
#get correlations of each features in dataset
corrmat = X1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(X1[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
##############################################################################
X=np.asarray(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1)
X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0
Y_train[np.isnan(Y_train)] = 0
Y_test[np.isnan(Y_test)] = 0


#15 25 1 31 11 5 19
#model = Sequential()
#model.add(Dense(256, input_dim=np.shape(X)[1], init='uniform', activation='sigmoid'))
##model.add(Dense(512, input_dim=3, init='normal', activation="sigmoid"))
##model.add(Dense(10, activation="sigmoid"))
#model.add(Dense(64, activation="sigmoid"))
##model.add(Dense(32, activation="relu"))
#model.add(Dense(1, activation="softmax"))
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#EPOCHS=100
#H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=EPOCHS, batch_size=32)








 
# define model
model = Sequential()
#model.add(Dense(256, activation='tanh',init='uniform', input_dim=np.shape(X_train)[1]))
model.add(Dense(128, activation='tanh',init='uniform', input_dim=np.shape(X_train)[1]))
model.add(Dense(64, activation="tanh"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
# fit model
model.fit(X_train, Y_train, epochs=2000, verbose=0)
# demonstrate prediction
x_input = X[[2]]
yhat = model.predict(X_test, verbose=0)
#print(yhat)
diff=np.subtract(yhat,Y_test)
diff=abs(diff)
maxx=max(diff['priceByArea'])
print(maxx)
mean_pred=np.mean(diff)
print(mean_pred)
std_pred=np.std(diff)
print(std_pred)

###################################################################################
#####           END PART TWO               #######################################
################################################################################




########        KNN CLASSIFIER              #####################
#from sklearn import preprocessing
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
#mm_scaler = preprocessing.MinMaxScaler()
#train_x =X_train# mm_scaler.fit_transform(X_train)
#test_x =X_test# mm_scaler.fit_transform(X_test)
#train_y=Y_train
#test_y=Y_test
#
#'''
#Create the object of the K-Nearest Neighbor model
#You can also add other parameters and test your code here
#Some parameters are : n_neighbors, leaf_size
#Documentation of sklearn K-Neighbors Classifier: 
#
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#
# '''
#model = KNeighborsClassifier()  
#
## fit the model with the training data
#model.fit(X_train,Y_train)
#
## Number of Neighbors used to predict the target
#print('\nThe number of neighbors used to predict the target : ',model.n_neighbors)
#
## predict the target on the train dataset
#predict_train = model.predict(X_train)
#print('\nTarget on train data',predict_train) 
#
## Accuray Score on train dataset
#accuracy_train = accuracy_score(Y_train,predict_train)
#print('accuracy_score on train dataset : ', accuracy_train)

## predict the target on the test dataset
#predict_test = model.predict(test_x)
#print('Target on test data',predict_test) 
#
## Accuracy Score on test dataset
#accuracy_test = accuracy_score(test_y,predict_test)
#print('accuracy_score on test dataset : ', accuracy_test)

###########################################################################
#############   GradientBoostingRegressor     ####################################
###########################################################################


reg = LinearRegression()
reg.fit(X_train,Y_train)
reg.score(X_test,Y_test)
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(X_train, Y_train)
clf.score(X_test,Y_test)
y_pred = clf.predict(X_test)
diff=np.subtract(y_pred[0],Y_test)
diff=abs(diff)
maxx=max(diff['priceByArea'])
print(maxx)
mean_pred=np.mean(diff)
print(mean_pred)
std_pred=np.std(diff)
print(std_pred)






###########################################################################
#############   LINEAR REGRESSION       ####################################
###########################################################################
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, Y_train)
prediction = (clf.predict(X_test))
diff=np.subtract(prediction,Y_test)
diff=abs(diff)
maxx=max(diff['priceByArea'])
print(maxx)
mean_pred=np.mean(diff)
print(mean_pred)
std_pred=np.std(diff)
print(std_pred)
#########################################################################
####### Findind best parameters using gridsearchcv        ############
############################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
parameters = {'kernel':['rbf', 'sigmoid'], 'C':np.logspace(np.log10(0.001), np.log10(200), num=20), 'gamma':np.logspace(np.log10(0.00001), np.log10(2), num=30)}
grid_searcher_red = GridSearchCV(SVR(kernel='rbf'), parameters, n_jobs=8, verbose=2)

#grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
grid_searcher_red.fit(X_train, Y_train)
params=grid_searcher_red.best_params_




###########################################################################
#############   SVM       ####################################
###########################################################################

#from sklearn.svm import SVR
#from sklearn.svm import SVC
svclassifier = SVR(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])
svclassifier.fit(X_train, Y_train)
y_pred1 = svclassifier.predict(X_test)
diff=np.subtract(y_pred1[0],Y_test)
diff=abs(diff)
maxx=max(diff['priceByArea'])
print(maxx)
mean_pred=np.mean(diff)
print(mean_pred)
std_pred=np.std(diff)
print(std_pred)
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))


############################################################################
########            TEXT GOES HERE          ################################
############################################################################
























