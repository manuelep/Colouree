import pandas as pd
import numpy as np
import re
import json
import requests
import time
from tqdm import tqdm
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

eurostat=pd.read_csv('house_price_index.tsv',delimiter='\t')
df1=pd.read_csv('idealista_data_milan.csv',index_col=0)
#df1 = pd.DataFrame()
#df1['latitude']=0
#df1['longitude']=0
#k=0

dx=pd.read_csv('only_amenity_tags1.csv',header=0,index_col=0).fillna(0)
import pandas as pd
dx1=pd.read_csv('only_amenity_tags.csv',encoding='latin-1')
amenity=dx1['amenity']
amenity=list(amenity)
distance=dx1['distance']
distance=list(distance)
all_tags=pd.read_csv(r'C:\Users\Colouree\Desktop\Colouree\idealista_tags1.csv',header=None,sep=',')

#for l in enumerate(all_tags):
#    ind=l[0]
#    i=l[1]
#    i=i.split(',')
##    i=i.replace()
#    i=[str(i).lstrip() for i in i]
#    i=[str(i).strip() for i in i]
#    for j in i:
#        j=str(j)
#        if j in amenity:
#            dx.loc[ind,j]+=1

#dx=dx.replace(0,np.nan)
#dx=dx.dropna(how='all')
#dx=dx.replace(np.nan,0)
#dx['latitude']=0
#dx['longitude']=0
#for i in enumerate(all_lats):
#    dx.loc[i[0],'latitude']=i[1]
#for i in enumerate(all_longs):
#    dx.loc[i[0],'longitude']=i[1]

#for i in df['json']:
#    i = json.loads(i)
#    lat=df.loc[k,'latitude']
#    lon=df.loc[k,'longitude']
#    df1=df1.append(i,ignore_index=True)
#    df1.loc[k,'latitude']=lat
#    df1.loc[k,'longitude']=lon
#    k+=1

k=0
df1['floor'].replace('[^0-9]', 0, regex=True,inplace=True)
df1['floor'].fillna(0,inplace=True)
df1['parkingSpace'].fillna(0,inplace=True)
df1['parkingSpace'].astype(str).replace('[^0]', 1, regex=True,inplace=True)
for j in df1['parkingSpace']:
    if str(j)=='0':
       df1.loc[k,'parkingSpace']=0
    else:
       df1.loc[k,'parkingSpace']=1  
    k+=1
for i in enumerate(df1['propertyType']):
    if str(i[1])=='flat':
        df1.loc[i[0],'propertyType']=1
    elif str(i[1])=='studio':
        df1.loc[i[0],'propertyType']=0
    elif str(i[1])=='chalet':
        df1.loc[i[0],'propertyType']=2
    elif str(i[1])=='duplex':
        df1.loc[i[0],'propertyType']=3
    elif str(i[1])=='penthouse':
        df1.loc[i[0],'propertyType']=4
    
for i in enumerate(df1['status']):
    if str(i[1])=='good':
        df1.loc[i[0],'status']=1
    elif str(i[1])=='nostatus':
        df1.loc[i[0],'status']=0
    elif str(i[1])=='renew':
        df1.loc[i[0],'status']=2
    else:
        df1.loc[i[0],'status']=3

##################       OVERPASS QUERY            ##########################
##import overpy
##api = overpy.Overpass()
##result = api.query("""<osm-script output="json" timeout="60"><union into="_"><query into="_" type="node"><has-kv k="qwerty" modv="not" regv="."/><bbox-query e="8.95385742188" n="44.4151452431" s="44.3925796184" w="8.92227172852"/></query></union><print e="" from="_" geometry="skeleton" limit="" mode="body" n="" order="id" s="" w=""/><recurse from="_" into="_" type="down"/><print e="" from="_" geometry="skeleton" limit="" mode="meta" n="" order="id" s="" w=""/></osm-script>""")
##lll=result.nodes
##-------------------------------------------------------------------######
     
#k=0
#all_tags=[]
#all_lats=[]
#all_longs=[]
#jj=True
#for i in tqdm(range(0,len(df1['latitude']))):
##with tqdm(total=k) as pbar:
#    while jj:
#        try:
#            lat=df1.loc[k,'latitude']
#            lon=df1.loc[k,'longitude']
#            req=requests.get('http://overpass-api.de/api/interpreter?data=[out:json];(node[%22amenity%22](around:300,'+str(lat)+','+str(lon)+'););out;%3E;')
#            req=req.json()['elements']
#            tags=[]
#            lats=[]
#            longs=[]
#            for l in range(0,len(req)):
#                 tag=req[l]['tags']['amenity']
#                 tags.append(tag)
#                 la=req[l]['lat']
#                 lo=req[l]['lon']
#                 lats.append(la)
#                 longs.append(lo)
#            all_tags.append(tags)
#            all_lats.append(lats)
#            all_longs.append(longs)
#            k+=1
#            break
#        except:
#            time.sleep(8)
#            jj=True

import pickle
#with open("all_tags.txt","wb") as fp:
#    pickle.dump(all_tags,fp)
#with open("all_lats.txt","wb") as fp:
#    pickle.dump(all_lats,fp)
#with open("all_longs.txt","wb") as fp:
#    pickle.dump(all_longs,fp)
with open("all_tags.txt","rb") as fp:
    all_tags=pickle.load(fp)
with open("all_lats.txt","rb") as fp:
    all_lats=pickle.load(fp)
with open("all_longs.txt","rb") as fp:
    all_longs=pickle.load(fp)

##########################################
#---------------getting distance between two coordinates------#
from math import sin,cos,sqrt,atan2,radians
def get_distance(lat1,lon1,lat2,lon2):
    R=6373.0
    lat1=radians(lat1)
    lon1=radians(lon1)
    lat2=radians(lat2)
    lon2=radians(lon2)
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 +cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c=2*atan2(sqrt(a),sqrt(1-a))
    distance=R*c
    return distance*1000
#-----------------------------------------------------------------#




#-counting the amenities and lat and long-#
for l in enumerate(all_tags):
    ind=l[0]
    i=l[1]
    i=i#.split(',')
#    i=i.replace()
    i=[str(i).lstrip() for i in i]
    i=[str(i).strip() for i in i]
    for j in i:
        j=str(j)
        if j in amenity:
            dx.loc[ind,j]+=1








dx=dx.replace(0,np.nan)
dx=dx.dropna(how='all')
dx=dx.replace(np.nan,0)
#dx['latitude']=0
#dx['longitude']=0
#for i in enumerate(all_lats):
##    l=i[1]
##    l=l.split(',')
##    for j in l:
#        
#    dx.loc[i[0],'latitude']=i[1]
#for i in enumerate(all_longs):
#    dx.loc[i[0],'longitude']=i[1]

p=0
all_dis=[]
for i,j,k in zip(all_lats,all_longs,all_tags):
    lax=i
    lox=j
    tax=k
    latx=df1.loc[p,'latitude']
    lonx=df1.loc[p,'longitude']
    c=0
    diss=[]
    for x,y in zip(lax,lox):
        dis=get_distance(latx,lonx,x,y)
        diss.append(dis)
        c+=1
    all_dis.append(diss)
    p+=1




dx1=dx.copy()
#-distance the amenities and lat and long-#
for l in enumerate(all_tags):
    ind=l[0]
    i=l[1]
    i=i#.split(',')
#    i=i.replace()
    i=[str(i).lstrip() for i in i]
    i=[str(i).strip() for i in i]
    for k,j in enumerate(i):
        dist=all_dis[ind][k]
        j=str(j)
        if j in amenity:
            dx1.loc[ind,j]+=dist

#dx1=pd.DataFrame(preprocessing.normalize(dx1.astype('float32')))           
dist_dx=[str(x)+'_dist' for x in dx.columns]
#dx1=pd.DataFrame(dx1,columns=dx.columns) 
dx1=pd.DataFrame(dx1.values/dx.values,columns=dx.columns)
dx1=dx1.replace(np.nan,0)
#final_dx=pd.concat([dx,dx2],axis=1)
#final_dx=final_dx.replace(np.nan,0)
#
##-counting the amenities and lat and long-#
#for l in tqdm(enumerate(all_tags)):
#    ind=l[0]
#    i=l[1]
##    i=i.split(',')
##    i=i.replace()
#    i=[str(i).lstrip() for i in i]
#    i=[str(i).strip() for i in i]
#    for amen in distance:
##    for l in enumerate(all_tags):
#
##        for j in tqdm(i):
##            nnn=str(j)+"_dis"
#    #        print(nnn)
##            indices=[]
#        indices = [y for y, x in enumerate(i) if str(x)+'_dis'==str(amen)]
##            for y,x in tqdm(enumerate(i)):
##                if str(amen)==str(nnn):
###                    print(y)
##                    indices.append(y)
#        disx=0
#        for inds in indices:
#            disx+=all_dis[ind][inds]
#        dx.loc[ind,str(amen)]+=(disx)/(len(indices)+0.001)
##                dx.loc[ind,str(amen)]=
#    #        indices = [y for y, x in enumerate(i) if x == nnn] 
#    #        print(indices)
#    #        dx.loc[]
#    #        if j in amenity:
#    #            dx.loc[ind,j]+=1

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


#amenitiy_count=dx.iloc[:,0:75]
#amenity_distance=dx.iloc[:,74:]
#
#
#amenity_df=pd.DataFrame(data={'tags':all_tags,'latitude':all_lats,'longitude':all_longs})








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
#df2=pd.read_csv('idealista_data.csv',index_col=0)
df2=df1#pd.read_csv('idealista_data_milan.csv',index_col=0)
df2['education']=0
df2['health']=0
df2['food_drink']=0
df2['business']=0
df2['public_spaces']=0
df2['residential']=0
df2['leisure']=0
df2['living']=0
df2['transportation']=0

tags_corr=pd.read_csv(r'C:\Users\Colouree\Desktop\Colouree\newest_tags_relations2.csv',index_col=0)#only_amenity_tags_corr
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
#numeric_attr_names=[
# 'education',
# 'health',
# 'food_drink',
# 'business',
# 'public_spaces',
# 'residential',
# 'leisure','transportation'
#]#'latitude','longitude', 'exterior',,'living','transportation'
#numeric_attr_names= ['bathrooms',
# 'distance',
# 'floor',
# 'has360',
# 'has3DTour',
# 'hasLift',
# 'hasPlan',
# 'hasVideo',
# 'newDevelopment',
# 'numPhotos',
# 'rooms',
# 'showAddress',
# 'size',
# 'parkingSpace']

numeric_attr_names=['bathrooms','floor','rooms','parkingSpace','propertyType','size','status']

#bin_att=['parkingSpace','propertyType']
#binary_feat=df2[bin_att]
#req_normalize_att=['size']
#req_nor_feat=df2[req_normalize_att]
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#req_nor_feat_scaler=scaler.fit(req_nor_feat)
#req_nor_feat=req_nor_feat_scaler.transform(req_nor_feat)
#size=pd.DataFrame({'size':req_nor_feat})
#numeric_target=['price']
# 'price','priceByArea',

#categorical_attr_names=['frenns_id','unique_frenns_id','paid','type','name', 'address', 'postcode', 'city', 'contact_person', 'email','invoice_number', 'currency','invoiceId', 'customerId', 'updateId','pay_date','issue_date', 'due_date', 'collection_date', 'creation_date', 'last_updated']
#numeric_attr = df2[numeric_attr_names].astype('float32') + 1e-7
#numeric_attr = numeric_attr.apply(np.log)
#ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())
#ori_dataset_numeric_attr.fillna(0,inplace=True)
#ori_dataset_categ_transformed = pd.get_dummies(df[categorical_attr_names])
#ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)
#print(ori_subset_transformed.head(10))

numeric_target=df2['priceByArea']
df4=pd.concat([df2[numeric_attr_names],dx],axis=1).fillna(0).astype('int')
#df4=dx.fillna(0).astype('int')
df5=pd.concat([dx,df2['priceByArea']],axis=1).fillna(0).astype('int')
df6=pd.concat([df4,df2['priceByArea']],axis=1).fillna(0).astype('int')
df7=preprocessing.normalize(df6)
df8=pd.DataFrame(data=df7[0:,0:],index=df6.index,columns =df6.columns)
idealista_corr=df8.corr(method='pearson').fillna(0)
sorted_pba=idealista_corr['priceByArea'].sort_values()

#import matplotlib.pyplot as plt
#correlations = idealista_corr
## plot correlation matrix
#fig = plt.figure(figsize=(50, 50))
#ax = fig.add_subplot(111)
#cax = ax.matshow(correlations, vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticksx = np.arange(0,len(correlations.index),1)
#ticksy = np.arange(0,len(correlations.columns),1)
#ax.set_xticks(ticksx)
#ax.set_yticks(ticksy)
#ax.set_xticklabels(correlations.index,rotation=45)
#ax.set_yticklabels(correlations.columns)
##plt.show()
#plt.savefig('idealista_correlation_matrix.png')



#X=df2[numeric_attr_names].astype('float32')
X=df4.astype('float32')
y=numeric_target.astype('float32')
X[np.isnan(X)] = 0
y[np.isnan(y)] = 0
#from sklearn import preprocessing
#std_scale = preprocessing.StandardScaler().fit(numeric_target)
#y = std_scale.transform(y)
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#std_scale=scaler.fit(df2['priceByArea'])
#y=std_scale.transform(df2['priceByArea'])
#y=pr eprocessing.scale(y)
#from sklearn.preprocessing import MinMaxScaler 
#scaler = MinMaxScaler(feature_range=(0, 1)) 
#X = scaler.fit_transform(X) 
#y=preprocessing.normalize(y)


#from sklearn.preprocessing import StandardScaler
#numeric=df2[numeric_attr_names].astype('float32')
#scalerX = StandardScaler().fit(numeric)
#X = scalerX.transform(numeric)



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

#X=pd.DataFrame(data=X[0:,0:],
#                index=[i for i in range(X.shape[0])],
#                columns=[str(i) for i in range(X.shape[1])])
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
#model1 = ExtraTreesClassifier()
#model1.fit(X,y)
#print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
##plot graph of feature importances for better visualization
#feat_importances = pd.Series(model1.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()
#feat_importances=feat_importances.nlargest(10)
#index_feat=list(feat_importances.index)
#selected_features=pd.DataFrame(data={'Value':feat_importances,'Number':index_feat},index=index_feat)
#selected_features=selected_features.replace(0,np.nan).dropna()
#selected_features1=list(selected_features['Number'])

y=y.astype(int)
import pickle
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model1 = ExtraTreesClassifier()
model1.fit(df4,y.astype(int))
print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model1.feature_importances_, index=df4.columns)
#feat_importances_1=feat_importances.nlargest(45)
#feat_importances_2=feat_importances.nlargest(15)
#selected_features=list(feat_importances_1.index)+list(feat_importances_2.index)
feat_csv=pd.read_csv('feature_corr.csv',index_col=0,header=0)

##########################################
#------NORMAZLIZE THE DATA --------------#
X_for_corr=preprocessing.normalize(df4.astype('float32'))
#----------------------------------------#
##########################################
#X_for_corr=df4.astype('float32')
for i in df4.columns:
    model1 = ExtraTreesClassifier()
    model1.fit(X_for_corr,df4[i].astype(int))
    corr_temp=model1.feature_importances_
    for j,k in enumerate(df4.columns):
        feat_csv.loc[k,i]=corr_temp[j]

for i in ['priceByArea']:
    model1 = ExtraTreesClassifier()
    model1.fit(X_for_corr,y.astype(int))
    corr_temp=model1.feature_importances_
    for j,k in enumerate(feat_csv.columns):#pd.concat([df4,df5['priceByArea']],axis=1).columns):
        try:
            feat_csv.loc[k,i]=corr_temp[j]
        except:
            pass
selected_features=[]
for i in enumerate(feat_importances):
    if i[1]>0:
        selected_features.append(feat_importances.index[i[0]])



import statsmodels.api as sm
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(df4)
y1=df5['priceByArea']
#Fitting sm.OLS model
model = sm.OLS(y1,X_1).fit()
model.pvalues
#Backward Elimination
cols = list(dx.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = dx[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y1,X_1).fit()
    try:
        p = pd.Series(model.pvalues.values[1:],index = cols)      
    except:
        p = pd.Series(model.pvalues.values[0:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


#index_feat=list(feat_importances.index)
#selected_features=pd.DataFrame(data={'Value':feat_importances,'Number':index_feat},index=index_feat)
#selected_features=selected_features.replace(0,np.nan).dropna()
#selected_features=list(selected_features['Number'])
with open("selected_features.txt","wb") as fp:
    pickle.dump(selected_features,fp)
#with open("selected_features.txt","rb") as fp:
#    selected_features=pickle.load(fp)
X=X[selected_features]
##########################################
#------NORMAZLIZE THE DATA --------------#
#X=preprocessing.normalize(X)
#----------------------------------------#
##########################################
X=np.asarray(X)


##############################################################################
##############################################################################
#-------CCA (canonical correlation analysis ) ---------------#
from sklearn.cross_decomposition import CCA
cca = CCA(n_components=1)
#for i,j in enumerate(X):
#    cca.fit(X[:,i:i+1],y)
#    x_t,y_t=cca.transform(X, y)
#    X_c.append(x_t)
#    Y_c.append(y_t)
cca.fit(X, y)
X_c, Y_c = cca.transform(X, y)
  
##############################################################################
##############################################################################






##############################################################################
##############################################################################

#K means clustering
from sklearn.cluster import KMeans
clusters=25
kmeans=KMeans(n_clusters=clusters)
kmeans.fit(X)
#from sklearn.cluster import KMeans
#from scipy.spatial.distance import cdist
#
#def plot_kmeans(kmeans, X, n_clusters=20, rseed=0, ax=None):
#    labels = kmeans.fit_predict(X)
#
#    # plot the input data
#    ax = ax or plt.gca()
#    ax.axis('equal')
#    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
#
#    # plot the representation of the KMeans model
#    centers = kmeans.cluster_centers_
#    radii = [cdist(X[labels == i], [center]).max()
#             for i, center in enumerate(centers)]
#    for c, r in zip(centers, radii):
#        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
#kmeans = KMeans(n_clusters=4, random_state=0)
#plot_kmeans(kmeans, X)
#y=kmeans.labels_

##############################################################################
##############################################################################

# training gaussian mixture model 
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=25)
gmm.fit(X)
gmm_labels = gmm.predict(X)
##############################################################################
##############################################################################







# Fuzzy c means

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist

class FCM:
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42):
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        N = X.shape[0]
        C = self.n_clusters
        centers = []

        # u = np.random.dirichlet(np.ones(C), size=N)
        r = np.random.RandomState(self.random_state)
        u = r.rand(N,C)
        u = u / np.tile(u.sum(axis=1)[np.newaxis].T,C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(X, u)
            u = self.next_u(X, centers)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.u = u
        self.centers = centers
        return self
    def fit_predict(self, X):
        return self.fit(X).u.argmax(axis=1)
    def next_centers(self, X, u):
        um = u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def next_u(self, X, centers):
        return self._predict(X, centers)

    def _predict(self, X, centers):
        power = float(2 / (self.m - 1))
        temp = cdist(X, centers) ** power
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X, self.centers)
        return np.argmax(u, axis=-1)

from seaborn import scatterplot as scatter
# fit the fuzzy-c-means
#no_clus=40*len(df4)/1000
fcm = FCM(n_clusters=25)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)
#fcm.fit_predict(df4)

# plot result
#%matplotlib inline
#f, axes = plt.subplots(1, 2, figsize=(11,5))
#scatter(X[:,0], X[:,1], ax=axes[0])
#scatter(X[:,0], X[:,1], ax=axes[1], hue=fcm_labels)
#scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
#plt.show()
##############################################################################
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10)
X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0
Y_train[np.isnan(Y_train)] = 0
Y_test[np.isnan(Y_test)] = 0
#print('###########################################################################')
#print('         Multiclass classification model             ')
#print('###########################################################################')
#----------------------------------#
#from sklearn.datasets import make_classification
#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.utils import shuffle
#X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
#y2 = shuffle(y1, random_state=1)
#y3 = shuffle(y1, random_state=2)
#Y = np.vstack((y1, y2, y3)).T
#n_samples, n_features = X.shape # 10,100
#n_outputs = Y_train.shape[1] # 3
#n_classes = 31
#forest = RandomForestClassifier(n_estimators=100, random_state=1)
#multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
#multi_target_forest.fit(X, Y).predict(X)
#--------------------------------------------------#
#from sklearn.svm import SVC
#svm_model_linear=SVC(kernel='rbf',C=100,gamma=0.01).fit(X_train,Y_train)
#svm_predictions=svm_model_linear.predict(X_test)
#accuracy=svm_model_linear.score(X_test,Y_test)
##------Saving scikit learn model#
#from sklearn.externals import joblib
## save the model to disk
#filename = 'svc_model.sav'
#joblib.dump(svm_model_linear, filename)



# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)
##############################################################################
##############################################################################
##############################################################################








##############################################################################
print('###########################################################################')
print('          Neural Network')
print('###########################################################################')


      

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
#model.add(Dense(1024, activation='tanh',init='uniform', input_dim=np.shape(X_train)[1]))
#model.add(Dense(512, activation='tanh',init='uniform', input_dim=np.shape(X_train)[1]))
#model.add(Dense(256, activation='tanh'))
#model.add(Dense(128, activation='tanh'))
#model.add(Dense(64, activation="tanh"))
#model.add(Dense(32, activation="relu"))
#model.add(Dense(1))

model.add(Dense(256, activation='relu',init='uniform', input_dim=np.shape(X_train)[1]))
model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])
        
# fit model
model.fit(X_train, Y_train, epochs=600, batch_size=128,verbose=0)
# demonstrate prediction
x_input = X[[2]]
yhat = model.predict(X_test, verbose=0)
#print(yhat)
diff=np.subtract(yhat.reshape(yhat.shape[0]),Y_test)
diff=abs(diff)
maxx=max(diff)
print(maxx)
mean_pred=np.mean(diff)
print(mean_pred)
std_pred=np.std(diff)
print(std_pred)


####################################################################################
json_file_path="property_model.json"
h5_file="property_model.h5"
            
            
            

#----------------Saving the model-------------#
##Serielixe to  Json
#json_file=model.to_json()
#with open(json_file_path,"w") as file:
#    file.write(json_file)
##Serialize weights to HDF5
#model.save_weights(h5_file)
##-----------xxxxxxxxxxxxxxxxx----------------#

#from keras.models import model_from_json
##load json and create model
#file=open(json_file_path,'r')
#model_json=file.read()
#file.close()
#
#loaded_model=model_from_json(model_json)
##load weights
#loaded_model.load_weights(h5_file)





reg = LassoCV()
reg.fit(X,numeric_target)#.drop(['restaurant'],axis=1)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,numeric_target))#.drop(['restaurant'],axis=1)
coef = pd.Series(reg.coef_, index = df4[selected_features].columns)#.drop(['restaurant'],axis=1)
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
##X=X[['3','4','6']]
##X=X[['2','17','9','1','12','16','20']]

#################################################################################
##       feature importance                  ##############################
#################################################################################
#from sklearn.ensemble import ExtraTreesClassifier
#import matplotlib.pyplot as plt
#model1 = ExtraTreesClassifier()
#model1.fit(X,y)
#print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
##plot graph of feature importances for better visualization
#feat_importances = pd.Series(model1.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()
#index_feat=list(feat_importances.index)
#selected_features=pd.DataFrame(data={'Value':feat_importances,'Number':index_feat},index=index_feat)
#selected_features=selected_features.replace(0,np.nan).dropna()
#selected_features=list(selected_features['Number'])
############     Correlation map         ##################################
#correlation_attributes=[
# 'education',
# 'health',
# 'food_drink',
# 'business',
# 'public_spaces',
# 'residential',
# 'leisure',
# 'bathrooms',
# 'distance',
#  'size',
# 'parkingSpace',
# 'rooms',
# 'numPhotos','priceByArea']
#import seaborn as sns
#
##X1=df2[correlation_attributes].astype('float32')
#
#X1=df5.astype('float32')
#X1[np.isnan(X)] = 0
##get correlations of each features in dataset
#corrmat = X1.corr()
#corrmat=corrmat.fillna(0)
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
##plot heat map
#g=sns.heatmap(X1[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()
###############################################################################
#print('###########################################################################')
#print('          Neural Network')
#print('###########################################################################')
#X=np.asarray(X)
#
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)
#X_train[np.isnan(X_train)] = 0
#X_test[np.isnan(X_test)] = 0
#Y_train[np.isnan(Y_train)] = 0
#Y_test[np.isnan(Y_test)] = 0
#
#
##15 25 1 31 11 5 19
##model = Sequential()
##model.add(Dense(256, input_dim=np.shape(X)[1], init='uniform', activation='sigmoid'))
###model.add(Dense(512, input_dim=3, init='normal', activation="sigmoid"))
###model.add(Dense(10, activation="sigmoid"))
##model.add(Dense(64, activation="sigmoid"))
###model.add(Dense(32, activation="relu"))
##model.add(Dense(1, activation="softmax"))
##model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
##EPOCHS=100
##H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=EPOCHS, batch_size=32)
#
#
#
#
#
#
#
#
#
## define model
#model = Sequential()
#model.add(Dense(1024, activation='tanh',init='uniform', input_dim=np.shape(X_train)[1]))
#model.add(Dense(512, activation='tanh'))
#model.add(Dense(256, activation='tanh'))
#model.add(Dense(128, activation='tanh'))
#model.add(Dense(64, activation="tanh"))
#model.add(Dense(32, activation="relu"))
#model.add(Dense(1))
#model.compile(optimizer='adam', loss='mean_squared_error')
## fit model
#model.fit(X_train, Y_train, epochs=2000, verbose=0)
## demonstrate prediction
#x_input = X[[2]]
#yhat = model.predict(X_test, verbose=0)
##print(yhat)
#diff=np.subtract(yhat,Y_test)
#diff=abs(diff)
#maxx=max(diff['priceByArea'])
#print(maxx)
#mean_pred=np.mean(diff)
#print(mean_pred)
#std_pred=np.std(diff)
#print(std_pred)
#
####################################################################################
######           END PART TWO               #######################################
#################################################################################
#
#
#
#
#########        KNN CLASSIFIER              #####################
##from sklearn import preprocessing
##from sklearn.neighbors import KNeighborsClassifier
##from sklearn.metrics import accuracy_score
##mm_scaler = preprocessing.MinMaxScaler()
##train_x =X_train# mm_scaler.fit_transform(X_train)
##test_x =X_test# mm_scaler.fit_transform(X_test)
##train_y=Y_train
##test_y=Y_test
##
##'''
##Create the object of the K-Nearest Neighbor model
##You can also add other parameters and test your code here
##Some parameters are : n_neighbors, leaf_size
##Documentation of sklearn K-Neighbors Classifier: 
##
##https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
##
## '''
##model = KNeighborsClassifier()  
##
### fit the model with the training data
##model.fit(X_train,Y_train)
##
### Number of Neighbors used to predict the target
##print('\nThe number of neighbors used to predict the target : ',model.n_neighbors)
##
### predict the target on the train dataset
##predict_train = model.predict(X_train)
##print('\nTarget on train data',predict_train) 
##
### Accuray Score on train dataset
##accuracy_train = accuracy_score(Y_train,predict_train)
##print('accuracy_score on train dataset : ', accuracy_train)
#
### predict the target on the test dataset
##predict_test = model.predict(test_x)
##print('Target on test data',predict_test) 
##
### Accuracy Score on test dataset
##accuracy_test = accuracy_score(test_y,predict_test)
##print('accuracy_score on test dataset : ', accuracy_test)
#
############################################################################
##############   GradientBoostingRegressor     ####################################
############################################################################
#print('###########################################################################')
#print('          GradientBoostingRegressor')
#print('###########################################################################')
#
#reg = LinearRegression()
#reg.fit(X_train,Y_train)
#reg.score(X_test,Y_test)
#from sklearn import ensemble
#clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
#          learning_rate = 0.1, loss = 'ls')
#clf.fit(X_train, Y_train)
#clf.score(X_test,Y_test)
#y_pred = clf.predict(X_test)
#diff=np.subtract(y_pred[0],Y_test)
#diff=abs(diff)
#maxx=max(diff['priceByArea'])
#print(maxx)
#mean_pred=np.mean(diff)
#print(mean_pred)
#std_pred=np.std(diff)
#print(std_pred)
#
#
#
#
#
#
############################################################################
##############   LINEAR REGRESSION       ####################################
############################################################################
#print('###########################################################################')
#print('          LINEAR REGRESSION ')
#print('###########################################################################')
#from sklearn.linear_model import LinearRegression
#clf = LinearRegression()
#clf.fit(X_train, Y_train)
#prediction = (clf.predict(X_test))
#diff=np.subtract(prediction,Y_test)
#diff=abs(diff)
#maxx=max(diff['priceByArea'])
#print(maxx)
#mean_pred=np.mean(diff)
#print(mean_pred)
#std_pred=np.std(diff)
#print(std_pred)
##########################################################################
######## Findind best parameters using gridsearchcv        ############
#############################################################################
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVR
#Cs = [0.001, 0.01, 0.1, 1, 10]
#gammas = [0.001, 0.01, 0.1, 1]
#param_grid = {'C': Cs, 'gamma' : gammas}
#parameters = {'kernel':['rbf', 'sigmoid'], 'C':np.logspace(np.log10(0.001), np.log10(200), num=20), 'gamma':np.logspace(np.log10(0.00001), np.log10(2), num=30)}
#grid_searcher_red = GridSearchCV(SVR(kernel='rbf'), parameters, n_jobs=8, verbose=2)
#
##grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
#grid_searcher_red.fit(X_train, Y_train)
#params=grid_searcher_red.best_params_
#
#
#
#
############################################################################
##############   SVM       ####################################
############################################################################
#print('###########################################################################')
#print('          Support Vector Machine ')
#print('###########################################################################')
##from sklearn.svm import SVR
##from sklearn.svm import SVC
#svclassifier = SVR(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])
#svclassifier.fit(X_train, Y_train)
#y_pred1 = svclassifier.predict(X_test)
#diff=np.subtract(y_pred1[0],Y_test)
#diff=abs(diff)
#maxx=max(diff['priceByArea'])
#print(maxx)
#mean_pred=np.mean(diff)
#print(mean_pred)
#std_pred=np.std(diff)
#print(std_pred)
##from sklearn.metrics import classification_report, confusion_matrix
##print(confusion_matrix(y_test, y_pred))
##print(classification_report(y_test, y_pred))


#############################################################################
#########            Geometric Plotting          ################################
#############################################################################

#
#import pandas as pd
#import matplotlib.pyplot as plt
#import descartes
#import geopandas as gpd
#from shapely.geometry import Point, Polygon
#street_map=gpd.read_file(r'C:\Users\Colouree\Desktop\Colouree\Italy_shapefile\it_10km.shp')
#
#fig,ax=plt.subplots(figsize=(15,15))
#street_map.plot(ax=ax)
#
#
#
#
#
#
#













