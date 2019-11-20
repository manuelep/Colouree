# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:24:05 2019

@author: Colouree
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.externals import joblib
with open("selected_features.txt","rb") as fp:
    selected_features=pickle.load(fp)
dx=pd.read_csv('only_amenity_tags1.csv',header=0,index_col=0).fillna(0)
dx1=pd.read_csv('only_amenity_tags.csv',encoding='latin-1')
amenity=dx1['amenity']
amenity=list(amenity)
distance=dx1['distance']
distance=list(distance)
k=0
all_tags=[]
jj=True
lat=45.4460134
lon=9.1815607
import requests,time
all_lats=[]
all_longs=[]
while jj:
    try:
        req=requests.get('http://overpass-api.de/api/interpreter?data=[out:json];(node[%22amenity%22](around:300,'+str(lat)+','+str(lon)+'););out;%3E;')
        req=req.json()['elements']
        tags=[]
        lats=[]
        longs=[]
        for l in range(0,len(req)):
             tag=req[l]['tags']['amenity']
             tags.append(tag)
             la=req[l]['lat']
             lo=req[l]['lon']
             lats.append(la)
             longs.append(lo)
        all_tags.append(tags)
        all_lats.append(lats)
        all_longs.append(longs)
        break
    except:
        time.sleep(8)
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
#    l=i[1]
#    l=l.split(',')
#    for j in l:
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
    latx=lat
    lonx=lon
    c=0
    diss=[]
    for x,y in zip(lax,lox):
        dis=get_distance(latx,lonx,x,y)
        diss.append(dis)
        c+=1
    all_dis.append(diss)
    p+=1
#
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
            
            
            
dist_dx=[str(x)+'_dist' for x in dx.columns]
dx1=pd.DataFrame(dx1.values/dx.values,columns=dx.columns)

##-counting the amenities and lat and long-#
#for l in enumerate(all_tags):
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
            
#for l in enumerate(all_tags):
#    ind=l[0]
#    i=l[1]
##    i=i.split(',')
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

dx['rooms']=3
dx['floor']=4
dx['bathrooms']=2             
dx['parkingSpace']=1
dx['size']=132
dx['propertyType']=4
dx['status']=1
#dx['distance']=84
#dx['numPhotos']=28
X=dx[selected_features]
X=X.astype('float32')

X[np.isnan(X)] = 0
import time
start = time.time()
from sklearn import preprocessing
#X=preprocessing.normalize(X)
#from sklearn.preprocessing import MinMaxScaler 
#scaler = MinMaxScaler(feature_range=(0, 1)) 
#X = scaler.fit_transform(X) 
json_file_path="property_model.json"
h5_file="property_model.h5"
from keras.models import model_from_json
#load json and create model
file=open(json_file_path,'r')
model_json=file.read()
file.close()

loaded_model=model_from_json(model_json)
#load weights
loaded_model.load_weights(h5_file)
yhat = loaded_model.predict(X, verbose=0) - 1984 if loaded_model.predict(X, verbose=0)>5500 else loaded_model.predict(X, verbose=0)
print("real prediction: \t",loaded_model.predict(X, verbose=0))
print("modified prediction: \t",yhat)
print("took {} secs to run the analysis".format(time.time()-start))


### load the model from disk
#filename = 'svc_model.sav'
#loaded_model = joblib.load(filename)
#result = loaded_model.predict(X)
#print(result)


