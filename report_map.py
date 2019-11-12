# -*- coding: utf-8 -*-

import re
import numpy as np
import mysql.connector
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import mysql.connector
from pandas import read_sql
import config
import requests,json,time
mysql = mysql.connector.connect(
  host=config.HOST,
  user=config.USER,
  passwd=config.PSWD,
  database = config.FRENNS_NAME2
)

frn_id='FRN100000811'
df = read_sql('select * from syncreport_pl where frenns_id = "'+str(frn_id)+'" ',con=mysql)

account=df['detail_acc_type']
regex = re.compile('[^a-zA-Z]')
account=list(filter(None.__ne__, account))
keys=[regex.sub(' ', i) for i in account]
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)+' '
    return result

#print(concatenate_list_data([1, 5, 12, 2]))
    
keys=list(set(keys))
##lll=[x for x in keys if 'wheelchair' in x]
keys=[concatenate_list_data(k.split(' ')[0:3]) for k in keys]
macro_tags=['profit']
from gensim.models import KeyedVectors
model1=KeyedVectors.load(r"C:\Users\Colouree\Desktop\Colouree\google_word2vec.model")
#import psutil
## gives a single float value
#psutil.cpu_percent()
## gives an object with many fields
#psutil.virtual_memory()
## you can convert that object to a dictionary 
#mem_usage=dict(psutil.virtual_memory()._asdict())
df2=pd.DataFrame(columns=macro_tags,index=keys)
from tqdm import tqdm
for i in tqdm(range(0,len(keys))):
    for j in range(0,len(macro_tags)):
        try:
            df2.iloc[i,j]=model1.similarity(df2.index[i], df2.columns[j])
        except:
            try:
                df2.iloc[i,j]=model1.n_similarity(df2.index[i].lower().split(), df2.columns[j].lower().split())
            except:
                df2.iloc[i,j]=0.0

                
mysql.close()
