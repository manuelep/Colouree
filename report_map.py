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




mysql.close()
