import pandas as pd
import json
import urllib
import requests as rq
import base64,time

def get_oauth_token():
    url = "https://api.idealista.com/oauth/token"    
    apikey= 'fh5cja5cky5hjsj1j5vqrdjvgj05jmmr' #sent by idealista
    secret= '3pNm1nFNVWRK'  #sent by idealista
    lll=apikey + ':' + secret
    lll=lll.encode('utf-8')
    auth = base64.b64encode(lll)
    headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' ,'Authorization' : 'Basic ' + auth.decode('utf-8')}
    params = urllib.parse.urlencode({'grant_type':'client_credentials'})
    content = rq.post(url,headers = headers, params=params)
    bearer_token = json.loads(content.text)['access_token']
    # print(bearer_token)
    return bearer_token

def search_api(token, url):  
    headers = {'Content-Type': 'Content-Type: multipart/form-data;', 'Authorization' : 'Bearer ' + token}
    content = rq.post(url, headers = headers)
    print(content)
    result = json.loads(content.text)#['access_token']
    # print(result)
    return result

country = 'it' #values: es, it, pt
locale = 'it' #values: es, it, pt, en, ca
language = 'en' #
max_items = '100'
operation = 'sale' 
property_type = 'homes'
order = 'distance' 
center = '45.462939,9.187840'
distance = '10000'
sort = 'asc'
bankOffer = 'false'

df_tot = pd.DataFrame()
limit = 10

for i in range(1,limit):
    while True:
        try:
            url = ('https://api.idealista.com/3.5/'+country+'/search?operation='+operation+#"&locale="+locale+
                   '&maxItems='+max_items+
                   '&order='+order+
                   '&center='+center+
                   '&distance='+distance+
                   '&propertyType='+property_type+
                   '&sort='+sort+ 
                   '&numPage=%s'+
                   '&language='+language) %(i)  
            a = search_api(get_oauth_token(), url)
            df = pd.DataFrame.from_dict(a['elementList'])
            df_tot = pd.concat([df_tot,df])
            break
        except:
            time.sleep(300)

df_tot = df_tot.reset_index()
df_tot.to_csv('idealista_data_milan.csv')