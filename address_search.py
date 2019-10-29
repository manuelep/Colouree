# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:15:31 2019

@author: Colouree
"""

import requests
import json
#address="Via Terpi 19"
address="Parc Naturel Régional du Verdon"
#def search(address):
#    try:
import requests
import json
import re
lang='fr'
if lang=='en':
    regex = re.compile('[^a-zA-Z0-9]')
    address=regex.sub(' ', address)
sq="https://nominatim.openstreetmap.org/search?q="+str(address.replace(' ','%20'))+"&format=geojson&limit=1&accept-language="+str(lang)+""
req=requests.get(sq).content.decode('utf-8')
req=json.loads(req)#['features'][0]['geometry']['coordinates']

#        return req
#    except:
#        req='Invalid address'
#        return req

#import sys
#arg=sys.argv[1]
#result=search(address)
#if result!='Invalid address':
    