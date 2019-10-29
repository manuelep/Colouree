import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import argparse
import re
import logging
import requests
import json
overpass_url=""
oerpass_query="""
[out:json];
area["ISO3166-1"="DE"][admin_level=2];
(node["amenity"="biergarten"](area);
);
out center;
"""
response=requests.get(overpass_url,params={'data': overpass_query})
data=response.json()
print(data)
print("type",type(data))







