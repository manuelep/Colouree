import requests
import json
# overpass_url = "http://overpass-api.de/api/interpreter"
# overpass_query = """
# [out:json];
# area["ISO3166-1"="DE"][admin_level=2];
# (node["amenity"="biergarten"](area);
#  way["amenity"="biergarten"](area);
#  rel["amenity"="biergarten"](area);
# );
# out center;
# """
# overpass_query="""[out:json][timeout:25];
# (
#   node["amenity"="post_box"];
# );
# out body;
# >;
# out skel qt;"""
# response = requests.get(overpass_url, 
#                         params={'data': overpass_query})
# data = response.json()
# import overpy
# api=overpy.Overpass()
########################################
# import overpass
# # api = overpass.API()
# api = overpass.API()
# response = api.get('node["name"="Salt Lake City"]')
# print(response)

###########################################
#GET /api/0.6/[node|way|relation]/#id/history
sq="https://api.openstreetmap.org/api/0.6/[nodes|ways|relations]/25496783/relations"
req=requests.get(sq).content.decode('utf-8')
print(req)

