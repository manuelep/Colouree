# import airbnb
# api = airbnb.Api()
import airbnb
api = airbnb.Api(randomize=True)
# api.get_calendar(listing_id)
# api.get_calendar(975964, starting_month=9, starting_year=2017, calendar_months=1)
import json
# l=api.get_homes(gps_lat=44.4037, gps_lng=8.9058)
l=api.get_listing_details(975964)
k=json.dumps(l)
print(k)#2097152