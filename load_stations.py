import json
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

with open("data/stations_2.json", "r") as f:
    stations = json.load(f)['stations']

df = pd.DataFrame(stations)
df['lat'] = df['location'].apply(lambda x: x['lat'])
df['lon'] = df['location'].apply(lambda x: x['lon'])
new_df = df[['id', 'name', 'address', 'districtCode', 'districtName','zipCode', 'lat', 'lon', 'stationType']]
new_df.columns = ['id', 'name', 'address', 'district_code', 'district_name','zip', 'latitude', 'longitude', 'stationtype']

engine = create_engine('postgresql://simonbedford@localhost/ecobici')
new_df.to_sql("stations", engine, index=False, if_exists='append')