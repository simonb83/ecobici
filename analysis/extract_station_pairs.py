import psycopg2
import pandas as pd

conn = psycopg2.connect("dbname='ecobici' user='simonbedford' host='localhost'")
cur = conn.cursor()

# Get the station ids for the zone of interest
sql = """SELECT DISTINCT(id) FROM stations
    WHERE hexagon_id >= 280 OR hexagon_id IN (248, 250, 252, 254, 256, 258);"""
cur.execute(sql)
station_ids = cur.fetchall()
station_ids = tuple([i[0] for i in station_ids])

# Get the trips based on the following conditions
# 1. start and end station ids are in the specified zone based on station ids above
# 2. trip starts and ends at a different station
# 3. trip takes place on 2016-04-06
# 4. Only get distinct routes
sql = """SELECT DISTINCT(start_station_id::text || '-' || end_station_id::text) AS route, 
    start_station_id AS start_id, end_station_id AS end_id
    FROM trips WHERE date(start_time) = '2016-04-06'
    AND EXTRACT(HOUR FROM start_time) >= 5
    AND start_station_id != end_station_id
    AND start_station_id IN {} AND end_station_id IN {};""".format(station_ids, station_ids)
cur.execute(sql)
distinct_trips = pd.DataFrame(cur.fetchall())
distinct_trips.columns = [d[0] for d in cur.description]

distinct_trips =  distinct_trips.drop('route', axis=1)
distinct_trips.to_csv('output/leg_station_ids.csv', index_label='index')