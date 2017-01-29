#!/bin/bash

createdb ecobici

psql ecobici -f create_schema.sql

for filename in data/*.csv; do
	echo "`date`: beginning load for ${filename}"
	sed $'s/\\\N//' $filename | psql ecobici -c "COPY trips_raw FROM stdin CSV HEADER;"

	echo "`date`: finished raw load for ${filename}"
done

psql ecobici -f create_trips_from_raw.sql
echo "`date`: populated trips"

psql ecobici -c "TRUNCATE TABLE trips_raw;"

echo "`date`: beginning loading stations"
python load_stations.py
echo "`date`: finished loading stations"

psql ecobici -f create_indexes.sql

psql ecobici -c "UPDATE stations SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);"

python load_stations.py