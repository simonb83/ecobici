CREATE TABLE leg_trips AS
SELECT trips.id, trip_legs.start_latitude, trip_legs.start_longitude, trip_legs.end_latitude, trip_legs.end_longitude
FROM trips, trip_legs_new AS trip_legs
WHERE trips.start_station_id = trip_legs.start_station_id
AND trips.end_station_id = trip_legs.end_station_id
AND date(trips.start_time) = '2016-04-06'
AND trips.start_station_id != trips.end_station_id
AND trips.start_station_id <= 452 AND trips.end_station_id <= 452;

SELECT AddGeometryColumn('leg_trips', 'start_geom', 4326, 'POINT', 2);
SELECT AddGeometryColumn('leg_trips', 'end_geom', 4326, 'POINT', 2);

UPDATE leg_trips
SET start_geom = ST_SetSRID(ST_MakePoint(start_longitude, start_latitude), 4326);

UPDATE leg_trips
SET end_geom = ST_SetSRID(ST_MakePoint(end_longitude, end_latitude), 4326);

SELECT AddGeometryColumn('leg_trips', 'line_geom', 4326, 'LINESTRING', 2);
UPDATE leg_trips
SET line_geom = ST_SetSRID(ST_MakeLine(start_geom, end_geom), 4326);


CREATE TABLE leg_trips_counts AS
SELECT COUNT(*) AS num_legs, line_geom FROM leg_trips
GROUP BY line_geom
ORDER BY num_legs ASC;