CREATE TABLE bike_station_ids AS
SELECT
	id,
	bike_id,
	start_time,
	end_time,
	end_station_id,
	lead(start_station_id, 1) OVER w AS next_start_station_id,
	lead(start_time, 1) OVER w AS next_start_time
FROM trips
WINDOW w AS (PARTITION BY bike_id ORDER BY start_time)
ORDER BY bike_id, start_time;

DELETE FROM bike_station_ids WHERE next_start_station_id IS NULL;

# Station aggregates table
CREATE TABLE station_aggregates AS
SELECT
	end_station_id,
	COUNT(*)::numeric AS total_drop_offs,
	SUM((end_station_id != next_start_station_id)::int) AS transported_to_other_station
FROM bike_station_ids
GROUP BY end_station_id;

CREATE TABLE monthly_station_aggregates AS
SELECT
	date(date_trunc('month', end_time)) AS month,
	end_station_id,
	COUNT(*)::numeric AS total_drop_offs,
	SUM((end_station_id != next_start_station_id)::int) AS transported_to_other_station
FROM bike_station_ids
GROUP BY end_station_id, month;

CREATE TABLE hourly_aggregate AS
SELECT
	COUNT(*) AS total_trips,
	COUNT(DISTINCT( date(start_time))) AS number_of_days,
	start_station_id = end_station_id AS circular,
	EXTRACT(HOUR FROM start_time) AS hour,
	EXTRACT(DOW FROM start_time) BETWEEN 1 AND 5 AS weekday
FROM trips
WHERE date(start_time) >= '2015-01-01' AND date(start_time) < '2016-08-01'
GROUP BY circular, hour, weekday;

CREATE TABLE anonymous_analysis_hourly AS
SELECT
  date_trunc('hour', start_time) truncated_to_hour,
  start_station_id,
  gender,
  age,
  COUNT(*) count
FROM trips
WHERE
  gender IN ('M', 'F')
  AND age IS NOT NULL
  AND age < 85
GROUP BY truncated_to_hour, start_station_id, gender, age;

CREATE TABLE google_estimates AS
SELECT
	SUM(leg_time_taken) AS trip_duration,
	SUM(leg_distance) AS trip_length,
	start_station_id,
	end_station_id
FROM trip_legs_new
GROUP BY start_station_id, end_station_id;

CREATE TABLE avg_speed AS
SELECT 
	AVG ( google_estimates.trip_length / trips.trip_duration ) AS speed,
	trips.gender AS gender,
	trips.age AS age,
	EXTRACT (DOW FROM trips.start_time) BETWEEN 1 AND 5 AS weekday
FROM trips, google_estimates
WHERE date(trips.start_time) >= '2015-01-01' AND date(trips.start_time) < '2016-08-01'
AND trips.age < 80
AND trips.start_station_id = google_estimates.start_station_id
AND trips.end_station_id = google_estimates.end_station_id
AND trips.start_station_id != trips.end_station_id
AND trips.trip_duration >= 60
GROUP BY gender, age, weekday;

