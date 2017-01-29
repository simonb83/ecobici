CREATE TABLE IF NOT EXISTS trip_legs_new(
	start_station_id integer,
	end_station_id integer,
	leg_number integer,
	start_latitude numeric,
	start_longitude numeric,
	end_latitude numeric,
	end_longitude numeric,
	leg_time_taken numeric,
	leg_distance numeric
);