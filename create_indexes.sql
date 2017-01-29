CREATE INDEX idx_trips_on_start_station ON trips (start_station_id);
CREATE INDEX idx_trips_on_end_station ON trips (end_station_id);
CREATE INDEX idx_trips_on_dow ON trips (EXTRACT(DOW FROM start_time));
CREATE INDEX idx_trips_on_hour ON trips (EXTRACT(HOUR FROM start_time));
CREATE INDEX idx_trips_on_year ON trips (EXTRACT(YEAR FROM start_time));
CREATE INDEX idx_trips_on_date ON trips (date(start_time));
CREATE INDEX idx_trips_on_bike_id ON trips (bike_id);
CREATE INDEX idx_trips_on_gender ON trips (gender);
CREATE INDEX idx_trips_on_age ON trips (age);