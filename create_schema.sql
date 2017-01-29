CREATE EXTENSION postgis;

CREATE TABLE trips_raw(
	Genero_Usuario char(1),
	Edad_Usuario int,
	Bici varchar,
	Ciclo_Estacion_Retiro int,
	Fecha_Retiro date,
	Hora_Retiro time,
	Ciclo_Estacion_Arribo int,
	Fecha_Arribo date,
	Hora_Arribo time
);

CREATE TABLE trips(
	id serial primary key,
	trip_duration numeric,
	start_time timestamp without time zone,
	end_time timestamp without time zone,
	start_station_id int,
	end_station_id int,
	bike_id varchar,
	gender char(1),
	age int
);

CREATE TABLE stations(
	id integer primary key,
	name varchar,
	address varchar,
	district char(3),
	zip char(5),
	latitude numeric,
	longitude numeric,
	status char(3),
	bikes int,
	stationType varchar
);

SELECT AddGeometryColumn('stations', 'geom', 4326, 'POINT', 2);
CREATE INDEX idx_stations_on_geom ON stations USING gist (geom);