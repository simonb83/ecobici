-- This function is thanks to http://rexdouglass.com/spatial-hexagon-binning-in-postgis/

CREATE TABLE hex_grid (gid serial not null primary key);
SELECT addgeometrycolumn('hex_grid','the_geom', 0, 'POLYGON', 2); 

CREATE OR REPLACE FUNCTION genhexagons(width float, xmin float,ymin  float,xmax float,ymax float  )
RETURNS float AS $total$
declare
	b float :=width/2;
	a float :=b/2; --sin(30)=.5
	c float :=2*a;
	height float := 2*a+c;  --1.1547*width;
	ncol float :=ceil(abs(xmax-xmin)/width);
	nrow float :=ceil(abs(ymax-ymin)/height);

	polygon_string varchar := 'POLYGON((' ||
	                                    0 || ' ' || 0     || ' , ' ||
	                                    b || ' ' || a     || ' , ' ||
	                                    b || ' ' || a+c   || ' , ' ||
	                                    0 || ' ' || a+c+a || ' , ' ||
	                                 -1*b || ' ' || a+c   || ' , ' ||
	                                 -1*b || ' ' || a     || ' , ' ||
	                                    0 || ' ' || 0     ||
	                            '))';
BEGIN
    INSERT INTO hex_grid (the_geom) SELECT st_translate(the_geom, x_series*(2*a+c)+xmin, y_series*(2*(c+a))+ymin)
    from generate_series(0, ncol::int , 1) as x_series,
    generate_series(0, nrow::int,1 ) as y_series,
    (
       SELECT polygon_string::geometry as the_geom
       UNION
       SELECT ST_Translate(polygon_string::geometry, b , a+c)  as the_geom
    ) as two_hex;
    ALTER TABLE hex_grid
	ALTER COLUMN the_geom TYPE geometry(Polygon, 4326)
	USING ST_SetSRID(the_geom,4326);
    RETURN NULL;
END;
$total$ LANGUAGE plpgsql;

--width in the units of the projection, xmin,ymin,xmax,ymax
SELECT genhexagons(0.0065,-99.22,19.33,-99.12,19.45);
-- SELECT genhexagons(0.0065,19.33, -99.22,19.45,-99.12);

ALTER TABLE hex_grid ADD COLUMN hexagon_id integer;
UPDATE hex_grid SET hexagon_id = gid;
ALTER TABLE stations ADD COLUMN hexagon_id integer;

UPDATE stations
SET hexagon_id = hex_grid.hexagon_id
FROM hex_grid
WHERE ST_Within(stations.geom, hex_grid.the_geom);

