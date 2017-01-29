### 38 Million Ecobici Rides

Accompanying code for blog post: [ecobocic](www.)

#### Instructions

***Getting set up***

1. Download data

```bash
python get_data.py
```

2. Clean the start and end dates for some months (this is necessary because for most months the dates are in the format yyyy-mm-dd, however for three months in 2016 they are in the format dd/mm/yyyy)

```bash
python clean_dates.py
```

3. Initialize the database and load the data

```bash
./load_trips.sh
```

***Obtaining Google Maps Instructions***

1. Add the table for saving the individual trip legs from Google Maps:

```bash
psql ecobici -f create_trip_legs.sql
```

2. Extract the relevant routes for obtaining directions from Google:

```bash
mkdir output
python extract_station_pairs.py
```

3. Use the `google_instructions.py` script to obtain directions from Google Maps for each distinct route (free account only allows for 2,500 queries per day)

```
python google_instructions.py -i 'Starting Index' -t 'Num Trips to Process'
```

***Analysis***

1. Prepare for analysis

```
psql ecobici -f prepare_analysis.sql
```

2. Count the different legs based on the Google Maps directions to identify most popular routes:

```bash
psql ecobici -f count_trip_legs.sql
pgsql2shp -f "line_segments/line_segments" -h 'HOST' -u 'USER' ecobici "SELECT * FROM leg_trips_counts ORDER BY num_legs ASC;"
```

3. Run analysis for Day in The Life and Popular Routes:

```bash
python day_in_the_life.py
```

4. Create hexagon bin-tiling and extract shapefile with relevant hexagons: 

```bash
psql ecobici -f hexagon_tiling.sql
mkdir hexagons
pgsql2shp -f "hexagons/hexagons" -h 'HOST' -u 'USER' ecobici "SELECT hexagon_id, ST_Force2D(the_geom) FROM hex_grid WHERE hexagon_id IN (SELECT DISTINCT(hexagon_id) FROM stations);"
```

5. Core analysis:

```bash
python analysis.py
```

5. Streets and Colonias shapefile


