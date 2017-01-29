"""
Extract data from database for training predictive model.
"""

import psycopg2
import pandas as pd


if __name__ == "__main__":

    conn = psycopg2.connect(
        "dbname='ecobici' user='simonbedford' host='localhost'")
    cur = conn.cursor()

    sql = """SELECT trips.gender = 'M' AS gender_male,
    trips.age AS age,
    EXTRACT(DOW FROM trips.start_time) IN (1, 2, 3, 4, 5) AS dow,
    EXTRACT(HOUR FROM trips.start_time) AS start_hour,
    EXTRACT(MONTH FROM trips.start_time) AS month,
    trips.trip_duration AS duration,
    stations.hexagon_id AS hexagon_id
    FROM trips, stations
    WHERE trips.start_station_id = stations.id
    AND trips.start_station_id <= 452
    AND trips.trip_duration > 60
    AND trips.trip_duration <= 3600
    AND trips.age <= 65
    AND date(trips.start_time) >= '2014-01-01'
    AND date(trips.start_time) < '2016-08-01';"""
    cur.execute(sql)
    results = cur.fetchall()

    res = pd.DataFrame(results)
    res.columns = [d[0] for d in cur.description]

    res.to_csv('output/model_data.csv', index_label='id')