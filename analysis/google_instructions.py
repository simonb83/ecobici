import urllib
import urllib.request
import polyline
import json
from datetime import timedelta
import pandas as pd
import psycopg2
import itertools
from builtins import zip as izip
import re
import time
import logging
import argparse


def each_cons(xs, n):
    return izip(*(xs[i:] for i in range(n)))


def get_coords(station_id, conn):
    sql = """SELECT latitude, longitude FROM stations WHERE id = {};""".format(
        station_id)
    cur.execute(sql)
    results = cur.fetchall()[0]
    lat, lon = str(results[0]), str(results[1])
    return lat, lon


def google_directions(start_lat, start_lon, end_lat, end_lon, GOOGLE_API_KEY):
    time.sleep(1)
    qry = {
        'origin': ','.join([start_lat, start_lon]),
        'destination': ','.join([end_lat, end_lon]),
        'mode': 'bicycling',
        'key': GOOGLE_API_KEY
    }
    url = "{}?{}".format(
        'https://maps.googleapis.com/maps/api/directions/json', urllib.parse.urlencode(qry))

    res = urllib.request.urlopen(url)
    res_body = res.read()
    j = json.loads(res_body.decode("utf-8"))
    return j


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--index", help="Start index for processing station ids")
    parser.add_argument(
        "-t", "--iterations", help="Total number of trips to process.")
    args = parser.parse_args()

    start_index = int(args.index)
    iterations = int(args.iterations)

    GOOGLE_API_KEY = 'AIzaSyAXqW_lD2GbLNfccz_q7K5PWy-9VoKohBo'
    logging.basicConfig(filename="google_directions_2.log", level=logging.INFO)

    conn = psycopg2.connect(
        "dbname='ecobici' user='simonbedford' host='localhost'")
    cur = conn.cursor()

    # Load the trip ids to process
    id_list = pd.read_csv("output/leg_station_ids.csv",
                          skiprows=range(1, start_index + 1), nrows=iterations)

    for idx, row in id_list.iterrows():
        # Get the trip information
        start_station_id = int(row['start_id'])
        end_station_id = int(row['end_id'])
        start_lat, start_lon = get_coords(start_station_id, conn)
        end_lat, end_lon = get_coords(end_station_id, conn)
        try:
            j = google_directions(start_lat, start_lon,
                                  end_lat, end_lon, GOOGLE_API_KEY)
        except urllib.error.HTTPError as e:
            logging.error(e)
            break
        markers = []
        steps = j['routes'][0]['legs'][0]['steps']
        leg_counter = 1
        for s in steps:
            # Total distance and time for the step
            distance = s['distance']['value']
            trip_time = s['duration']['value']
            # Convert polyline into points
            points = polyline.decode(s['polyline']['points'])
            num_points = len(points)
            for p1, p2 in each_cons(points, 2):
                d = {}
                d['start_station_id'] = start_station_id
                d['end_station_id'] = end_station_id
                d['leg_number'] = leg_counter
                d['start_latitude'] = p1[0]
                d['start_longitude'] = p1[1]
                d['end_latitude'] = p2[0]
                d['end_longitude'] = p2[1]
                d['leg_time_taken'] = trip_time / num_points
                d['leg_distance'] = distance / num_points
                markers.append(d)
                leg_counter += 1

        for m in markers:
            cur.execute('INSERT INTO trip_legs_new (start_station_id, end_station_id, leg_number, start_latitude, start_longitude, end_latitude, end_longitude, leg_time_taken, leg_distance) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (m[
                        'start_station_id'], m['end_station_id'], m['leg_number'], m['start_latitude'], m['start_longitude'], m['end_latitude'], m['end_longitude'], m['leg_time_taken'], m['leg_distance']))
        conn.commit()
        logging.info("Processed trip with index {}".format(row['index']))
