import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import psycopg2
import pandas as pd
import numpy as np
import psycopg2.extras
from datetime import datetime, timedelta


def styling(n):
    if n >= 500:
        return 6, '#F11810'
    elif n < 500 and n >= 250:
        return 3, '#2167AB'
    elif n < 250 and n >= 50:
        return 1, '#2167AB'
    elif n < 50:
        return 0.2, '#2167AB'


if __name__ == "__main__":

    conn = psycopg2.connect(
        "dbname='ecobici' user='simonbedford' host='localhost'")
    cur = conn.cursor()
    dict_cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Get distinct pairs of stations
    sql = """SELECT DISTINCT(start_station_id::text || '-' || end_station_id::text) FROM trip_legs_new;"""
    cur.execute(sql)
    existing_pairs = cur.fetchall()
    existing_pairs = tuple([i[0] for i in existing_pairs])

    # Get the stations in the area of interest
    sql = """SELECT DISTINCT(id) FROM stations
    WHERE hexagon_id >= 280 OR hexagon_id IN (248, 250, 252, 254, 256, 258);"""
    cur.execute(sql)
    station_ids = cur.fetchall()

    station_ids = tuple([i[0] for i in station_ids])

    # Get relevant trip_ids
    sql = """SELECT DISTINCT(id) FROM trips WHERE date(start_time) = '2016-04-06'
    AND EXTRACT(HOUR FROM start_time) >= 5
    AND start_station_id != end_station_id
    AND start_station_id IN {} AND end_station_id IN {}
    AND start_station_id::text || '-' || end_station_id::text IN {}""".format(station_ids, station_ids, existing_pairs)
    cur.execute(sql)
    trips = cur.fetchall()

    # Get the points for each trip
    points = []
    for t in trips:
        sql = """SELECT start_time, start_station_id, end_station_id FROM trips WHERE id = {};""".format(t[
                                                                                                         0])
        dict_cur.execute(sql)
        trip = dict_cur.fetchall()[0]
        start_time = trip['start_time']
        start_station_id = trip['start_station_id']
        end_station_id = trip['end_station_id']

        sql = """SELECT * FROM trip_legs_new WHERE start_station_id = {} AND end_station_id = {}
        ORDER BY leg_number""".format(start_station_id, end_station_id)
        dict_cur.execute(sql)
        legs = dict_cur.fetchall()

        d = {}
        d['longitude'] = legs[0]['start_longitude']
        d['latitude'] = legs[0]['start_latitude']
        d['time'] = start_time
        points.append(d)

        for l in legs:
            # print(l['leg_number'])
            d = {}
            d['longitude'] = l['end_longitude']
            d['latitude'] = l['end_latitude']
            start_time += timedelta(seconds=int(l['leg_time_taken']))
            d['time'] = start_time
            points.append(d)

    p = pd.DataFrame(points)
    p.to_csv('output/trip_points.csv', index=None)

    # Plot the map of most popular routes

    fig, ax = plt.subplots(figsize=(10,20))

    m = Basemap(
        resolution='c',
        #resolution = 'f',
        projection='merc',
        lat_0=19.4, lon_0=-99.17,
        llcrnrlon=-99.22, llcrnrlat=19.35, urcrnrlon=-99.12, urcrnrlat=19.45)

    m.fillcontinents(color='#252525', lake_color='#46bcec')
    m.drawmapboundary(fill_color='#252525')

    m.readshapefile('output/line_segments/line_segments', 'line_segments', )

    df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.line_segments],
        'num_legs': [c['NUM_LEGS'] for c in m.line_segments_info]
    })

    patches = []

    for n, shape in zip(df_poly['num_legs'], m.line_segments):
        lw, c = styling(n)
        patch = Polygon(np.array(shape), True, facecolor=None,
                        edgecolor=c, lw=lw, zorder=2)
        ax.add_patch(patch)

    plt.savefig('images/ecobici_popular_routes.png',
                dpi=200, bbox_inches='tight')
