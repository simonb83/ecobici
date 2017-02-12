import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
import psycopg2
import pandas as pd
import matplotlib.ticker as tkr
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

plt.style.use("ggplot")
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

conn = psycopg2.connect(
    "dbname='ecobici' user='simonbedford' host='localhost'")
cur = conn.cursor()

# Number of trips per year
sql = """SELECT COUNT(*) as count, Extract(year FROM start_time) as year FROM trips GROUP BY year;"""
cur.execute(sql)
results = cur.fetchall()
results = pd.DataFrame(results)
results.columns = ['num_trips', 'year']

fig, ax = plt.subplots(figsize=(12, 7))
x_ticks = [i + 1 for i in range(7)]
ax.bar(x_ticks, results['num_trips'])
ax.set_xlim(0.6, 8.2)
ax.set_xticks([i + 0.4 for i in x_ticks])
ax.set_xticklabels(["{}".format(int(y)) for y in results['year']])
for x, y in zip(x_ticks, results['num_trips']):
    ax.text(x + 0.4, y, "{:.1f} M".format(y / 1000000),
            ha='center', va='bottom', fontsize=15)
ax.set_title("Total Number of Trips per Year", fontsize=18)
ax.get_yaxis().set_major_formatter(
    tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.savefig("images/ecobici_number_of_trips_per_year.png",
            dpi=200, bbox_inches='tight')


# Number of bikes in circulation per year
sql = """SELECT COUNT(DISTINCT(bike_id)) as count, Extract(year FROM start_time) as year FROM trips GROUP BY year;"""
cur.execute(sql)
results = cur.fetchall()
results = pd.DataFrame(results)
results.columns = ['Unique_bikes', 'year']

fig, ax = plt.subplots(figsize=(12, 7))
x_ticks = [i + 1 for i in range(7)]
ax.bar(x_ticks, results['Unique_bikes'])
ax.set_xlim(0.6, 8.2)
ax.set_xticks([i + 0.4 for i in x_ticks])
ax.set_xticklabels(["{}".format(int(y)) for y in results['year']])
for x, y in zip(x_ticks, results['Unique_bikes']):
    ax.text(x + 0.4, y, "{:,}".format(y),
            ha='center', va='bottom', fontsize=15)
ax.set_title("Number of Bikes in Circulation by Year", fontsize=18)
ax.get_yaxis().set_major_formatter(
    tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.savefig("images/ecobici_number_of_bikes_per_year.png",
            dpi=200, bbox_inches='tight')


# 28-day rolling window; trips per month 2013 - 2016
sql = """SELECT COUNT(*) as count, "date"(start_time) as day FROM trips GROUP BY day;"""
cur.execute(sql)
results = cur.fetchall()
results = pd.DataFrame(results)
results.columns = ['count', 'date']
results = results.sort_values('date')
rolling = results.rolling(center=False, window=28).sum()

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(rolling["date"], rolling["count"], lw=3, color='#0570b0')
ax.set_xlim('2012-11-01', '2017-01-15')
ax.set_ylim(0, 900000)
ax.set_title("Monthly Bike Trips 2013-2016", fontsize=18)
ax.get_yaxis().set_major_formatter(
    tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_xticks(['2013-01-01', '2013-07-01', '2014-01-01', '2014-07-01',
               '2015-01-01', '2015-07-01', '2016-01-01', '2016-07-01', '2017-01-01'])
ax.set_xticklabels(["Dec-12", "Jun-13", "Dec-13", "Jun-14",
                    "Dec-15", "Jun-15", "Dec-16", "Jun-16", "Dec-17"])
ax.set_ylabel("Number of Trips, Trailing 28 Days", fontsize=16)
plt.savefig("images/ecobici_monthly_bike_trips.png",
            dpi=200, bbox_inches='tight')


# Average usage by hour of day
sql = """SELECT * FROM hourly_aggregate;"""
cur.execute(sql)
results = cur.fetchall()
per_hour = pd.DataFrame(results)
per_hour.columns = [desc[0] for desc in cur.description]
per_hour['avg'] = per_hour['total_trips'] / per_hour['number_of_days']
per_hour = per_hour.sort_values('hour')
week_day = per_hour[per_hour['weekday'] == True].groupby('hour')['avg'].sum()
w_end = per_hour[per_hour['weekday'] == False].groupby('hour')['avg'].sum()

fig, ax = plt.subplots(2, 1, figsize=(11, 9))
plt.rc('xtick.major', size=5)
x_ticks = [i + 1 for i in range(len(week_day))]
x_labels = ['12 AM', '6 AM', '9 AM', '12 PM', '15 PM', '18 PM', '21 PM']
ax[0].bar(x_ticks, week_day)
ax[1].bar(x_ticks, w_end)
ax[0].set_xlim(0, 22)
ax[1].set_xlim(0, 22)
ax[0].set_ylim(0, 3500)
ax[1].set_ylim(0, 3500)
ax[0].set_xticks([i + 0.4 for i in [1, 3, 6, 9, 12, 15, 18]])
ax[1].set_xticks([i + 0.4 for i in [1, 3, 6, 9, 12, 15, 18]])
ax[0].get_xaxis().set_tick_params(top='off')
ax[1].get_xaxis().set_tick_params(top='off')
ax[0].set_xticklabels([])
ax[1].set_xticklabels(x_labels, fontsize=14)
for i in (0, 1):
    ax[i].get_yaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: format(int(x), ',')))

ax[0].set_title("Weekday", fontsize=16)
ax[1].set_title("Weekend", fontsize=16)
fig.text(0.03, 0.5, 'Avg Number Trips per Hour',
         va='center', rotation='vertical', fontsize=16)
fig.text(0.5, 0.98, 'Mexico City Ecobici Usage by Hour of Day',
         va='center', ha='center', fontsize=18)
fig.text(0.5, 0.95, 'Based on Rides between Jan 2015 - Jul 2016',
         va='center', ha='center', fontsize=14)
plt.savefig("images/ecobici_hourly_usage.png", dpi=200, bbox_inches='tight')


# Number of trips per year by gender
sql = """SELECT COUNT(*) as count, Extract(YEAR FROM start_time) as year, gender FROM trips GROUP BY year, gender;"""
cur.execute(sql)
results = cur.fetchall()
results = pd.DataFrame(results)
results.columns = ['trips', 'year', 'gender']
male = results[results['gender'] == 'M']['trips'].tolist()
female = results[results['gender'] == 'F']['trips'].tolist()
years = results[results['gender'] == 'M']['year']

fig, ax = plt.subplots(figsize=(12, 7))
x_ticks = [i + 1 for i in range(7)]
ax.plot(x_ticks, male, lw=4, color='#238b45')
ax.plot(x_ticks, female, lw=4, color='#ae017e')
ax.set_xlim(0.5, 8)
ax.set_ylim(0, 7.25e6)
ax.set_xticks(x_ticks)
ax.set_xticklabels(["{}".format(int(y)) for y in years])
ax.text(7.1, male[-1], "Male", fontsize=15, va='center')
ax.text(7.1, female[-1], "Female", fontsize=15, va='center')
ax.set_title("Total Number of Trips per Year by Gender", fontsize=18)
ax.get_yaxis().set_major_formatter(
    tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.savefig("images/ecobici_number_of_trips_per_year_by_gender.png",
            dpi=200, bbox_inches='tight')


# Ride distribution by gender and age
# Male riders
sql = """SELECT 
SUM(CASE WHEN age <= 20 THEN 1 ELSE 0 END) AS gp1,
SUM(CASE WHEN age > 20 AND age <= 25 THEN 1 ELSE 0 END) AS gp2,
SUM(CASE WHEN age > 25 AND age <= 30 THEN 1 ELSE 0 END) AS gp3,
SUM(CASE WHEN age > 30 AND age <= 35 THEN 1 ELSE 0 END) AS gp4,
SUM(CASE WHEN age > 35 AND age <= 40 THEN 1 ELSE 0 END) AS gp5,
SUM(CASE WHEN age > 40 AND age <= 45 THEN 1 ELSE 0 END) AS gp6,
SUM(CASE WHEN age > 45 AND age <= 50 THEN 1 ELSE 0 END) AS gp7,
SUM(CASE WHEN age > 50 AND age <= 55 THEN 1 ELSE 0 END) AS gp8,
SUM(CASE WHEN age > 55 AND age <= 60 THEN 1 ELSE 0 END) AS gp9,
SUM(CASE WHEN age > 60 AND age <= 85 THEN 1 ELSE 0 END) AS gp10
FROM trips
WHERE gender = 'M';"""
cur.execute(sql)
results_M = np.array(cur.fetchall())[0]

# Female riders
sql = """SELECT 
SUM(CASE WHEN age <= 20 THEN 1 ELSE 0 END) AS gp1,
SUM(CASE WHEN age > 20 AND age <= 25 THEN 1 ELSE 0 END) AS gp2,
SUM(CASE WHEN age > 25 AND age <= 30 THEN 1 ELSE 0 END) AS gp3,
SUM(CASE WHEN age > 30 AND age <= 35 THEN 1 ELSE 0 END) AS gp4,
SUM(CASE WHEN age > 35 AND age <= 40 THEN 1 ELSE 0 END) AS gp5,
SUM(CASE WHEN age > 40 AND age <= 45 THEN 1 ELSE 0 END) AS gp6,
SUM(CASE WHEN age > 45 AND age <= 50 THEN 1 ELSE 0 END) AS gp7,
SUM(CASE WHEN age > 50 AND age <= 55 THEN 1 ELSE 0 END) AS gp8,
SUM(CASE WHEN age > 55 AND age <= 60 THEN 1 ELSE 0 END) AS gp9,
SUM(CASE WHEN age > 60 AND age <= 85 THEN 1 ELSE 0 END) AS gp10
FROM trips
WHERE gender = 'F';"""
cur.execute(sql)
results_F = np.array(cur.fetchall())[0]

bin_labels = ("16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60",
              "61-85")

fig, ax = plt.subplots(figsize=(14, 6))
m = results_M / sum(results_M)
f = results_F / sum(results_F)
x_ticks = [2 * i for i in range(len(bin_labels))]
ax.bar(x_ticks, m, color='#31a354', label='m')
ax.bar([i + 0.8 for i in x_ticks], f, color='#dd1c77', label='f')
ax.set_xlim(-0.6, 20)
ax.set_ylim(0, 0.4)
ax.set_xticks([i + 0.8 for i in x_ticks])
ax.set_xticklabels(bin_labels)
y_ticks = [0.1, 0.2, 0.3, 0.4]
ax.set_yticks(y_ticks)
ax.set_yticklabels(["{:.0f}%".format(i * 100) for i in y_ticks])
ax.get_xaxis().set_tick_params(top='off')
for x, y in zip(x_ticks, m):
    ax.text(x + 0.4, y, "{:.0f}%".format(y * 100),
            ha='center', va='bottom', fontsize=14)
for x, y in zip(x_ticks, f):
    ax.text(x + 1.2, y, "{:.0f}%".format(y * 100),
            ha='center', va='bottom', fontsize=14)
ax.legend(fontsize=14)
ax.set_title("Ecobici Rider Age Distribution by Gender", fontsize=18)
plt.savefig("images/ecobici_age_distribution_by_gender.png",
            dpi=200, bbox_inches='tight')


# Female usage by hour of day
sql = """SELECT SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END)::float / COUNT(*),
EXTRACT(HOUR FROM start_time) AS hour
FROM trips
WHERE date(start_time) >= '2015-01-01'
AND date(start_time) < '2016-08-01'
AND EXTRACT(DOW FROM start_time) IN (1, 2, 3, 4, 5)
GROUP BY hour ORDER BY hour;"""
cur.execute(sql)
results = cur.fetchall()
week_day_female_pc = np.array(results)[:, 0]

sql = """SELECT SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END)::float / COUNT(*),
EXTRACT(HOUR FROM start_time) AS hour
FROM trips
WHERE date(start_time) >= '2015-01-01'
AND date(start_time) < '2016-08-01'
AND EXTRACT(DOW FROM start_time) IN (0, 6)
GROUP BY hour ORDER BY hour;"""
cur.execute(sql)
results = cur.fetchall()
wend_female_pc = np.array(results)[:, 0]

fig, ax = plt.subplots(2, 1, figsize=(11, 9))
plt.rc('xtick.major', size=5)
x_ticks = [i + 1 for i in range(len(week_day_female_pc))]
x_labels = ['12 AM', '6 AM', '9 AM', '12 PM', '15 PM', '18 PM', '21 PM']
ax[0].bar(x_ticks, week_day_female_pc, color='#dd1c77')
ax[1].bar(x_ticks, wend_female_pc, color='#dd1c77')

ax[0].set_xticklabels([])
ax[1].set_xticklabels(x_labels, fontsize=14)

yticks = [0.1, 0.2, 0.3, 0.4]
for i in (0, 1):
    ax[i].set_xlim(0, 22)
    ax[i].set_ylim(0, 0.4)
    ax[i].set_xticks([i + 0.4 for i in [1, 3, 6, 9, 12, 15, 18]])
    ax[i].get_xaxis().set_tick_params(top='off')
    ax[i].set_yticks(yticks)
    ax[i].set_yticklabels(["{:.0f}%".format(i * 100) for i in yticks])
    ax[i].axhline(0.26, 0, 0.97, lw=2, color='#252525')
    ax[i].text(20.5, 0.265, 'Avg - 26%', fontsize=14,
               color='#252525', style='italic', ha='center', va='bottom')

ax[0].set_title("Weekday", fontsize=16)
ax[1].set_title("Weekend", fontsize=16)
fig.text(0.04, 0.5, 'Female as % of Avg Hourly Trips',
         va='center', rotation='vertical', fontsize=16)
fig.text(0.5, 0.98, 'Mexico City Ecobici Female Ridership % by Hour of Day',
         va='center', ha='center', fontsize=18)
fig.text(0.5, 0.95, 'Based on Rides between Jan 2015 - Jul 2016',
         va='center', ha='center', fontsize=14)
plt.savefig("images/ecobici_hourly_usage_females.png",
            dpi=200, bbox_inches='tight')

# Female usage by area of city
sql = """SELECT SUM(CASE WHEN trips.gender = 'F' THEN 1.0 ELSE 0 END) / COUNT(*) * 100 AS female_pc,
stations.hexagon_id AS hexagon_id
FROM trips, stations
WHERE date(trips.start_time) >= '2014-01-01' AND trips.start_station_id <= 452
AND trips.start_station_id = stations.id
GROUP BY hexagon_id ORDER BY hexagon_id;"""
cur.execute(sql)
gender = cur.fetchall()
gender = pd.DataFrame(gender)
gender.columns = [d[0] for d in cur.description]


def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def female_color(x):
    if x >= 30:
        return '#980043'
    elif x < 30 and x > 26.5:
        return '#e7298a'
    elif x < 25.5:
        return '#43a2ca'
    else:
        return '#FFCC00'

fig, ax = plt.subplots(figsize=(10, 12))

m = Basemap(
    resolution='c',
    projection='merc',
    lat_0=19.4, lon_0=-99.17,
    llcrnrlon=-99.22, llcrnrlat=19.35, urcrnrlon=-99.12, urcrnrlat=19.45)

m.fillcontinents(color='#252525', lake_color='#46bcec')
m.drawmapboundary(fill_color='#252525')
m.readshapefile('../data/Colonias/df/df', 'colonias', drawbounds=False)
m.readshapefile('../data/Streets/cdmx/cdmx', 'streets', color='grey')

colonia_names = [c['SETT_NAME'] for c in m.colonias_info]
ring_num = [c['RINGNUM'] for c in m.colonias_info]
for r, n, shape in zip(ring_num, colonia_names, m.colonias):
    if r < 2:
        x, y = centeroidnp(np.array(shape))
        ax.text(x, y, n, ha='center', va='center', fontsize=12,
                color='#969696', weight='bold', zorder=20)

m.readshapefile('hexagons/hexagons', 'hexagons', drawbounds=True)
df_poly = pd.DataFrame({
    'shapes': [Polygon(np.array(shape), True) for shape in m.hexagons],
    'hexagon_id': [c['HEXAGON_ID'] for c in m.hexagons_info]})

df_poly = df_poly.merge(gender, on='hexagon_id', how='left')
df_poly = df_poly.fillna(0)
for x, shape in zip(df_poly['female_pc'], m.hexagons):
    patch = Polygon(np.array(shape), True, facecolor=female_color(
        x), edgecolor=None, lw=0, zorder=2, alpha=0.7)
    ax.add_patch(patch)

# Custom legend
leg_colors = ('#43a2ca', '#FFCC00', '#e7298a')
leg_labels = ('Below avg.', 'Average', 'Above avg.')
leg_patches = []
for t, c in enumerate(leg_colors):
    leg_patches.append(ax.add_patch(patches.Rectangle((t, 0.5), 5, 0.1,
                                                      facecolor=c, label=c, lw=0)))
legend = plt.legend(leg_patches, leg_labels, loc=4)
legend.get_frame().set_facecolor('#d9d9d9')
plt.savefig('images/ecobici_female_usage_by_hexagon.png',
            dpi=150, bbox_inches='tight')

# Speed estimates
sql = """SELECT * FROM avg_speed WHERE age >= 16 AND age <= 65;"""
cur.execute(sql)
speeds = cur.fetchall()
speeds = pd.DataFrame(speeds)
speeds.columns = [d[0] for d in cur.description]
speeds['kmh'] = speeds['speed'].apply(lambda x: x * 60 * 60 / 1000)
speeds = speeds.sort_values(['weekday', 'gender', 'age'])
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
plt.rc('xtick.major', size=5)
x1 = speeds[(speeds['weekday'] == True) & (speeds['gender'] == 'F')]['age']
y1 = speeds[(speeds['weekday'] == True) & (speeds['gender'] == 'F')]['kmh']
x2 = speeds[(speeds['weekday'] == True) & (speeds['gender'] == 'M')]['age']
y2 = speeds[(speeds['weekday'] == True) & (speeds['gender'] == 'M')]['kmh']

x3 = speeds[(speeds['weekday'] == False) & (speeds['gender'] == 'F')]['age']
y3 = speeds[(speeds['weekday'] == False) & (speeds['gender'] == 'F')]['kmh']
x4 = speeds[(speeds['weekday'] == False) & (speeds['gender'] == 'M')]['age']
y4 = speeds[(speeds['weekday'] == False) & (speeds['gender'] == 'M')]['kmh']

ax[0].plot(x2, y2, lw=3, color='#238b45', label='Male')
ax[0].plot(x1, y1, lw=3, color='#ae017e', label='Female')
ax[1].plot(x3, y3, lw=3, color='#ae017e')
ax[1].plot(x4, y4, lw=3, color='#238b45')

for i in (0, 1):
    ax[i].set_xlim(12, 69)
    ax[i].set_ylim(0, 15)
    ax[i].get_xaxis().set_tick_params(top='off')
    ax[i].set_xticks([20, 30, 40, 50, 60])
    ax[i].set_yticks([3, 6, 9, 12])
    ax[i].set_yticklabels([3, 6, 9, 12], fontsize=14)

ax[0].set_xticklabels([])
ax[1].set_xticklabels([20, 30, 40, 50, 60], fontsize=14)
ax[0].set_title("Weekday", fontsize=16)
ax[1].set_title("Weekend", fontsize=16)
ax[1].set_xlabel("Age (years)", fontsize=16)

fig.text(0.07, 0.5, 'Avg Speed (km / h)', va='center',
         rotation='vertical', fontsize=16)
fig.text(0.5, 0.98, 'Mexico City Ecobici Avg Speed by Age and Gender',
         va='center', ha='center', fontsize=18)
fig.text(0.5, 0.95, 'Based on Rides between Jan 2015 - Jul 2016',
         va='center', ha='center', fontsize=14)
ax[0].legend(fontsize=14)
plt.savefig("images/ecobici_avg_speed.png", dpi=200, bbox_inches='tight')

# Magical Transports % by month
sql = """SELECT
    month,
    SUM(transported_to_other_station) / SUM(total_drop_offs) frac
  FROM monthly_station_aggregates
  WHERE month < '2016-08-01'
  GROUP BY month
  ORDER BY month;"""
cur.execute(sql)
results = np.array(cur.fetchall())
x_ticks = []
x_labels = []
for year in (2010, 2011, 2012, 2013, 2014, 2015, 2016):
    x_ticks.append("{}-01".format(year))
    x_labels.append("Jan '{}".format(str(year)[2:]))
    x_ticks.append("{}-07".format(year))
    x_labels.append("Jul '{}".format(str(year)[2:]))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results[:, 0], results[:, 1], lw=3, color='#0570b0')
ax.set_ylim(0, 0.3)
ax.set_yticks([0.1, 0.2, 0.3])
ax.set_yticklabels(['10%', '20%', '30%'])
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, fontsize=14)
fig.text(0.5, 0.99, 'Magical Transports by Month',
         va='center', ha='center', fontsize=18)
fig.text(0.5, 0.935, 'Based on Rides between Jan 2010 - Jul 2016',
         va='center', ha='center', fontsize=14)
ax.set_ylabel("% of rides", fontsize=18)
plt.savefig("images/ecobici_magical_transports_by_month.png",
            dpi=200, bbox_inches='tight')

# Magical Transports by City Region
sql = """SELECT
    SUM(transported_to_other_station) / SUM(total_drop_offs) frac,
    stations.hexagon_id AS hexagon_id
  FROM monthly_station_aggregates, stations
  WHERE month < '2016-08-01' AND month >= '2015-01-01'
  AND end_station_id = stations.id
  GROUP BY hexagon_id;"""
cur.execute(sql)
results = pd.DataFrame(cur.fetchall())
results.columns = [d[0] for d in cur.description]
results['frac'] = results['frac'].apply(lambda x: float(x))
results['frac_cat'] = pd.cut(results['frac'], 6, labels=[i for i in range(6)])


def magical_transport_color(x):
    colors = ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
    return colors[int(x)]

fig, ax = plt.subplots(figsize=(10, 12))
m = Basemap(
    resolution='c',
    projection='merc',
    lat_0=19.4, lon_0=-99.17,
    llcrnrlon=-99.22, llcrnrlat=19.35, urcrnrlon=-99.12, urcrnrlat=19.45)

m.fillcontinents(color='#252525', lake_color='#46bcec')
m.drawmapboundary(fill_color='#252525')
m.readshapefile('../data/Colonias/df/df', 'colonias', drawbounds=False)
m.readshapefile('../data/Streets/cdmx/cdmx', 'streets', color='grey')
colonia_names = [c['SETT_NAME'] for c in m.colonias_info]
ring_num = [c['RINGNUM'] for c in m.colonias_info]
for r, n, shape in zip(ring_num, colonia_names, m.colonias):
    if r < 2:
        x, y = centeroidnp(np.array(shape))
        ax.text(x, y, n, ha='center', va='center', fontsize=12,
                color='#969696', weight='bold', zorder=20)
m.readshapefile('hexagons/hexagons', 'hexagons', drawbounds=True)
df_poly = pd.DataFrame({
    'shapes': [Polygon(np.array(shape), True) for shape in m.hexagons],
    'hexagon_id': [c['HEXAGON_ID'] for c in m.hexagons_info]
})
df_poly = df_poly.merge(results, on='hexagon_id', how='left')
df_poly = df_poly.fillna(0)

for x, shape in zip(df_poly['frac_cat'], m.hexagons):
    patch = Polygon(np.array(shape), True, facecolor=magical_transport_color(
        x), edgecolor=None, lw=0, zorder=2, alpha=0.7)
    ax.add_patch(patch)
fig.text(0.5, 0.87, 'Magical Transports - % of Rides by Region',
         va='center', ha='center', fontsize=18)
ax3 = fig.add_axes([0.695, 0.21, 0.18, 0.03])

cmap = mpl.colors.ListedColormap(
    ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000'])
bounds = [2, 12, 22, 32, 42, 52, 62]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                norm=norm,
                                boundaries=bounds,
                                ticks=bounds,
                                spacing='uniform',
                                orientation='horizontal')
cb3.set_ticks([2, 22, 42, 62])
cb3.set_ticklabels(['2%', '22%', '42%', '62%'])
cb3.ax.tick_params(labelcolor='w', labelsize=14)
plt.savefig('images/ecobici_magical_transport_by_hexagon.png',
            dpi=150, bbox_inches='tight')

# Magical Transports by Hour of Day
center = (360, 362, 364, 366, 368, 359, 361, 363,
          365, 367, 324, 326, 328, 330, 332, 334)
polanco = (379, 381, 383, 385, 346, 348, 350, 352, 354,
           345, 347, 349, 351, 353, 314, 316, 318, 320)
south = (147, 149, 151, 153, 155, 157, 116, 118,
         120, 122, 117, 119, 121, 123, 86, 88)
sql = """SELECT
    SUM((end_station_id != next_start_station_id)::int) / COUNT(*)::numeric frac,
    EXTRACT (HOUR FROM end_time) AS drop_off_hour
  FROM bike_station_ids
  WHERE date(end_time) < '2016-08-01' AND date(end_time) >= '2015-01-01'
  AND end_station_id IN (SELECT id FROM stations WHERE hexagon_id IN {})
  GROUP BY drop_off_hour ORDER BY drop_off_hour;""".format(center)
cur.execute(sql)
results_center = pd.DataFrame(cur.fetchall())
results_center.columns = [d[0] for d in cur.description]
sql = """SELECT
    SUM((end_station_id != next_start_station_id)::int) / COUNT(*)::numeric frac,
    EXTRACT (HOUR FROM end_time) AS drop_off_hour
  FROM bike_station_ids
  WHERE date(end_time) < '2016-08-01' AND date(end_time) >= '2015-01-01'
  AND end_station_id IN (SELECT id FROM stations WHERE hexagon_id IN {})
  GROUP BY drop_off_hour ORDER BY drop_off_hour;""".format(south)
cur.execute(sql)
results_south = pd.DataFrame(cur.fetchall())
results_south.columns = [d[0] for d in cur.description]

fig, ax = plt.subplots(2, 1, figsize=(11, 9))
plt.rc('xtick.major', size=5)
x_ticks = [0, 6, 12, 18]
x_labels = ['12 AM', '6AM', '12 PM', '6 PM']
ax[0].bar(results_center['drop_off_hour'], results_center['frac'])
ax[1].bar(results_south['drop_off_hour'], results_south['frac'])
for i in (0, 1):
    ax[i].set_xlim(-0.2, 22)
    ax[i].set_ylim(0, 0.4)
    ax[i].set_xticks(x_ticks)
    ax[i].set_yticks([0.1, 0.2, 0.3, 0.4])
    ax[i].set_yticklabels(['10%', '20%', '30%', '40%'])
    ax[i].get_xaxis().set_tick_params(top='off')

ax[0].set_xticklabels([])
ax[1].set_xticklabels(x_labels, fontsize=14)
ax[0].set_title("Centro Historico", fontsize=16)
ax[1].set_title("South", fontsize=16)
fig.text(0.5, 0.98, 'Mexico City Ecobici Magical Transports by Hour of Day',
         va='center', ha='center', fontsize=18)
fig.text(0.5, 0.95, 'Based on Rides between Jan 2015 - Jul 2016',
         va='center', ha='center', fontsize=14)
fig.text(0.04, 0.5, '% Magical Transports',
         va='center', rotation='vertical', fontsize=16)
plt.savefig("images/ecobici_magical_transport_by_hour.png",
            dpi=200, bbox_inches='tight')

# Magical Transports Distances
sql = """SELECT
    ST_Distance(ST_Transform(s1.geom, 3857), ST_Transform(s2.geom, 3857))
    FROM bike_station_ids, stations s1, stations s2
    WHERE bike_station_ids.end_station_id = s1.id
    AND bike_station_ids.next_start_station_id = s2.id
    AND bike_station_ids.end_station_id != bike_station_ids.next_start_station_id
    AND date(bike_station_ids.start_time) >= '2015-01-01'
    AND date(bike_station_ids.start_time) < '2016-08-01'
    AND s1.hexagon_id IN {};""".format(south)
cur.execute(sql)
results_south = np.array(cur.fetchall())
sql = """SELECT
    ST_Distance(ST_Transform(s1.geom, 3857), ST_Transform(s2.geom, 3857))
    FROM bike_station_ids, stations s1, stations s2
    WHERE bike_station_ids.end_station_id = s1.id
    AND bike_station_ids.next_start_station_id = s2.id
    AND bike_station_ids.end_station_id != bike_station_ids.next_start_station_id
    AND date(bike_station_ids.start_time) >= '2015-01-01'
    AND date(bike_station_ids.start_time) < '2016-08-01'
    AND s1.hexagon_id IN {};""".format(center)
cur.execute(sql)
results_center = np.array(cur.fetchall())

mins = [np.min(results_south), np.min(results_center)]
maxes = [np.max(results_south), np.max(results_center)]
medians = [np.median(results_south), np.median(results_center)]
perc_25 = [np.percentile(results_south, 25), np.percentile(results_center, 25)]
perc_75 = [np.percentile(results_south, 75), np.percentile(results_center, 75)]

fig, ax = plt.subplots(figsize=(10, 7))
y_vals = [1, 3]
ax.plot([mins[0], maxes[0]], [1.5, 1.5], lw=3, color='#f03b20')
ax.plot([mins[1], maxes[1]], [3.5, 3.5], lw=3, color='#1c9099')
ax.barh(y_vals[0], [m - p for m, p in zip(medians, perc_25)][0],
        left=perc_25[0], zorder=10, linewidth=0, color='#f03b20', height=1)
ax.barh(y_vals[1], [m - p for m, p in zip(medians, perc_25)][1],
        left=perc_25[1], zorder=10, linewidth=0, color='#1c9099', height=1)
ax.barh(y_vals[0], [p - m for m, p in zip(medians, perc_75)][0],
        left=medians[0], zorder=10, linewidth=0, color='#f03b20', height=1)
ax.barh(y_vals[1], [p - m for m, p in zip(medians, perc_75)][1],
        left=medians[1], zorder=10, linewidth=0, color='#1c9099', height=1)
ax.set_ylim(0.3, 4.7)
ax.set_xlim(-400, max(maxes) + 1000)
ax.scatter(mins[0], [y + 0.5 for y in y_vals][0],
           marker='|', s=500, lw=3, color='#f03b20')
ax.scatter(mins[1], [y + 0.5 for y in y_vals][1],
           marker='|', s=500, lw=3, color='#1c9099')
ax.scatter(maxes[0], [y + 0.5 for y in y_vals][0],
           marker='|', s=500, lw=3, color='#f03b20')
ax.scatter(maxes[1], [y + 0.5 for y in y_vals][1],
           marker='|', s=500, lw=3, color='#1c9099')
ax.plot([medians[0], medians[0]], [1.0, 2.0], color='w', zorder=20, lw=3)
ax.plot([medians[1], medians[1]], [3.0, 4.0], color='w', zorder=20, lw=3)
ax.set_yticks([1.5, 3.5])
ax.set_yticklabels(['South', 'Centro Historico'])
ax.get_xaxis().set_major_formatter(
    tkr.FuncFormatter(lambda x, p: format(int(x / 1000), ',')))
ax.text(12, 4.1, "Min", ha='center', va='bottom',
        fontsize=13, style='italic', color='#636363')
ax.text(maxes[1], 4.1, "Max", ha='center', va='bottom',
        fontsize=13, style='italic', color='#636363')
ax.text(medians[1], 4.1, "Median", ha='center', va='bottom',
        fontsize=13, style='italic', color='#636363')
ax.text(perc_25[1], 2.9, "25th\nPercentile", ha='center',
        va='top', fontsize=13, style='italic', color='#636363')
ax.text(perc_75[1], 2.9, "75th\nPercentile", ha='center',
        va='top', fontsize=13, style='italic', color='#636363')
ax.set_xlabel("Distance Bike Transported (km)", fontsize=16)
fig.text(0.5, 0.965, 'Distance Bikes are Transported',
         va='center', ha='center', fontsize=18)
fig.text(0.5, 0.93, 'Based on Rides between Jan 2015 - Jul 2016',
         va='center', ha='center', fontsize=14)
plt.savefig('images/ecobici_magical_transport_distances.png',
            dpi=150, bbox_inches='tight')

# Uniquely identifiable trips
sql = """SELECT
    gender,
    age,
    SUM(CASE WHEN count = 1 THEN 1.0 ELSE 0 END) / SUM(count) AS uniq_frac,
    SUM(count) AS total
  FROM anonymous_analysis_hourly
  WHERE age BETWEEN 16 AND 70
  GROUP BY gender, age
  ORDER BY gender, age;"""
cur.execute(sql)
results = pd.DataFrame(cur.fetchall())
results.columns = [d[0] for d in cur.description]
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(results[results['gender'] == 'F']['age'], results[
        results['gender'] == 'F']['uniq_frac'], lw=4, color='#ae017e', label="Female")
ax.plot(results[results['gender'] == 'M']['age'], results[
        results['gender'] == 'M']['uniq_frac'], lw=4, color='#238b45', label="Male")
ax.set_ylim(0.6, 1.01)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1])
ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'])
ax.legend(loc=2)
ax.set_xlabel("Rider Age", fontsize=16)
ax.set_title("Percentage of Uniquely Identifiable Trips", fontsize=18)
plt.savefig("images/ecobici_unique_trips.png", dpi=200, bbox_inches='tight')

# Trip Duration by Gender and Age
# Male riders query
sql = """SELECT age, trip_duration FROM trips
    WHERE gender = 'M'
    AND trips.start_station_id <= 452
    AND date(trips.start_time) >= '2014-01-01'
    AND date(trips.start_time) < '2016-08-01'
    AND trips.age <= 65
    AND trips.trip_duration > 60
    AND trips.trip_duration <= 3600
ORDER BY random()
LIMIT 10000;"""
cur.execute(sql)
results_m = cur.fetchall()
results_m = pd.DataFrame(results_m)
results_m.columns = [d[0] for d in cur.description]

# Feale riders query
sql = """SELECT age, trip_duration FROM trips
    WHERE gender = 'F'
    AND trips.start_station_id <= 452
    AND date(trips.start_time) >= '2014-01-01'
    AND date(trips.start_time) < '2016-08-01'
    AND trips.age <= 65
    AND trips.trip_duration > 60
    AND trips.trip_duration <= 3600
ORDER BY random()
LIMIT 10000;"""
cur.execute(sql)
results_f = cur.fetchall()
results_f = pd.DataFrame(results_m)
results_f.columns = [d[0] for d in cur.description]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].scatter(results_m['age'], results_m['trip_duration'], color='#238b45')
ax[1].scatter(results_f['age'], results_f['trip_duration'], color='#ae017e')
for i in (0, 1):
    ax[i].set_ylim(0, 4000)
ax[1].set_yticks([])
ax[0].get_yaxis().set_major_formatter(
    tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
fig.text(0.5, 0.99, 'Trip Duration vs. Age',
         va='center', ha='center', fontsize=24)
fig.text(0.5, 0.93, 'Random Sample of 10,000 rides between Jan 2015 - Jul 2016',
         va='center', ha='center', fontsize=20)
fig.text(0.5, 0.03, 'Rider Age in Years',
         va='center', ha='center', fontsize=22)
fig.text(0.05, 0.5, 'Trip Duration in Seconds',
         va='center', rotation='vertical', fontsize=22)
plt.savefig("images/ecobici_duration_v_age.png",
            dpi=200, bbox_inches='tight')


# Trip Duration -> Specific Sub Segment
sql = """SELECT trip_duration FROM trips, stations
    WHERE age = 25
    AND gender = 'F'
    AND trips.start_station_id <= 452
    AND date(trips.start_time) >= '2014-01-01'
    AND date(trips.start_time) < '2016-08-01'
    AND trips.trip_duration > 60
    AND trips.trip_duration <= 3600
    AND EXTRACT(DOW FROM trips.start_time) IN (1, 2, 3, 4, 5)
    AND EXTRACT(HOUR FROM trips.start_time) = 9
    AND trips.start_station_id = stations.id
    AND stations.hexagon_id = 379;"""
cur.execute(sql)
res = np.array(cur.fetchall()).astype(float)

bins = [60, 600, 1200, 1800, 2400, 3000, 3600]
labels = ['1-10', '10-20', '20-30', '30-40', '40-50', '50-60']
hist, bins = np.histogram(res, bins=bins)
hist = hist / len(res)
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax.bar(center, hist, align='center', width=width)
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%'])
ax.set_xlim(0, 3660)
ax.set_xticks(center)
ax.set_xticklabels(labels)
for x, y in zip(center, hist):
    ax.text(x, y, "{:.0f}%".format(y * 100),
            ha='center', va='bottom', fontsize=15)
fig.text(0.5, 0.04, 'Trip Duration (Minutes)',
         va='center', ha='center', fontsize=16)
plt.savefig("images/ecobici_subset_distribution.png",
            dpi=200, bbox_inches='tight')
