import os
import sys
import urllib.request

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016]

months = [ str(x).zfill(2) for x in range(1, 13)]

for y in years:
    for m in months:
        url = "https://www.ecobici.cdmx.gob.mx/sites/default/files/data/usages/{}-{}.csv".format(y, m)
        file_name = os.path.join("data", "{}-{}.csv".format(y, m))

        if not os.path.exists(file_name):
            print("Retrieving {}".format(file_name))
            try:
                urllib.request.urlretrieve(url, file_name)
            except urllib.error.URLError as e:
                print(e.code, file_name)
            except urllib.error.HTTPError as e:
                print(e.args, file_name)