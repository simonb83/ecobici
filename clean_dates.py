import pandas as pd


def convert_date(d):
    return "2016-{}-{}".format(d[3:5], d[0:2])

file_names = ('data/2016-08.csv', 'data/2016-09.csv', 'data/2016-10.csv')

for f in file_names:
    df = pd.read_csv(f)

    df['Fecha_Retiro'] = df['Fecha_Retiro'].apply(convert_date)
    df['Fecha_Arribo'] = df['Fecha_Arribo'].apply(convert_date)

    df.to_csv(f, index=False)
