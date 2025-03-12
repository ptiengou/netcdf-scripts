import pandas as pd

class Airport():
    icao: 'str'
    iata: 'str'
    fullname: 'str'
    city: 'str'
    country: 'str'
    lon: float
    lat: float
    alt: float

def get_airports():
    import os
    df_airports = pd.read_csv(os.path.join(os.path.dirname(__file__), '../OPTIMISATION/data/airports.csv'),
            index_col=0,
            usecols=['id', 'ident', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft',
                     'iso_country', 'municipality', 'iata_code'])

    return df_airports

def get_airport(code, case=True, exact=False):

    if exact and not case:
        raise ValueError('You cannot have exact=True and case=False at the same time.')

    if exact:
        df_airport = df_airports[df_airports.eq(code).any(axis=1)]
    else:
        df_airport_icao = df_airports[df_airports['ident'].str.contains(code, na=False, case=case)]
        df_airport_iata = df_airports[df_airports['iata_code'].str.contains(code, na=False, case=case)]
        df_airport_name = df_airports[df_airports['name'].str.contains(code, na=False, case=case)]
        df_airport_city = df_airports[df_airports['municipality'].str.contains(code, na=False, case=case)]
        df_airport = pd.concat((df_airport_icao, df_airport_iata, df_airport_name, df_airport_city)).drop_duplicates()

    if len(df_airport) == 0:
        raise ValueError(f'No airport found for "{code}"; please be less specific.')
    elif len(df_airport) > 1:
        raise ValueError(f'Multiple airports found for "{code}"; please be more specific.\n{df_airport}')

    airport = Airport()
    airport.icao = df_airport['ident'].values[0]
    airport.iata = df_airport['iata_code'].values[0]
    airport.fullname = df_airport['name'].values[0]
    airport.city = df_airport['municipality'].values[0]
    airport.country = df_airport['iso_country'].values[0]
    airport.lon = df_airport['longitude_deg'].values[0]
    airport.lat = df_airport['latitude_deg'].values[0]
    airport.alt = df_airport['elevation_ft'].values[0]

    return airport

df_airports = get_airports()
