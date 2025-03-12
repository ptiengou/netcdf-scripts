import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from pycontrails import Flight
from pycontrails.physics import units

warnings.filterwarnings(message=".*Rate of climb/descent values greater than nominal.*", action="ignore")

def get_aircraft_data(yr, mth):

    if not isinstance(yr, str): yr = str(yr)
    if not isinstance(mth, str): mth = str(mth).zfill(2)
    last_day = '30' if mth in ('06', '09') else '31'

    df_aircraft_all = pd.read_csv(f'/bdd/Eurocontrol/{yr}/{yr}{mth}/Flights_{yr}{mth}01_{yr}{mth}{last_day}.csv.gz', compression='gzip')
    df_aircraft_all.set_index('ECTRL ID', inplace=True)

    return df_aircraft_all


def get_flightpath_data(yr, mth, filed=True):

    if not isinstance(yr, str): yr = str(yr)
    if not isinstance(mth, str): mth = str(mth).zfill(2)
    last_day = '30' if mth in ('06', '09') else '31'

    mode = 'Filed' if filed else 'Actual'

    df_flightpath_all = pd.read_csv(f'/bdd/Eurocontrol/{yr}/{yr}{mth}/Flight_Points_{mode}_{yr}{mth}01_{yr}{mth}{last_day}.csv.gz', compression='gzip')
    df_flightpath_all.drop_duplicates(['ECTRL ID', 'Sequence Number'], inplace=True)
    df_flightpath_all.set_index('ECTRL ID', inplace=True)
    indexes = np.unique(df_flightpath_all.index)

    return df_flightpath_all, indexes


def get_monthly_dataset(yr, mth, filed=True):

    df_flightpath_all, indexes = get_flightpath_data(yr, mth, filed)
    df_aircraft_all = get_aircraft_data(yr, mth)

    return df_flightpath_all, df_aircraft_all, indexes


def get_individual_flight(df_flightpath_all, df_aircraft_all, ID):

    df_flightpath = df_flightpath_all.loc[ID]

    sequence  = df_flightpath['Sequence Number'].values
    time      = convert_timestamp(df_flightpath['Time Over'].values, is_str=False)
    altitude  = units.ft_to_m(df_flightpath['Flight Level'].values*100.)
    latitude  = df_flightpath['Latitude'].values
    longitude = df_flightpath['Longitude'].values
    
    aircraft_prop = df_aircraft_all.loc[ID]
    aircraft_type = aircraft_prop['AC Type']
    ADEP = aircraft_prop['ADEP']
    ADES = aircraft_prop['ADES']
    attrs = dict(flight_id=ID, aircraft_type=aircraft_type,
                 ADEP=ADEP, ADES=ADES)
    
    flight = Flight(longitude=longitude, latitude=latitude,
                    altitude=altitude, time=time,
                    attrs=attrs, drop_duplicated_times=True)
    flight = flight.resample_and_fill()
    flight['level'] = units.m_to_pl(flight['altitude'])

    return flight


def get_flights_from_route(ADEP, ADES, df_flightpath_all, df_aircraft_all, indexes):

    mask = (df_aircraft_all['ADEP'] == ADEP) & (df_aircraft_all['ADES'] == ADES)
    df_flightpath_route = df_flightpath_all.loc[mask]
    df_aircraft_route = df_aircraft_all.loc[mask]
    indexes_route = np.unique(df_flightpath_route.index)

    return df_flightpath_route, df_aircraft_route, indexes_route


def convert_timestamp(time_str_series, is_str=None):

    if is_str is None: is_str = isinstance(time_str_series, str)
    
    if is_str:
        time_dt_series = np.datetime64(datetime.strptime(time_str_series, '%d-%m-%Y %H:%M:%S'), 's')
    else:
        time_dt_series = np.empty_like(time_str_series, dtype='datetime64[s]')
        for i, time_str in enumerate(time_str_series):
            time_dt_series[i] = convert_timestamp(time_str, is_str=True)

    return time_dt_series
