import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.geometry import Point
import rtree

def clean_events_data(df):
    print("Cleaning events data...")

    # delete index with 24
    df = df.drop(df.index[23359])
    df = df.drop(df.index[11088])

    # convert start and end time to dates
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M')
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M')

    # add columns
    df['date'] = pd.to_datetime(df.date)
    df['is_event'] = 1
    df['event_duration'] = ((df.end_time-df.start_time).dt.total_seconds())/60
    df['DOW'] = df.date.dt.weekday_name
    df['TOD'] = df.start_time.dt.hour

    # aggregate data for only 2017
    df = df[df.date.dt.year == 2017]

    return df

def clean_events_location_data(df):
    print("Cleaning events location data...")
    # add taxi zones column
    df['taxi_zone'] = assign_taxi_zones(df, 'long', 'lat', 'taxi_zones')
    df['taxi_zone'] = df.taxi_zone.fillna(0).astype(int)

    return df

def clean_taxi_data(df):
    print("Cleaning taxi data...")

    # add columns
    df['lpep_pickup_datetime'] = pd.to_datetime(df.lpep_pickup_datetime)
    df['lpep_dropoff_datetime'] = pd.to_datetime(df.lpep_dropoff_datetime)
    df['date'] = df.lpep_pickup_datetime.dt.date
    df['taxi_duration'] = ((df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds())//60
    df['DOW'] = df.lpep_pickup_datetime.dt.weekday_name
    df['TOD']= df.lpep_pickup_datetime.dt.hour

    # Reformat taxi_zone column
    df = df.rename(columns={'PULocationID' : 'taxi_zone'})
    df['taxi_zone'] = df.taxi_zone.fillna(0).astype(int)

    return df

def assign_taxi_zones(df, lon_var, lat_var, locid_var):
    """Joins DataFrame with Taxi Zones shapefile.
    This function takes longitude values provided by `lon_var`, and latitude
    values provided by `lat_var` in DataFrame `df`, and performs a spatial join
    with the NYC taxi_zones shapefile. 
    The shapefile is hard coded in, as this function makes a hard assumption of
    latitude and longitude coordinates. It also assumes latitude=0 and 
    longitude=0 is not a datapoint that can exist in your dataset. Which is 
    reasonable for a dataset of New York, but bad for a global dataset.
    Only rows where `df.lon_var`, `df.lat_var` are reasonably near New York,
    and `df.locid_var` is set to np.nan are updated. 
    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        DataFrame containing latitudes, longitudes, and location_id columns.
    lon_var : string
        Name of column in `df` containing longitude values. Invalid values 
        should be np.nan.
    lat_var : string
        Name of column in `df` containing latitude values. Invalid values 
        should be np.nan
    locid_var : string
        Name of series to return. 
    """

    import geopandas
    from shapely.geometry import Point


    # make a copy since we will modify lats and lons
    localdf = df[[lon_var, lat_var]].copy()
    
    # missing lat lon info is indicated by nan. Fill with zero
    # which is outside New York shapefile. 
    localdf[lon_var] = localdf[lon_var].fillna(value=0.)
    localdf[lat_var] = localdf[lat_var].fillna(value=0.)
    

    shape_df = geopandas.read_file('data/taxizones/taxi_zones.shp')
    shape_df.drop(['OBJECTID', "Shape_Area", "Shape_Leng", "borough", "zone"],
                  axis=1, inplace=True)
    shape_df = shape_df.to_crs({'init': 'epsg:4326'})

    try:
        local_gdf = geopandas.GeoDataFrame(
            localdf, crs={'init': 'epsg:4326'},
            geometry=[Point(xy) for xy in
                      zip(localdf[lon_var], localdf[lat_var])])

        local_gdf = geopandas.sjoin(
            local_gdf, shape_df, how='left', op='within')
#         return local_gdf
        return local_gdf.LocationID.rename(locid_var)
    except ValueError as ve:
        print(ve)
        print(ve.stacktrace())
        series = localdf[lon_var]
        series = np.nan
    return series