import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.geometry import Point
import rtree
import pickle
import random

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from data_clean import *

def run():
    print("Read data...")
    event_list_df = pd.read_csv("data/NYC_Parks_Events_Listing___Event_Listing.csv", parse_dates=True)
    loc_df = pd.read_csv('data/NYC_Parks_Events_Listing___Event_Locations.csv')
    
    #taxi_df = pd.read_csv("data/2017_Green_Taxi_Trip_Data.csv")

    filename = "data/2017_Green_Taxi_Trip_Data.csv"
    n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
    s = 1000000 #desired sample size
    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
    taxi_df = pd.read_csv(filename, skiprows=skip)

    event_list_df = clean_events_data(event_list_df)
    loc_df = clean_events_location_data(loc_df)
    taxi_df = clean_taxi_data(taxi_df)

    print("Mergeing data...")

    merged_df = pd.merge(event_list_df, loc_df, on=['event_id'])

    del event_list_df
    del loc_df

    merged_df = merged_df.merge(taxi_df, on=['DOW', 'TOD', 'taxi_zone'], how='right')

    del taxi_df

    merged_df['is_event'] = merged_df.is_event.fillna(0)
    df = merged_df.groupby(['date_y', 'taxi_zone', 'TOD', 'DOW', 'is_event'])[['VendorID']].count().reset_index()

    randomforestmodel(df)

def randomforestmodel(df):
    print("Running Model...")
    df = df.sort_values('date_y')
    rows = df.shape[0]

    train = df.head(int(rows*0.8))
    test = df.tail(int(rows*0.2))

    rfrmodel = RandomForestRegressor(n_estimators=20, n_jobs=-1)
    reg = rfrmodel.fit(train[['taxi_zone', 'TOD']], train['VendorID'])

    training_accuracy = reg.score(train[['taxi_zone', 'TOD']], train['VendorID'])
    test_accuracy = reg.score(test[['taxi_zone', 'TOD', 'is_event']], test['VendorID'])
    print("############# based on standard predict ################")
    print("R^2 on training data: %0.4f" % (training_accuracy))
    print("R^2 on test data:     %0.4f" % (test_accuracy))

if __name__ == '__main__':
    run()
