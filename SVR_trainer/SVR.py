import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from sklearn.svm import SVR
import pickle

import datetime
import os
import subprocess
import sys

def loadDataFiles():
    market_data_filename = 'Market_train'
    news_data_filename = 'News_train'
    data_dir = 'gs://uwotm8'

    subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir, market_data_filename), market_data_filename], stderr=sys.stdout)
    subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir, news_data_filename), news_data_filename], stderr=sys.stdout)


    market_df = pickle.load(open('Market_train',"rb"))
    news_df = pickle.load(open("News_train", "rb"))
    print('Finished loading datafiles!')
    return market_df, news_df

def preprocess_data(mkt_df, news_df):
    mkt_df['time'] = pd.to_datetime(mkt_df['time'])
    news_df['time'] = pd.to_datetime(news_df['time'])
    mkt_df['time'] = mkt_df['time'].dt.date
    news_df['time'] = news_df['time'].dt.date
    assetCodes = []
    index = 0
    for x in news_df['assetCodes']:
        x = x.split(',')[0].split("'")[1]
        assetCodes.append(x)
    news_df['assetCode'] = np.asarray(assetCodes)
    irrelevantColumns = ['sourceTimestamp', 'firstCreated', 'sourceId', 
                         'headline', 'provider', 'subjects', 'audiences',
                        'headlineTag', 'marketCommentary', 'assetCodes', 'assetName']
    news_df.drop(irrelevantColumns, axis=1, inplace=True)
    mkt_df.drop(['assetName'], axis=1, inplace=True)
    modifiednews = news_df.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    merged = pd.merge(mkt_df, modifiednews, how='left', on=['time', 'assetCode'], copy=False) 
    merged = merged.fillna(0)
    print('Finished preprocessing data!')
    return merged

BUCKET_ID = 'uwotm8'
market_data, news_data = loadDataFiles()
X = preprocess_data(market_data, news_data)

def normalizeY(ydf):
    ydf = (ydf + 1) / 2
    return ydf

X = X[X['returnsOpenNextMktres10'] >=-1]
X = X[X['returnsOpenNextMktres10'] <=1]

y = X['returnsOpenNextMktres10']

X.drop(['returnsOpenNextMktres10'], axis=1, inplace=True)
y = normalizeY(y)
assetCodesAndTime = X.iloc[:, :2]
X = X.iloc[:, 2:]

def regularize(df):
    for column in df:
        colmin = np.amin(df[column])
        colmax = np.amax(df[column])
        df[column] = (df[column] - colmin) / (colmax - colmin)
        print(df[column])
    return df

X = regularize(X)

model = SVR(cache_size=1000, verbose = True)
model.fit(X.iloc[:round(len(X) * 0.7), :], y.iloc[:round(len(y) * 0.7)])
