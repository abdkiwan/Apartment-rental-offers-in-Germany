#!/usr/bin/env python3

'''
This script contains helping function for data preparation for training.
Data preparation includes splitting data into train and test sets, and rescale data.
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_data(df, target_col, columns_to_drop, with_normalization=False):
    '''
    This function extracts features and targets from data frame, splits data into train and test sets,
    and rescale the features (optional)
    '''
    target = df[target_col]
    features = df.drop(target_col, axis=1)
    features = df.drop(columns_to_drop, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)

    if with_normalization:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    return X_train, y_train, X_test, y_test
