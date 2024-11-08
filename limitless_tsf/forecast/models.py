################################################################################
# Name: models.py
# Purpose: Generate time series predictions using different statiscal and ML models
# Date                          Version                Created By
# 8-Nov-2024                  1.0         Rajesh Kumar Jena(Initial Version)
################################################################################
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    Normalizer,
)

def linear_regression_forecast(**kwargs):
    """
    Perform a linear regression forecast, predicting the value from the corresponding period in the training set.
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training exogenious/features data (a numpy array).
        - 'test_x': The test exogenious/features data (a numpy array).
        - 'train_y': The training time series data (a numpy array).
        - 'test_y': The test time series data (a numpy array).
        - 'scaling_method': Standardization of datasets is a common requirement for many machine learning estimators
    Returns:
    - Y_pred: A numpy array containing the predicted values for the test set and in-train predictions.
    Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 102, 104, 103, 105, 107, 108, 110, 112])
    test_y = np.array([121, 122, 124, 123])
    model_params = {'scaling_method' : 'MinMaxScaler' }
    # Using kwargs to pass train_x, test_x, and season_length
    predicted= linear_regression_forecast(train_x= train_x , test_x=test_x ,
                                          test_y = test_y, train_y =  train_y, model_params= model_params )
    # Output the predicted values
    print("Predicted In-Train and Test Values:", predicted)
    """

    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    # feature scaling
    if kwargs["model_params"]["scaling_method"] == "StandardScaler":
        scaler = StandardScaler()
    elif kwargs["model_params"]["scaling_method"] == "RobustScaler":
        scaler = RobustScaler()
    elif kwargs["model_params"]["scaling_method"] == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif kwargs["model_params"]["scaling_method"] == "Normalizer":
        scaler = Normalizer()
    else:
        scaler = MinMaxScaler()
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.fit_transform(test_x)
    lr_model = LinearRegression()
    lr_model.fit(scaled_train_x, train_y)
    Y_pred = np.append(
        lr_model.predict(scaled_train_x), lr_model.predict(scaled_test_x)
    )
    return Y_pred, lr_model

def seasonal_naive_forecast(**kwargs):
    """
    Perform a seasonal naive forecast, predicting the value from the corresponding period in the training set.
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training exogenious/features data (a numpy array).
        - 'test_x': The test exogenious/features data (a numpy array).
        - 'train_y': The training time series data (a numpy array).
        - 'test_y': The test time series data (a numpy array).
        - 'season_length': The length of the seasonal period (e.g., 12 for monthly data with yearly seasonality).
    Returns:
    - Y_pred: A numpy array containing the predicted values for the test set and in-train predictions.
    Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 102, 104, 103, 105, 107, 108, 110, 112])
    test_y = np.array([121, 122, 124, 123])
    model_params = {'season_length' : 12 }
    # Using kwargs to pass train_x, test_x, and season_length
    predicted= seasonal_naive_forecast(train_x= train_x , test_x=test_x , test_y = test_y, 
                                       train_y =  train_y, model_params= model_params )
    # Output the predicted values
    print("Predicted In-Train and Test Values:", predicted)
    """
    
    # Extract values from kwargs
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    season_length = kwargs["model_params"]["season_length"]
    # Number of observations in the test set
    n_test = len(test_y)
    # Create an array to store the predictions
    predicted_test_y = np.zeros(n_test)
    # Loop through each test point and assign the value from the corresponding period in train_x
    for i in range(n_test):
        # Calculate corresponding index in the seasonal period
        season_index = (i + len(train_y) - n_test) % season_length
        predicted_test_y[i] = train_y[
            -season_length + season_index
        ]  # Seasonal index in training set
    Y_pred = np.append(train_y, predicted_test_y)
    return Y_pred