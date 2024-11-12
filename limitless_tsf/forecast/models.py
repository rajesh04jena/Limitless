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
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS


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

def auto_arima_forecast(**kwargs):
    """
    Perform Auto ARIMA forecasting, automatically tuning the parameters p, d, q, and seasonal components.
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_y': The training data (a numpy array or list).
        - 'test_y': The test data (a numpy array or list).
        - 'season_length': The length of the seasonal period (optional, defaults to None).
        - Other parameters for `auto_arima` (e.g., 'm' for seasonality, 'start_p', 'start_q', etc.).
    Returns:
    - predicted_test_y: A numpy array containing the predicted values for the test set.
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
    predicted_test_y = auto_arima_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                           train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted_test_y)
    """
    # Extract values from kwargs
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    m = kwargs["model_params"]["season_length"]
    # We will let auto_arima automatically tune p, d, q, and seasonal components (P, D, Q, m)
    # Ensure there are atleast 2 cycles to detect seasonality or set seasonal=False in auto-arima
    if len(train_y) <= m * 2:
        model = pm.auto_arima(
            train_y,
            seasonal=False,
            m=m,
            trace=True,
            stepwise=True,
            suppress_warnings=True,
        )
    else:
        model = pm.auto_arima(
            train_y,
            seasonal=True,
            m=m,
            trace=True,
            stepwise=True,
            suppress_warnings=True,
        )
    # Make forecasts for the test set and append the in-train values
    predicted_test_y = model.predict(n_periods=len(test_y))
    Y_pred = np.append(train_y, predicted_test_y)
    return Y_pred, model

def simple_exponential_smoothing(**kwargs):
    """
    Simple Exponential Smoothing (SES)
    Parameters (via kwargs):
    - 'train_y': The training data (a numpy array or list).
    - 'test_y': The test data (a numpy array or list).
    - alpha: Smoothing parameter (0 < alpha < 1).
    Returns:
    - forecast: Array of predicted values using SES.
    #Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 102, 104, 103, 105, 107, 108, 110, 112])
    test_y = np.array([121, 122, 124, 123])
    model_params = {'smoothening_parameter' : 0.8 }
    # Using kwargs to pass train_x, test_x, and season_length
    predicted_test_y = simple_exponential_smoothing(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted_test_y)
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    alpha = kwargs["model_params"]["smoothening_parameter"]
    # Initialize the forecasted values list with the first value
    forecast = [train_y[0]]  # The first forecast is just the first data point
    # Apply SES formula to generate forecasts for the training data
    for t in range(1, len(train_y)):
        next_forecast = alpha * train_y[t - 1] + (1 - alpha) * forecast[t - 1]
        forecast.append(next_forecast)
    # Now, recursively forecast for the test period (i.e., beyond the training data)
    for t in range(len(train_y), len(train_y) + len(test_y)):
        next_forecast = alpha * forecast[t - 1] + (1 - alpha) * forecast[t - 1]
        forecast.append(next_forecast)
    return forecast

def double_exponential_smoothing(**kwargs):
    """
    Double Exponential Smoothing (DES)
    Parameters (via kwargs):
    - 'train_y': The training data (a numpy array or list).
    - 'test_y': The test data (a numpy array or list).
    - alpha: Smoothing parameter for level.
    - beta: Smoothing parameter for trend.
    Returns:
    - forecast: Array of predicted values using DES.
    #Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 102, 104, 103, 105, 107, 108, 110, 112])
    test_y = np.array([121, 122, 124, 123])
    model_params = {'level_smoothening_parameter' : 0.8 , 'trend_smoothening_parameter' : 0.8 }
    # Using kwargs to pass train_x, test_x, and season_length
    predicted_test_y = double_exponential_smoothing(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted_test_y)
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    alpha = kwargs["model_params"]["level_smoothening_parameter"]
    beta = kwargs["model_params"]["trend_smoothening_parameter"]
    level = [train_y[0]]  # Initial level
    trend = [train_y[1] - train_y[0]]  # Initial trend (first difference)
    forecast = [level[0] + trend[0]]  # First forecast

    # Apply DES on training data
    for t in range(1, len(train_y)):
        level.append(alpha * train_y[t] + (1 - alpha) * (level[t - 1] + trend[t - 1]))
        trend.append(beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1])
        forecast.append(level[t] + trend[t])  # Forecast is level + trend
    # Forecast for test set
    for t in range(len(train_y), len(train_y) + len(test_y)):
        level.append(level[-1] + trend[-1])  # Update level
        trend.append(trend[-1])
        # Forecast is level + trend
        forecast.append(level[-1] + trend[-1])
    return forecast

def holt_winters_forecast(**kwargs):
    """
    Holt-Winters forecasting using statsmodels with automatic model selection
    between additive and multiplicative based on training data.
    Parameters:
    - 'train_y': The training data (a numpy array or list).
    - 'test_y': The test data (a numpy array or list).
    - alpha: Level smoothing parameter.
    - beta: Trend smoothing parameter.
    - gamma: Seasonal smoothing parameter.
    - m: Seasonal period length (12 for monthly data).

    Returns:
    - forecast: The forecasted values for both training and test periods.
    - selected_model: The model chosen ('additive' or 'multiplicative').

    #Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0, 103, 200,300 ,
                       300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0, 103, 200,300]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0,71002, 16900,120301,
                       41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0,71002, 16900,120301]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 102, 104, 103, 105, 107, 108, 110, 112, 200 , 301, 411,
                        100, 102, 104, 103, 105, 107, 108, 110, 112, 200 , 301, 411])
    test_y = np.array([121, 122, 124, 123])
    model_params = {'level_smoothening_parameter' : 0.8 , 'trend_smoothening_parameter' : 0.8 , 'seasonal_smoothening_parameter' : 0.2 ,'seasonal_length' : 12 }
    # Using kwargs to pass train_x, test_x, and season_length
    predicted_test_y = holt_winters_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted_test_y)
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    alpha = kwargs["model_params"]["level_smoothening_parameter"]
    beta = kwargs["model_params"]["trend_smoothening_parameter"]
    gamma = kwargs["model_params"]["seasonal_smoothening_parameter"]
    m = kwargs["model_params"]["seasonal_length"]
    # Ensure the train data is a pandas Series
    train_series = pd.Series(train_y)
    # Check for additive vs multiplicative based on seasonal variation
    seasonal_additive = np.std(
        np.diff(train_series[:m])
    )  # Standard deviation of first m periods (Additive test)
    seasonal_multiplicative = np.std(
        np.diff(np.log(train_series[:m]))
    )  # Std of log-differenced series (Multiplicative test)
    # Choose model based on seasonal variation
    if seasonal_multiplicative < seasonal_additive:
        selected_model = "multiplicative"
    else:
        selected_model = "additive"
    # Initialize and fit the Holt-Winters model
    model = ExponentialSmoothing(
        train_series,
        trend="add",  # or 'mul' for multiplicative trend
        seasonal="add",  # or 'mul' for multiplicative seasonality
        seasonal_periods=m,
    ).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
    # Fit the multiplicative model if chosen
    if selected_model == "multiplicative":
        model = ExponentialSmoothing(
            train_series,
            trend="mul",  # Multiplicative trend
            seasonal="mul",  # Multiplicative seasonality
            seasonal_periods=m,
        ).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
    # Forecast the test periods
    forecast = model.forecast(len(test_y))
    # Combine training data with forecast
    forecast_combined = np.concatenate((train_y, forecast))
    return forecast_combined, selected_model

def croston_tsb_forecast(**kwargs):
    """
    Croston's TSB (Teunter, Syntetos, and Babai) method for intermittent demand forecasting.    
    Parameters:
    - 'train_y': The training data (a numpy array or list).
    - 'test_y': The test data (a numpy array or list).        
    - test_len: The number of periods to forecast.
    - alpha: Smoothing parameter for average demand.
    - beta: Smoothing parameter for demand period length.    
    Returns:
    - forecast: Forecasted values for the test periods.
    #Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0, 103, 200,300 ,
                           300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0, 103, 200,300]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0,71002, 16900,120301,
                           41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0,71002, 16900,120301]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 0, 104, 0, 105, 0, 108, 0, 112, 0 , 301, 0,
                        100, 0, 104, 0, 105, 0, 108, 0, 112, 0 , 301, 0])
    test_y = np.array([121, 0, 124, 0])
    model_params = {'demand_smoothening_parameter' : 0.3 ,
                    'period_length_smoothening_parameter' : 0.3}
    # Using kwargs to pass train_x, test_x, and season_length
    predicted_test_y = croston_tsb_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted_test_y)
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    alpha = kwargs["model_params"]["demand_smoothening_parameter"]
    beta = kwargs["model_params"]["period_length_smoothening_parameter"]
    # Initialize level (intermittent demand) and period length
    level = [train_y[0]]
    period_length = [1]  # The first demand's period length (assuming the first non-zero value occurs at t=1)    
    # Croston TSB Method Algorithm
    for t in range(1, len(train_y)):
        if train_y[t] != 0:
            level.append(alpha * train_y[t] + (1 - alpha) * level[-1])
            period_length.append(beta * (t - np.argmax(train_y[:t][::-1] != 0)) + (1 - beta) * period_length[-1])
        else:
            level.append(level[-1])
            period_length.append(period_length[-1])
    # Applying bias correction to improve prediction accuracy
    level_corrected = [l / p if p != 0 else 0 for l, p in zip(level, period_length)]    
    # Forecasting for the test period
    forecast = []
    for _ in range(len(test_y)):
        forecast.append(level_corrected[-1])
    forecast_combined = np.append(train_y, forecast)    
    return forecast_combined

def tbats_forecast(**kwargs):
    """
    Perform a multi-seasonality forecast using the TBATS model, 
    capturing multiple seasonalities in the time series.    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_y': The training time series data (a numpy array).
        - 'test_len': The number of periods to forecast.
        - 'seasonal_periods': A list of seasonal periods (e.g., [12, 7] for yearly and weekly seasonality).
        - 'n_jobs': The number of jobs to run in parallel (default is -1 to use all available CPUs).
    
    Returns:
    - Y_pred: The forecasted values for the test period, including the original training data.
    - model: The fitted model 
    #Example Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0,
                       538.0, 118.0, 212.0, 103, 200,300 ]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  ,
                       38506.0 , 84499.0 , 84004.0,71002, 16900,120301]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 120, 130, 140, 110, 115, 150, 160, 170, 165, 180, 190])
    test_y = np.array([121, 0, 124, 0])
    model_params = {'seasonal_periods' : [12,7]}
    # Using kwargs to pass train_x, test_x, and season_length
    predicted_test_y = tbats_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted_test_y)
    """    
    # Extract values from kwargs    
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    seasonal_periods = kwargs["model_params"]["seasonal_periods"]    
    # Initialize TBATS model with seasonal periods
    model = TBATS(seasonal_periods=seasonal_periods)    
    # Fit the model on training data
    fitted_model = model.fit(train_y)    
    # Forecast the next 'test_len' periods
    forecast = fitted_model.forecast(steps= len(test_y))    
    # Combine the original training data with the forecasted values
    Y_pred = np.append(train_y, forecast)    
    return Y_pred, fitted_model

