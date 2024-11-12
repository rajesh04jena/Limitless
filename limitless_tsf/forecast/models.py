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
from prophet import Prophet

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

def prophet_forecast(**kwargs):
    """
    Perform forecasting using the Prophet model for time series data    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_y': The training time series data (a numpy array or pandas Series).
        - 'test_len': The number of periods to forecast.
        - 'train_x': The dates corresponding to the training time series (pandas DatetimeIndex).
        - 'holidays_train': A dataframe of holidays for the training period (optional).
        - 'holidays_future': A dataframe of holidays for the future prediction period (optional).
        - 'n_changepoints': The number of potential changepoints (default is 25).
        - 'changepoint_range': The proportion of history in which to place changepoints (default is 0.8).
        - 'seasonality_prior_scale': The prior scale for seasonal components (default is 10.0).
        - 'changepoint_prior_scale': The prior scale for changepoints (default is 0.05).
        - 'interval_width': The width of the uncertainty intervals (default is 0.80).
        - 'uncertainty_samples': The number of samples for uncertainty (default is 1000).    
    Returns:
    - Y_pred: The forecasted values for the test period, including the original training data.
    # Example Usage:
    # Generating some synthetic data for training
    train_x = pd.date_range(start='2021-01-01', periods=100, freq='D')  # 100 days of data
    train_y = np.sin(np.linspace(0, 10, 100)) * 20 + 100  # Synthetic time series data (sinusoidal pattern)
    test_y =  pd.date_range(start='2021-04-11', periods=10, freq='D')
    # Forecasting the next 10 days
    test_len = len(test_y)
    # Creating a DataFrame for holidays during the training period
    holidays_train = pd.DataFrame({
        'ds': pd.to_datetime(['2021-02-14', '2021-04-01', '2021-12-25']),  # Example holidays
        'holiday': ['Valentine', 'Easter', 'Christmas']
    })
    # Creating a DataFrame for future holidays
    holidays_future = pd.DataFrame({
        'ds': pd.to_datetime(['2022-02-14', '2022-04-01', '2022-12-25']),  # Example future holidays
        'holiday': ['Valentine', 'Easter', 'Christmas']
    })
    # Using Prophet to forecast with holiday effects and automatic seasonality mode selection
    forecasted_data = prophet_forecast(
        train_x=train_x,
        train_y=train_y,
        test_len=test_len,
        holidays_train=holidays_train,
        holidays_future=holidays_future
    )
    print("Forecasted Data: ", forecasted_data)
    """
    # Extract values from kwargs    
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )    
    holidays_train = kwargs.get("holidays_train", None)  # Holidays during training
    holidays_future = kwargs.get("holidays_future", None)  # Holidays during future prediction    
    # Other hyperparameters
    n_changepoints = kwargs.get("n_changepoints", 25)
    changepoint_range = kwargs.get("changepoint_range", 0.8)
    seasonality_prior_scale = kwargs.get("seasonality_prior_scale", 10.0)
    changepoint_prior_scale = kwargs.get("changepoint_prior_scale", 0.05)
    interval_width = kwargs.get("interval_width", 0.80)
    uncertainty_samples = kwargs.get("uncertainty_samples", 1000)    
    # Prepare the training data for Prophet
    df_train = pd.DataFrame({'ds': train_x, 'y': train_y})    
    # **Automatically select seasonality mode (additive or multiplicative) based on data:**
    seasonality_mode = 'additive'  # Default to additive    
    # Analyze the coefficient of variation (CV) to determine seasonality type
    # CV = (Standard deviation of seasonal component) / (Mean of seasonal component)
    seasonal_variation = np.std(train_y) / np.mean(train_y)    
    if seasonal_variation > 0.2:
        seasonality_mode = 'multiplicative'  # Use multiplicative if variance increases with level
    else:
        seasonality_mode = 'additive'  # Otherwise, use additive        
    # Initialize the Prophet model with the passed parameters
    model = Prophet(
        holidays=holidays_train,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=interval_width,
        uncertainty_samples=uncertainty_samples
    )    
    # Fit the model to the training data
    model.fit(df_train)    
    # Prepare the future dataframe for prediction
    future = model.make_future_dataframe(df_train, periods= len(test_y))    
    # If holidays for future predictions are provided, include them
    if holidays_future is not None:
        future = future.merge(holidays_future, on='ds', how='left')
    # Forecast the future
    forecast = model.predict(future)    
    # Extract the forecasted values
    Y_pred = forecast['yhat'][-len(test_y):].values
    Y_combined = np.append(train_y ,Y_pred)
    return Y_combined, model
