################################################################################
# Name: accuracy_metrics.py
# Purpose: Tabulate a host of forecast accuracy metrics
# Date                          Version                Created By
# 9-Nov-2024                  1.0         Rajesh Kumar Jena(Initial Version)
################################################################################
import numpy as np

def mean_squared_error(y_true, y_pred):    
    """
    Calculate Mean Squared Error (MSE) between true and predicted values.
    Parameters:
        -- y_true (numpy array or list): True values
        -- y_pred (numpy array or list): Predicted values

    Returns:
        -- mse (float): Mean Squared Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #squared the differences
    squared_diff = (y_true - y_pred)**2
    #find the avg of squared_diff
    mse = np.mean(squared_diff)
    return mse

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted values.
    Parameters:
        -- y_true (numpy array or list): True values
        -- y_pred (numpy array or list): Predicted values

    Returns:
        -- mse (float): Mean Squared Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #squared the differences
    squared_diff = (y_true - y_pred)**2
    #find the avg of squared_diff
    mse = np.mean(squared_diff)
    return np.sqrt(mse)

def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
        -- y_true (numpy array or list): True values
        -- y_pred (numpy array or list): Predicted values

    Returns:
        -- mae (float): Mean Absolute Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #squared the differences
    absolute_diff = np.absolute(y_true - y_pred)
    #find the avg of squared_diff
    mae = np.mean(absolute_diff)
    return mae

def r2_score(y_true, y_pred):
    """
    Calculate R-squared value given predicted value and actual label

    Parameters:
        -- y_true (numpy array or list): True values
        -- y_pred (numpy array or list): Predicted values

    Returns:
        -- r^2 (float): Coefficient of determination(R^2)
    """
    #convert then into umpy array if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #sum of squared residuals
    SSR = np.sum((y_true - y_pred)**2)
    #sum of squared total
    y_avg = np.mean(y_true)
    SST = np.sum((y_avg-y_true)**2)
    return 1 - (float(SSR)/float(SST))

def adjusted_r2_score(y_true, y_pred, k):
    """
    Calculate Adjusted R-squared value given predicted and actual label and number of predictors

    Parameters:
        -- y_true (numpy array or list): True values
        -- y_pred (numpy array or list): Predicted values
        -- k (integer): Number of predictors

    Returns:
        -- adjusted r^2 (float): Coefficient of determination(R^2)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #number of observations
    n = len(y_true)
    #sum of squared residuals
    SSR = np.sum((y_true - y_pred)**2)
    #sum of squared total
    y_avg = np.mean(y_true)
    SST = np.sum((y_avg-y_true)**2)    
    #r2_score
    r2_score = 1 - (float(SSR)/float(SST))
    #adjusted r2 square
    r2_score_adj = 1 - float((1-r2_score**2)*(n-1))/float(n - k - 1)    
    return r2_score_adj

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    - y_true: Array-like or list of true values.
    - y_pred: Array-like or list of predicted values.
    
    Returns:
    - mape: The calculated MAPE value.
    """
    # Convert input to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    # Avoid division by zero by handling zero true values
    epsilon = 1e-10  # small value to avoid division by zero    
    # Calculate the absolute percentage error
    abs_percentage_error = np.abs((y_true - y_pred) / (y_true + epsilon))    
    # Calculate MAPE
    mape = np.mean(abs_percentage_error) * 100    
    return mape

def symmetric_mape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parameters:
    - y_true: Array-like or list of true values.
    - y_pred: Array-like or list of predicted values.
    
    Returns:
    - smape: The calculated SMAPE value.
    """
    # Convert input to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-10  # small value to avoid division by zero    
    # Calculate the numerator and denominator of the SMAPE formula
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0    
    # Calculate SMAPE
    smape = np.mean(numerator / (denominator + epsilon)) * 100    
    return smape

def mean_absolute_scaled_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Scaled Error (MASE).
    
    Parameters:
    - y_true: Array-like or list of true values.
    - y_pred: Array-like or list of predicted values.
    
    Returns:
    - mase: The calculated MASE value.
    """
    # Convert inputs to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    # Calculate the absolute error of the forecast
    absolute_error = np.abs(y_true - y_pred)    
    # Calculate the naive forecast errors (difference between consecutive true values)
    naive_errors = np.abs(np.diff(y_true))    
    # Calculate the mean absolute error of the forecast and naive model
    mae_forecast = np.mean(absolute_error)
    mae_naive = np.mean(naive_errors)    
    # Calculate MASE (scaling by the naive model's mean absolute error)
    mase = mae_forecast / mae_naive    
    return mase

def weighted_quantile_loss(y_true, y_pred, weights, tau=0.5):
    """
    Calculate the Weighted Quantile Loss (wQL) for a given quantile (tau).
    
    Parameters:
    - y_true: Array-like or list of true values.
    - y_pred: Array-like or list of predicted values.
    - weights: Array-like or list of weights for each data point.
    - tau: The quantile (default is 0.5 for median quantile).    
    Returns:
    - wql: The calculated Weighted Quantile Loss.
    """
    # Convert inputs to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)    
    # Calculate the residuals (errors) between true and predicted values
    residuals = y_true - y_pred    
    # Calculate the loss for each data point
    loss = weights * np.maximum(tau * residuals, (tau - 1) * residuals)    
    # Calculate the mean weighted quantile loss
    wql = np.mean(loss)    
    return wql

def bias(y_true, y_pred):
    """
    Calculate the Bias (mean of the prediction errors).
    
    Parameters:
    - y_true: Array-like or list of true values.
    - y_pred: Array-like or list of predicted values.    
    Returns:
    - bias: The calculated Bias value.
    """
    # Convert inputs to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    
    # Calculate Bias
    bias_value = np.mean(y_pred - y_true)    
    return bias_value

def weighted_bias(y_true, y_pred, weights):
    """
    Calculate the Weighted Bias (mean of the weighted prediction errors).    
    Parameters:
    - y_true: Array-like or list of true values.
    - y_pred: Array-like or list of predicted values.
    - weights: Array-like or list of weights for each data point.
    
    Returns:
    - weighted_bias: The calculated Weighted Bias value.
    """
    # Convert inputs to numpy arrays for element-wise operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)    
    # Calculate Weighted Bias
    weighted_bias_value = np.sum(weights * (y_pred - y_true)) / np.sum(weights)    
    return weighted_bias_value
