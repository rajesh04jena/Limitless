################################################################################
# Name: models.py
# Purpose: Generate time series predictions using different statistical and ML models
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
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
#import catboost as cb

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
    model_params = {'scaling_method' : 'MinMaxScaler' }
    # Using kwargs to pass train_x, test_x, and season_length
    fitted, predicted, lr_model = linear_regression_forecast(train_x= train_x , test_x=test_x ,
                                          test_y = test_y, train_y =  train_y, model_params= model_params )
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    Y_fitted = lr_model.predict(scaled_train_x)    
    Y_pred = lr_model.predict(scaled_test_x)
    
    return Y_fitted, Y_pred, lr_model

def lasso_regression_forecast(**kwargs):
    """
    Perform a Lasso regression forecast, predicting the value from the corresponding period in the training set.
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training exogenous/features data (a numpy array).
        - 'test_x': The test exogenous/features data (a numpy array).
        - 'train_y': The training time series data (a numpy array).
        - 'test_y': The test time series data (a numpy array).
        - 'scaling_method': Standardization of datasets is a common requirement for many machine learning estimators.
        - 'alpha': The regularization strength for the Lasso model (float).
    Returns:
    - Y_pred: A numpy array containing the predicted values for the test set and in-train predictions.
    - lasso_model: The fitted Lasso model.
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
    model_params = {'scaling_method' : 'MinMaxScaler' ,"alpha": 0.1 }
    # Using kwargs to pass train_x, test_x, and season_length
    fitted, predicted, lasso_model = lasso_regression_forecast(train_x= train_x , test_x=test_x ,
                                          test_y = test_y, train_y =  train_y, 
                                          model_params= model_params )
    # Output the predicted values
    print("Predicted Test Data Values:", predicted) 
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )    
    # Scaling the features
    scaling_method = kwargs["model_params"].get("scaling_method", "MinMaxScaler")
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_method == "Normalizer":
        scaler = Normalizer()
    else:
        scaler = MinMaxScaler()  # Default scaler
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.transform(test_x)  
    # Lasso regression model
    alpha = kwargs["model_params"].get("alpha", 1.0)  # Default regularization strength is 1.0
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(scaled_train_x, train_y)    
    # Predicting values for both train and test data
    Y_fitted = lasso_model.predict(scaled_train_x)
    Y_pred =  lasso_model.predict(scaled_test_x)
        
    return Y_fitted, Y_pred, lasso_model

def ridge_regression_forecast(**kwargs):
    """
    Perform a Ridge regression forecast, predicting the value from the corresponding period in the training set.
    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training exogenous/features data (a numpy array).
        - 'test_x': The test exogenous/features data (a numpy array).
        - 'train_y': The training time series data (a numpy array).
        - 'test_y': The test time series data (a numpy array).
        - 'scaling_method': Standardization of datasets is a common requirement for many machine learning estimators.
        - 'alpha': The regularization strength for the Ridge model (float).    
    Returns:
    - Y_pred: A numpy array containing the predicted values for the test set and in-train predictions.
    - ridge_model: The fitted Ridge model.    
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
    model_params = {'scaling_method' : 'MinMaxScaler' ,"alpha": 0.1 }
    # Using kwargs to pass train_x, test_x, and season_length
    fitted, predicted, ridge_model = ridge_regression_forecast(train_x= train_x , test_x=test_x ,
                                      test_y = test_y, train_y =  train_y, 
                                      model_params= model_params )
    # Output the predicted values
    print("Predicted Test Data Values:", predicted)    
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )    
    # Scaling the features
    scaling_method = kwargs["model_params"].get("scaling_method", "MinMaxScaler")
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_method == "Normalizer":
        scaler = Normalizer()
    else:
        scaler = MinMaxScaler()  # Default scaler
    scaled_train_x = scaler.fit_transform(train_x)
    scaled_test_x = scaler.transform(test_x)  # Only use transform on the test set
    # Ridge regression model
    alpha = kwargs["model_params"].get("alpha", 1.0)  # Default regularization strength is 1.0
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(scaled_train_x, train_y)    
    # Predicting values for both train and test data
    Y_fitted = ridge_model.predict(scaled_train_x)
    Y_pred = ridge_model.predict(scaled_test_x)
        
    return Y_fitted, Y_pred, ridge_model


def xgboost_regression_forecast(**kwargs):
    """
    Perform an XGBoost regression forecast, predicting the value from the corresponding period in the training set.
    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training exogenous/features data (a numpy array).
        - 'test_x': The test exogenous/features data (a numpy array).
        - 'train_y': The training time series data (a numpy array).
        - 'test_y': The test time series data (a numpy array).
        - 'xgb_params': Dictionary containing hyperparameters for the XGBoost model.
    
    Returns:
    - Y_pred: A numpy array containing the predicted values for the test set and in-train predictions.
    - xgb_model: The fitted XGBoost model.
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
    fitted, predicted, model = xgboost_regression_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        xgb_params={        
                "objective": "reg:squarederror",  # Regression task
                "booster": "gbtree",  # Tree-based boosting
                "n_estimators": 100,  # Number of boosting rounds
                "learning_rate": 0.05,  # Learning rate
                "max_depth": 6,  # Depth of each tree
                "subsample": 0.8,  # Fraction of samples for each tree
                "colsample_bytree": 0.8,  # Fraction of features per tree
                "eval_metric": "rmse",  # Root Mean Squared Error for regression
                "early_stopping_rounds": 10,  # Early stopping rounds        
        }
    )    
    # Output the predicted values
    print("Predicted Test Values:", predicted)        
    """
    # Extract arguments
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )    
    # Extract XGBoost hyperparameters
    xgb_params = kwargs.get("xgb_params", {
        "objective": "reg:squarederror",  # Default objective for regression
        "booster": "gbtree",  # Tree-based boosting
        "n_estimators": 100,  # Default number of trees
        "learning_rate": 0.1,  # Step size shrinkage to improve robustness
        "max_depth": 6,  # Maximum depth of trees
        "min_child_weight": 1,  # Minimum sum of instance weight (hessian) needed in a child
        "subsample": 1,  # Fraction of samples used for each tree
        "colsample_bytree": 1,  # Fraction of features used for each tree
        "colsample_bylevel": 1,  # Fraction of features used for each split level
        "colsample_bynode": 1,  # Fraction of features used for each split node
        "gamma": 0,  # Minimum loss reduction required to make a further partition
        "scale_pos_weight": 1,  # Control the balance of positive and negative weights
        "lambda": 1,  # L2 regularization term on weights
        "alpha": 0,  # L1 regularization term on weights
        "tree_method": "auto",  # Tree construction algorithm ("auto", "exact", "hist", "gpu_hist")
        "eval_metric": "rmse",  # Evaluation metric (root mean square error for regression)
        "early_stopping_rounds": 10,  # Stop training after this many rounds without improvement
        "n_jobs": -1,  # Use all available CPUs
    })    
    # Initialize XGBoost model with specified hyperparameters
    xgb_model = xgb.XGBRegressor(**xgb_params)    
    # Fit the model
    xgb_model.fit(train_x, train_y, 
                  eval_set=[(test_x, test_y)],                   
                  verbose=False)
    # Predicting values for both train and test data
    Y_fitted =  xgb_model.predict(train_x)
    Y_pred = xgb_model.predict(test_x)
    
    return Y_fitted, Y_pred, xgb_model

def random_forest_regression_forecast(**kwargs):
    """
    Perform Random Forest Regression, predicting the value from the corresponding period in the training set.    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training features (a numpy array).
        - 'test_x': The test features (a numpy array).
        - 'train_y': The training target values (a numpy array).
        - 'test_y': The test target values (a numpy array).
        - 'model_params': Dictionary containing the hyperparameters for the Random Forest model.
            - 'n_estimators': Number of trees in the forest (default is 100).
            - 'max_depth': Maximum depth of the tree (default is None).
            - 'min_samples_split': The minimum number of samples required to split an internal node (default is 2).
            - 'min_samples_leaf': The minimum number of samples required to be at a leaf node (default is 1).
            - 'max_features': The number of features to consider when looking for the best split (default is 'auto').
            - 'max_samples': The number of samples to train each tree on, if using bootstrap sampling (default is None).
            - 'bootstrap': Whether bootstrap samples are used when building trees (default is True).
            - 'oob_score': Whether to use out-of-bag samples to estimate the generalization accuracy (default is False).
            - 'n_jobs': The number of jobs to run in parallel for training the model (default is 1).
            - 'random_state': Random seed for reproducibility.
            - 'verbose': Controls the verbosity of the tree-building process.
            - 'warm_start': Whether to reuse the solution of the previous call to fit and add more estimators (default is False).
            - 'class_weight': Weights associated with classes, only applicable for classification.
    
    Returns:
    - Y_pred: A numpy array containing the predicted values for both the training and test sets.
    - rf_model: The trained Random Forest model.
    #Usage
    model_params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": 2,
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 1,
    "warm_start": False
    }
    # Call the Random Forest regression function
    Y_pred, rf_model = random_forest_regression_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        model_params=model_params
    )
    # Print the combined predictions
    print("Predictions: ", Y_pred)
    """    
    # Extract input data
    train_x, train_y, test_x, test_y = kwargs["train_x"], kwargs["train_y"], kwargs["test_x"], kwargs["test_y"]    
    # Extract model parameters, using default values if not provided
    model_params = kwargs.get("model_params", {})
    n_estimators = model_params.get("n_estimators", 100)  # Default 100 trees
    max_depth = model_params.get("max_depth", None)  # No limit on tree depth
    min_samples_split = model_params.get("min_samples_split", 2)  # Default value 2
    min_samples_leaf = model_params.get("min_samples_leaf", 1)  # Default value 1
    max_features = model_params.get("max_features", 2)  # Default is 'auto' (sqrt of number of features)
    max_samples = model_params.get("max_samples", None)  # Default is None (use all samples)
    bootstrap = model_params.get("bootstrap", True)  # Default is True
    oob_score = model_params.get("oob_score", False)  # Default is False
    n_jobs = model_params.get("n_jobs", 1)  # Default is 1 (use a single core)
    random_state = model_params.get("random_state", 42)  # Default random seed
    verbose = model_params.get("verbose", 0)  # Default verbosity is 0 (silent)
    warm_start = model_params.get("warm_start", False)  # Default is False (no warm start)
    
    # Initialize Random Forest Regressor with all the parameters
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start
    )    
    # Train the model on the training data
    rf_model.fit(train_x, train_y)    
    # Predict on both train and test sets
    Y_train_pred = rf_model.predict(train_x)
    Y_test_pred = rf_model.predict(test_x)    
    # Combine train and test predictions into a single array
    Y_pred = np.append(Y_train_pred, Y_test_pred)        
    return Y_pred, rf_model

def lightgbm_regression_forecast(**kwargs):
    """
    Perform LightGBM Regression, predicting the value from the corresponding period in the training set.    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training features (a numpy array or pandas DataFrame).
        - 'test_x': The test features (a numpy array or pandas DataFrame).
        - 'train_y': The training target values (a numpy array).
        - 'test_y': The test target values (a numpy array).
        - 'model_params': Dictionary containing the hyperparameters for the LightGBM model.
            - 'boosting_type': Type of boosting ('gbdt', 'dart', 'goss').
            - 'num_leaves': Maximum number of leaves in one tree.
            - 'max_depth': Maximum depth of the tree.
            - 'learning_rate': Step size for updating the model's weights.
            - 'n_estimators': Number of boosting rounds (trees).
            - 'objective': Objective function ('regression').
            - 'metric': Evaluation metric ('l2' for regression).
            - 'subsample': Fraction of data to use for each boosting round.
            - 'colsample_bytree': Fraction of features to use for each tree.
            - 'min_child_samples': Minimum number of data points required to create a new leaf.
            - 'reg_alpha': L1 regularization term.
            - 'reg_lambda': L2 regularization term.
            - 'random_state': Random seed for reproducibility.
            - 'n_jobs': Number of threads for parallel computation.
    
    Returns:
    - Y_pred: A numpy array containing the predicted values for both the training and test sets.
    - lgb_model: The trained LightGBM model.
    
    #Usage
    # Set the LightGBM hyperparameters
    model_params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "objective": "regression",
        "metric": "l2",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1
    }
    # Call the LightGBM regression function
    Y_pred, lgb_model = lightgbm_regression_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        model_params=model_params
    )
    # Print the combined predictions
    print("Predictions: ", Y_pred)
    """    
    # Extract input data
    train_x, train_y, test_x, test_y = kwargs["train_x"], kwargs["train_y"], kwargs["test_x"], kwargs["test_y"]    
    # Extract model parameters
    model_params = kwargs.get("model_params", {})
    boosting_type = model_params.get("boosting_type", "gbdt")  # Default: gbdt (Gradient Boosting Decision Trees)
    num_leaves = model_params.get("num_leaves", 31)  # Default: 31
    max_depth = model_params.get("max_depth", -1)  # Default: No limit
    learning_rate = model_params.get("learning_rate", 0.1)  # Default: 0.1
    n_estimators = model_params.get("n_estimators", 100)  # Default: 100 trees
    objective = model_params.get("objective", "regression")  # Default: regression (for continuous target)
    metric = model_params.get("metric", "l2")  # Default: l2 (Mean Squared Error)
    subsample = model_params.get("subsample", 1.0)  # Default: 1 (use all data)
    colsample_bytree = model_params.get("colsample_bytree", 1.0)  # Default: 1 (use all features)
    min_child_samples = model_params.get("min_child_samples", 20)  # Default: 20
    reg_alpha = model_params.get("reg_alpha", 0.0)  # L1 regularization
    reg_lambda = model_params.get("reg_lambda", 0.0)  # L2 regularization
    random_state = model_params.get("random_state", 42)  # Random seed
    n_jobs = model_params.get("n_jobs", -1)  # Default: -1 (use all cores)
    # Prepare LightGBM Dataset
    train_data = lgb.Dataset(train_x, label=train_y)
    test_data = lgb.Dataset(test_x, label=test_y, reference=train_data)
    # Set up parameters for the model
    params = {
        "boosting_type": boosting_type,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "objective": objective,
        "metric": metric,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_samples": min_child_samples,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": random_state,
        "n_jobs": n_jobs
    }
    # Train the LightGBM model
    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data]               
    )
    # Predict on both train and test sets
    Y_train_pred = lgb_model.predict(train_x, num_iteration=lgb_model.best_iteration)
    Y_test_pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)    
    # Combine train and test predictions into a single array
    Y_pred = np.append(Y_train_pred, Y_test_pred)    
    return Y_pred, lgb_model

def catboost_regression_forecast(**kwargs):
    """
    Perform CatBoost Regression, predicting the value from the corresponding period in the training set.    
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_x': The training features (a numpy array or pandas DataFrame).
        - 'test_x': The test features (a numpy array or pandas DataFrame).
        - 'train_y': The training target values (a numpy array).
        - 'test_y': The test target values (a numpy array).
        - 'model_params': Dictionary containing the hyperparameters for the CatBoost model.
            - 'iterations': Number of boosting iterations (trees).
            - 'learning_rate': Step size for updating model weights.
            - 'depth': Depth of the trees.
            - 'l2_leaf_reg': L2 regularization term on leaf values.
            - 'loss_function': The loss function ('RMSE' for regression).
            - 'metric_period': Period for printing metrics during training.
            - 'random_state': Random seed for reproducibility.
            - 'cat_features': List of categorical feature indices (if any).
            - 'verbose': Whether to print training process logs (default is False).
            - 'thread_count': Number of threads to use for parallel computation.

    Returns:
    - Y_pred: A numpy array containing the predicted values for both the training and test sets.
    - catboost_model: The trained CatBoost model.
    """    
    # Extract input data
    train_x, train_y, test_x, test_y = kwargs["train_x"], kwargs["train_y"], kwargs["test_x"], kwargs["test_y"]    
    # Extract model parameters
    model_params = kwargs.get("model_params", {})
    iterations = model_params.get("iterations", 1000)  # Number of boosting iterations (default 1000)
    learning_rate = model_params.get("learning_rate", 0.1)  # Learning rate (default 0.1)
    depth = model_params.get("depth", 6)  # Depth of trees (default 6)
    l2_leaf_reg = model_params.get("l2_leaf_reg", 3.0)  # L2 regularization (default 3.0)
    loss_function = model_params.get("loss_function", "RMSE")  # Loss function (default: RMSE for regression)
    metric_period = model_params.get("metric_period", 100)  # Print metrics every `metric_period` iterations
    random_state = model_params.get("random_state", 42)  # Random seed for reproducibility
    cat_features = model_params.get("cat_features", None)  # Categorical feature indices (default: None)
    verbose = model_params.get("verbose", False)  # Whether to print logs during training (default: False)
    thread_count = model_params.get("thread_count", -1)  # Number of threads to use (-1 means use all available cores)
    # Initialize CatBoost Regressor
    catboost_model = cb.CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        metric_period=metric_period,
        random_state=random_state,
        cat_features=cat_features,
        verbose=verbose,
        thread_count=thread_count
    )
    # Train the CatBoost model
    catboost_model.fit(train_x, train_y)
    # Predict on both train and test sets
    Y_train_pred = catboost_model.predict(train_x)
    Y_test_pred = catboost_model.predict(test_x)    
    # Combine train and test predictions into a single array
    Y_pred = np.append(Y_train_pred, Y_test_pred)
    return Y_pred, catboost_model

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
    # Prepare the training and test data for Prophet
    df_train = pd.DataFrame({'ds': train_x, 'y': train_y})  
    df_test = pd.DataFrame({'ds': test_x, 'y': test_y})  
    
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
    future = model.make_future_dataframe(df_test, periods= len(test_y))    
    # If holidays for future predictions are provided, include them
    if holidays_future is not None:
        future = future.merge(holidays_future, on='ds', how='left')
    # Forecast the future
    forecast = model.predict(future)    
    # Extract the forecasted values
    Y_pred = forecast['yhat'][-len(test_y):].values
    Y_combined = np.append(train_y ,Y_pred)
    return Y_combined, model
