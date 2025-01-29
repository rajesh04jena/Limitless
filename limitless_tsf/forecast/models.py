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
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import catboost as cb
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
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - lr_model : The fitted linear regression model.
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
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
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
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
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
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
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
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
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
    fitted, predicted, model = lightgbm_regression_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        model_params=model_params
    )    
    # Print the combined predictions
    print("Predictions: ", predicted)
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
    # Combine train and test predictions into a single array
    Y_fitted = lgb_model.predict(train_x, num_iteration=lgb_model.best_iteration)
    Y_pred = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)    
    return Y_fitted, Y_pred, lgb_model

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
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
        - rf_model: The trained Random Forest model.
    Usage:
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
    fitted, predicted, model = random_forest_regression_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        model_params=model_params
    )    
    # Print the combined predictions
    print("Predictions: ", predicted)            
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
   
    # Combine train and test predictions into a single array
    Y_fitted = rf_model.predict(train_x)
    Y_pred = rf_model.predict(test_x)  
       
    return Y_fitted, Y_pred, rf_model

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
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - catboost_model: The trained CatBoost model.
    Usage:
    train_feature_1 = [300.0, 722.0, 184.0, 913.0, 635.0, 427.0, 538.0, 118.0, 212.0]
    train_feature_2 = [41800.0 , 0.0 , 12301.0, 88104.0  , 21507.0 ,  98501.0  , 38506.0 , 84499.0 , 84004.0]
    train_x = pd.DataFrame({ 'feature_1' : train_feature_1 , 'feature_2' : train_feature_2 }).values
    test_feature_1 = [929.0, 148.0, 718.0, 282.0]
    test_feature_2 = [ 98501.0  , 38506.0 , 84499.0 , 84004.0]
    test_x = np.array([ test_feature_1 , test_feature_2 ])
    test_x = pd.DataFrame({ 'feature_1' : test_feature_1 , 'feature_2' : test_feature_2 }).values
    train_y = np.array([100, 102, 104, 103, 105, 107, 108, 110, 112])
    test_y = np.array([121, 122, 124, 123]) 
    model_params = {
    "iterations": 1000,
    "learning_rate":  0.1 ,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "metric_period": 100,
    "random_state": 42,
    "cat_features": None,
    "verbose": False,
    "thread_count": -1
    }
    # Call the catboost refression function
    fitted, predicted, model = catboost_regression_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        model_params=model_params
    )
    # Print the predictions
    print("Predictions: ", predicted)            
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
    Y_fitted = catboost_model.predict(train_x)
    Y_pred = catboost_model.predict(test_x)    
   
    return Y_fitted, Y_pred, catboost_model

class SeasonalNaiveModel:
    """
    A lightweight class to represent the seasonal naive forecasting model.
    This class provides information about the model type and the seasonal length.
    """    
    def __init__(self, season_length):
        """
        Initialize the Seasonal Naive model with its parameters.
        
        Parameters:
         - 'season_length': The length of the seasonal period 
         (e.g., 12 for monthly data with yearly seasonality).  
        """
        self.season_length = season_length
        self.model_name = "Seasonal Naive Forecast"
        
    def __repr__(self):
        """
        String representation of the model.
        """
        return (f"{self.model_name}\n"
                f"Parameters:\n"
                f"- season period: {self.season_length}")

def seasonal_naive_forecast(**kwargs):
    """
    Perform a seasonal naive forecast, predicting the value from the corresponding period in the training set.
    Parameters:
    - kwargs: Keyword arguments that can include:
        - 'train_y': The training time series data (a numpy array).
        - 'test_y': The test time series data (a numpy array).
        - 'season_length': The length of the seasonal period (e.g., 12 for monthly data with yearly seasonality).
    Returns:
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
        - model: The details about seasonal naive model parameters.
        
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
    fitted, predicted, model = seasonal_naive_forecast(train_x= train_x , test_x=test_x , test_y = test_y, 
                                       train_y =  train_y, model_params= model_params )    
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    Y_fitted = train_y
    Y_pred =  predicted_test_y
    model = SeasonalNaiveModel(season_length=season_length)    
    return Y_fitted, Y_pred, model

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
        - Y_fitted : A numpy array containing the fitted values for training data.
        - Y_pred: A numpy array containing the predicted values for test data.
        - model: The details about arima model parameters.
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
    fitted, predicted, model = auto_arima_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                           train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    
    Y_fitted = train_y
    Y_pred =  model.predict(n_periods=len(test_y))
    
    return Y_fitted, Y_pred, model

class SimpleExponentialSmoothingModel:
    """
    A class to represent the Simple Exponential Smoothing (SES) model, including its parameters and identifiers.
    """
    def __init__(self, smoothening_parameter):
        """
        Initialize the SES model with its parameters.
        
        Parameters:
        - smoothening_parameter: Smoothing parameter (0 < alpha < 1).
        """
        self.smoothening_parameter = smoothening_parameter
        self.model_name = "Simple Exponential Smoothing (SES) Model"
    
    def __repr__(self):
        """
        String representation of the model.
        """
        return (f"{self.model_name}\n"
                f"Parameters:\n"
                f"- Smoothing Parameter (Alpha): {self.smoothening_parameter}")

def simple_exponential_smoothing(**kwargs):
        
    """
    Simple Exponential Smoothing (SES)
    Parameters (via kwargs):
    - 'train_y': The training data (a numpy array or list).
    - 'test_y': The test data (a numpy array or list).
    - alpha: Smoothing parameter (0 < alpha < 1).
    Returns:
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - model: The details about SES model parameters.
    
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
    fitted, predicted, model = simple_exponential_smoothing(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    model = SimpleExponentialSmoothingModel(smoothening_parameter = alpha)
    Y_fitted = train_y
    Y_pred  = forecast 
    return Y_fitted, Y_pred, model

class DoubleExponentialSmoothingModel:
    """
    A class to represent the Double Exponential Smoothing (DES) model, including its parameters and identifiers.
    """
    def __init__(self, alpha, beta):
        """
        Initialize the DES model with its parameters.
        
        Parameters:
        - alpha: Smoothing parameter for level.
        - beta: Smoothing parameter for trend.
        """
        self.alpha = alpha
        self.beta = beta
        self.model_name = "Double Exponential Smoothing (DES) Model"
    
    def __repr__(self):
        """
        String representation of the model.
        """
        return (f"{self.model_name}\n"
                f"Parameters:\n"
                f"- Alpha (Level Smoothing Parameter): {self.alpha}\n"
                f"- Beta (Trend Smoothing Parameter): {self.beta}")

def double_exponential_smoothing(**kwargs):
    """
    Double Exponential Smoothing (DES)
    Parameters (via kwargs):
    - 'train_y': The training data (a numpy array or list).
    - 'test_y': The test data (a numpy array or list).
    - alpha: Smoothing parameter for level.
    - beta: Smoothing parameter for trend.
    Returns:
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - model: The details about DES model parameters.
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
    fitted, predicted, model  = double_exponential_smoothing(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted)
    """
    train_x, train_y, test_x, test_y = (
        kwargs["train_x"],
        kwargs["train_y"],
        kwargs["test_x"],
        kwargs["test_y"],
    )
    alpha = kwargs["model_params"]["level_smoothening_parameter"]
    beta = kwargs["model_params"]["trend_smoothening_parameter"]
    
    # Initialize level, trend, and fitted values
    level = [train_y[0]]  # Initial level
    trend = [train_y[1] - train_y[0]]  # Initial trend (first difference)
    fitted_values = [level[0] + trend[0]]  # First fitted value

    # Apply DES on training data
    for t in range(1, len(train_y)):
        level.append(alpha * train_y[t] + (1 - alpha) * (level[t - 1] + trend[t - 1]))
        trend.append(beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1])
        fitted_values.append(level[t] + trend[t])  # Fitted value is level + trend
    
    # Forecast for test set
    forecast = []
    for _ in range(len(test_y)):
        level.append(level[-1] + trend[-1])  # Update level
        trend.append(trend[-1])
        forecast.append(level[-1] + trend[-1])  # Forecast is level + trend
    
    # Create an instance of the DoubleExponentialSmoothingModel class
    model = DoubleExponentialSmoothingModel(alpha=alpha, beta=beta)
    Y_fitted = fitted_values
    Y_pred =  forecast
    
    return Y_fitted, Y_pred, model

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
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - model: The trained Holt Winters model.
    
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
    fitted, predicted, model  = holt_winters_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                   train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    Y_pred = model.forecast(len(test_y))
    # Get the fitted values (in-train predictions)
    Y_fitted = model.fittedvalues
    
    return Y_fitted, Y_pred, model

class CrostonTSBModel:
    """
    A class to represent the Croston TSB model, including its parameters and identifiers.
    """
    def __init__(self, alpha, beta):
        """
        Initialize the Croston TSB model with its parameters.
        
        Parameters:
        - alpha: Smoothing parameter for average demand.
        - beta: Smoothing parameter for demand period length.
        """
        self.alpha = alpha
        self.beta = beta
        self.model_name = "Croston TSB (Teunter, Syntetos, and Babai) Intermittent Demand Forecasting Model"
    
    def __repr__(self):
        """
        String representation of the model.
        """
        return (f"{self.model_name}\n"
                f"Parameters:\n"
                f"- Alpha (Demand Smoothing Parameter): {self.alpha}\n"
                f"- Beta (Period Length Smoothing Parameter): {self.beta}")

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
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - model: The trained Croston TSB model.
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
    fitted, predicted, model = croston_tsb_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    fitted_values = [train_y[0]]  # Initialize fitted values with the first training value    
    # Croston TSB Method Algorithm
    for t in range(1, len(train_y)):
        if train_y[t] != 0:
            level.append(alpha * train_y[t] + (1 - alpha) * level[-1])
            period_length.append(beta * (t - np.argmax(train_y[:t][::-1] != 0)) + (1 - beta) * period_length[-1])
        else:
            level.append(level[-1])
            period_length.append(period_length[-1])
        
        # Calculate fitted values for the training period
        fitted_values.append(level[-1] / period_length[-1] if period_length[-1] != 0 else 0)    
    # Applying bias correction to improve prediction accuracy
    level_corrected = [l / p if p != 0 else 0 for l, p in zip(level, period_length)]    
    # Forecasting for the test period
    forecast = []
    for _ in range(len(test_y)):
        forecast.append(level_corrected[-1])    
    # Create an instance of the CrostonTSBModel class
    model = CrostonTSBModel(alpha=alpha, beta=beta)    
    Y_fitted = fitted_values
    Y_pred = forecast
    
    return Y_fitted, Y_pred, model

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
    - Y_fitted : A numpy array containing the fitted values for training data.
    - Y_pred: A numpy array containing the predicted values for test data.
    - model: The fitted TBATS model(T: Trigonometric seasonality , 
        B: Box-Cox transformation
        A: ARIMA errors
        T: Trend
        S: Seasonal components)
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
    
    fitted, predicted, model  = tbats_forecast(train_x= train_x , test_x=test_x , test_y = test_y,
                                       train_y =  train_y, model_params = model_params)
    # Output the predicted values
    print("Predicted Test Values:", predicted)
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
    model = model.fit(train_y)    
    # Forecast the next 'test_len' periods
    Y_pred = model.forecast(steps= len(test_y))    
    # Combine the original training data with the forecasted values
    # Get the fitted values (in-train predictions)
    Y_fitted = model.y_hat

    return Y_fitted, Y_pred, model

def prophet_forecast(**kwargs):
    """
    Perform univariate or multivariate forecasting using Prophet    
    Parameters:
    - kwargs: Keyword arguments including:
        - 'train_x': Dates (DatetimeIndex) or DataFrame with 'ds' + regressors
        - 'train_y': Target values (array-like)
        - 'test_x': Future dates (DatetimeIndex) or DataFrame with 'ds' + regressors
        - 'test_y': Future target values (array-like) - optional for prediction
        - 'holidays_train': Historical holidays
        - 'holidays_future': Future holidays
        - Prophet hyperparameters    
    Returns:
        - Y_fitted: Fitted values
        - Y_pred: Forecasted values
        - model: Trained Prophet model
    Usage :    
    #Univariate Forecasting
    # Simple time series without external regressors
    train_x = pd.date_range(start='2023-01-01', periods=100, freq='D')
    train_y = np.random.randn(100).cumsum() + 100
    test_x = pd.date_range(start='2023-04-10', periods=30, freq='D')
    # Run forecast
    fitted, predicted, model = prophet_forecast(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x
    )
    print("Forecasted Data: ", predicted)
    #Multivariate Forecasting
    # Create dataset with regressors
    train_data = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'sales': np.random.randn(100).cumsum() + 100,  # Target
        'marketing_spend': np.random.uniform(1000, 5000, 100),
        'holiday_flag': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
    test_data = pd.DataFrame({
        'ds': pd.date_range(start='2023-04-10', periods=30, freq='D'),
        'marketing_spend': np.random.uniform(1000, 5000, 30),
        'holiday_flag': np.zeros(30)  # Assume no future holidays
    })
    # Run forecast with external regressors
    fitted, predicted, model = prophet_forecast(
        train_x=train_data[['ds', 'marketing_spend', 'holiday_flag']],
        train_y=train_data['sales'],
        test_x=test_data
    )
    print("Forecasted Data: ", predicted)    
    """
    # Extract core data
    train_x = kwargs["train_x"]
    train_y = kwargs["train_y"]
    test_x = kwargs["test_x"]
    
    # Prepare training dataframe
    if isinstance(train_x, pd.DataFrame):
        df_train = train_x.copy()
        df_train['y'] = train_y
    else:
        df_train = pd.DataFrame({
            'ds': pd.to_datetime(train_x),
            'y': train_y
        })

    # Prepare test dataframe (future data)
    if isinstance(test_x, pd.DataFrame):
        df_test = test_x.copy()
    else:
        df_test = pd.DataFrame({
            'ds': pd.to_datetime(test_x)
        })

    # Identify regressor columns (exclude ds/y)
    regressor_cols = [col for col in df_train.columns 
                     if col not in ['ds', 'y'] and not col.startswith('holiday')]

    # Handle holidays
    holidays = pd.concat([
        kwargs.get("holidays_train", pd.DataFrame()),
        kwargs.get("holidays_future", pd.DataFrame())
    ]).drop_duplicates('ds')

    # Seasonality mode detection
    mean_y = np.mean(train_y)
    if mean_y == 0:
        seasonality_mode = 'additive'
    else:
        seasonal_variation = np.std(train_y) / mean_y
        seasonality_mode = 'multiplicative' if seasonal_variation > 0.2 else 'additive'

    # Initialize model
    model = Prophet(
        holidays=holidays if not holidays.empty else None,
        n_changepoints=kwargs.get("n_changepoints", 25),
        changepoint_range=kwargs.get("changepoint_range", 0.8),
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=kwargs.get("seasonality_prior_scale", 10.0),
        changepoint_prior_scale=kwargs.get("changepoint_prior_scale", 0.05),
        interval_width=kwargs.get("interval_width", 0.80),
        uncertainty_samples=kwargs.get("uncertainty_samples", 1000)
    )

    # Add regressors for multivariate case
    for col in regressor_cols:
        model.add_regressor(col)

    # Fit model
    model.fit(df_train)

    # Prepare future dataframe with regressors
    future = df_test[['ds'] + regressor_cols].copy()

    # Make prediction
    forecast = model.predict(future)
    Y_pred = forecast['yhat'].values
    Y_fitted = model.predict(df_train)['yhat'].values

    return Y_fitted, Y_pred, model
