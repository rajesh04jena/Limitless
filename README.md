# **limitless_tsf**

`limitless_tsf` is an open-source time series forecasting library aimed at democratizing access to high-quality forecasting tools and challenging the status quo of expensive and arcane demand planning solutions offered by enterprise software giants like SAP APO, Kinaxis, Blue Yonder, O9 Solutions etc.

With `limitless_tsf`, companies can take control of their time series forecasting needs without relying on costly or difficult-to-integrate proprietary software.

---

## **About**

`limitless_tsf` is built to make advanced time series forecasting techniques accessible, scalable, and easy to implement. Whether you're a small business owner or part of a large enterprise, `limitless_tsf` provides tools that can help you forecast sales, inventory, production, or any other time-dependent data.

It was created in response to the need for a **cost-effective alternative** to the complex and expensive demand planning solutions, which are often inaccessible to small and medium-sized businesses.

By offering a **user-friendly interface** and a suite of modern time series forecasting models, `limitless_tsf` can help organizations make more accurate predictions with minimal effort.

---

## **Salient Features**

- **Cutting-Edge Forecasting Algorithms**: Includes a variety of state-of-the-art algorithms such as ARIMA, Exponential Smoothing (Holt-Winters), Prophet, and more.
- **Easy-to-Use**: Intuitive and simple to integrate into your Python workflow.
- **Flexible Model Configuration**: Choose between different models and fine-tune hyperparameters to optimize your forecast accuracy.
- **Auto Model Selection**: Meta-learner can predict probability of lowest error to all univariate and multivariate algorithms by extracting 42 time series features like entropy, flat spots, acf features, crossing points etc.
- **Pretrained Model Registry**: Model registry helps business users to use pretrained metamodels to improve forecast accuracy at scale by rapid prototyping
- **Seasonality Detection**: Automatically detects multi seasonality in time series data and selects the appropriate model.
- **Cross-Validation Support**: Built-in support for cross-validation and backtesting, allowing you to assess model performance and avoid overfitting.
- **Visualization**: Built-in visualization tools to help you understand and interpret forecasts with ease.
- **Open-Source**: Free and open-source, enabling wide adoption and contribution from the community.
- **This library provides implementations for various time series forecasting models. Below is a sample list of available methods**

|  Multivariate Models           |  Univariate Models              | Stacking Ensemble Model        |
|--------------------------------|---------------------------------|--------------------------------|
| **Linear Regression**          | **Double Exponential Smoothing**| **FFORMA**                     |
| **Lasso Regression**           | **Holt-Winters**                | **Transformer Based Stacking** |
| **Ridge Regression**           | **Croston TSB**                 | **LSTM Based Stacking**        |
| **Xgboost Regression**         | **TBATS**                       | **Bayesian Stacking**          |
| **Light GBM Regression**       | **Seasonal Naive Forecast**     | **Simple Average Stacking**    |
| **Catboost Regression**        | **Auto ARIMA**                  |                                |
| **Random Forest Regression**   | **Simple Exponential Smoothing**|                                |
|                                | **Prophet**                     |                                |


# **Installation**

Step 1: Clone this repository to your local machine and
         navigate to the project directory
```sh
git init
git clone git@github.com:rajesh04jena/DemandForecast.git
cd DemandForecast
```

Step 2: Create a conda environment, install dependencies and launch IDE

Alternative 1(execute command in terminal sequentially) :

```sh
 conda create -n limtless-test-env
 conda activate limtless-test-env
 conda install conda-forge::catboost
 conda install dask
 conda install conda-forge::prophet
 pip install -r requirements.txt
 jupyter notebook 
```

Alternative 2(execute command in terminal sequentially) :
```sh
conda env create -f environment.yml
conda activate limtless-test-env
jupyter notebook
```

# **Example Usage**

Here are a few sample snippets from a subset of offerings:

### **Future Horizon Forecasting and Backtesting of all Multivariate and Univariate models**

```python
import numpy as np
import pandas as pd
from limitless_tsf.predict import combined_forecast

data = pd.DataFrame(
    {
        "date": pd.date_range(start="2023-01-01", periods=30, freq="D"),
        "feature1": np.random.randint(1, 100, 30),
        "feature2": np.random.choice(["A", "B", "C"], 30),
        "feature3": np.random.randn(30) * 10,
        "feature4": np.random.uniform(5, 50, 30),
        "target": np.random.randint(10, 200, 30),
    }
)

models_to_use = [
    "linear_regression_forecast",
    "lasso_regression_forecast",
    "ridge_regression_forecast",
    "xgboost_regression_forecast",
    "lightgbm_regression_forecast",
    "random_forest_regression_forecast",
    "catboost_regression_forecast",
    "seasonal_naive_forecast",
    "auto_arima_forecast",
    "simple_exponential_smoothing",
    "double_exponential_smoothing",
    "holt_winters_forecast",
    "croston_tsb_forecast",
    "tbats_forecast",
    "prophet_forecast",
]

n_periods = 10  # Number of future time steps to forecast
backtest_periods = 5  # Number of backward time steps to forecast

forecast_results = combined_forecast(
    data,
    target_col="target",
    model_list=models_to_use,
    n_periods=n_periods,
    mode="forward",
)

backtest_results = combined_forecast(
    data,
    target_col="target",
    model_list=models_to_use,
    n_periods=n_periods,
    mode="backtest",
    backtest_periods=5,
)

```