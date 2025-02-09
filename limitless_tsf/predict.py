import pandas as pd
import numpy as np
import logging
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from dask import delayed, compute
from limitless_tsf.forecast.models import (
    linear_regression_forecast,
    lasso_regression_forecast,
    ridge_regression_forecast,
    xgboost_regression_forecast,
    lightgbm_regression_forecast,
    random_forest_regression_forecast,
    catboost_regression_forecast,
    seasonal_naive_forecast,
    auto_arima_forecast,
    simple_exponential_smoothing,
    double_exponential_smoothing,
    holt_winters_forecast,
    croston_tsb_forecast,
    tbats_forecast,
    prophet_forecast,
)
from limitless_tsf.forecast.FeatureEngineering import FeatureGen

np.bool = np.bool_

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Identify date column and frequency
def identify_date_and_frequency(df):
    date_cols = df.select_dtypes(include=["datetime64[ns]", "object"]).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            inferred_freq = pd.infer_freq(df[col].dropna())
            if inferred_freq:
                logging.info(
                    f"Identified date column: {col} with frequency: {inferred_freq}"
                )
                return col, inferred_freq
        except Exception:
            continue
    logging.warning("No date column with identifiable frequency found.")
    return None, None


def generate_future_dates(df, date_column, frequency, n_periods):
    """
    Generate future dates based on frequency and number of periods.

    Parameters:
    df (pd.DataFrame): Input dataframe with a date column.
    date_column (str): Name of the date column.
    frequency (str): Frequency of future dates ('D' for daily, 'W' for weekly, 'M' for monthly, etc.).
    n_periods (int): Number of future periods to generate.

    Returns:
    pd.DataFrame: DataFrame with appended current and future dates.
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame")

    last_date = df[date_column].max()
    future_dates = pd.date_range(
        start=last_date, periods=n_periods + 1, freq=frequency
    )[1:]
    future_df = df.iloc[-n_periods:].copy()
    future_df[date_column] = future_dates.values

    return pd.concat([df, future_df], ignore_index=True)


# Fill missing dates and handle gaps
def handle_missing_dates(df, date_col, freq):
    df = df.set_index(date_col)
    df = df.asfreq(freq)
    df = df.fillna(method="ffill").fillna(method="bfill")
    logging.info("Handled missing dates with forward and backward filling.")
    return df.reset_index()


# Data Cleaning and Preprocessing Function
def clean_data(df):
    logging.info("Starting data cleaning process...")

    date_col, freq = identify_date_and_frequency(df)
    if date_col:
        df = handle_missing_dates(df, date_col, freq)

    num_imputer = KNNImputer(n_neighbors=5)
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    logging.info(f"Imputing missing values for numerical columns: {list(num_cols)}")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    logging.info(f"Imputing missing values for categorical columns: {list(cat_cols)}")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_cat = encoder.fit_transform(df[cat_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cat, columns=encoder.get_feature_names_out(cat_cols)
    )

    df_processed = pd.concat(
        [
            df[date_col].reset_index(drop=True),
            df[num_cols].reset_index(drop=True),
            encoded_cat_df,
        ],
        axis=1,
    )
    logging.info("Data cleaning and preprocessing completed")

    return df_processed, date_col, freq


# Helper function for parallel execution
def model_forecast(model_name, train_x, train_y, test_x, model_params):
    try:
        model_func = globals().get(model_name)
        if model_func:
            _, preds, _ = model_func(
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                model_params=model_params,
            )
            return preds
    except Exception as e:
        logging.error(f"Error in model {model_name}: {e}")
        return []


# Forecast Combination Function
def combined_forecast(
    df, target_col, model_list, n_periods, mode="forward", backtest_periods=None
):
    logging.info(f"Starting {mode} process...")
    results = []

    cleaned_df, date_col, freq = clean_data(df)
    train_y = df[target_col].values
    train_x = cleaned_df.drop(columns=[target_col], errors="ignore")

    last_date = df[date_col].max()
    future_dates = pd.date_range(
        start=last_date + to_offset(freq), periods=n_periods, freq=freq
    )

    seasonal_period = 12 if freq == "M" else 7 if freq == "W" else 1
    alpha, beta, gamma = 0.8, 0.2, 0.1

    tbats_seasonal_periods = [12, 7] if freq in ["M", "W"] else [7]

    date_frequency = "Monthly" if freq == "M" else "Weekly" if freq == "W" else "Daily"
    lag_values = int(seasonal_period * 2)
    tsf_features = FeatureGen(
        p_max=3,
        d_max=1,
        q_max=3,
        scaling_method="RobustScaler",
        feature_selection_method="SequentialFeatureSelector",
    )
    if mode == "backtest":
        time_ft_data = cleaned_df.copy()
        time_ft_data = tsf_features.date_feature_gen(time_ft_data, date_column=date_col)
        time_ft_data = time_ft_data.drop(columns=["Elapsed", target_col])
        try:
            new_train_x = tsf_features.target_feature_gen(
                1,
                cleaned_df,
                int(train_x.shape[0] - backtest_periods),
                backtest_periods,
                target_col,
                date_frequency,
            )

            new_data = pd.concat(
                [
                    new_train_x.reset_index(drop=True),
                    pd.DataFrame(df[target_col], columns=[target_col]).reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            )
            new_train_x = new_train_x[
                tsf_features.feature_selection(new_data, target_col)
            ]

        except:
            new_train_x = pd.DataFrame()

        new_train_x = pd.concat(
            [new_train_x.reset_index(drop=True), time_ft_data.reset_index(drop=True)],
            axis=1,
        )

    new_cleaned_data = generate_future_dates(cleaned_df, date_col, freq, n_periods)

    time_ft_data_new = new_cleaned_data.copy()
    time_ft_data_new = tsf_features.date_feature_gen(
        time_ft_data_new, date_column=date_col
    )
    time_ft_data_new = time_ft_data_new.drop(columns=["Elapsed"])
    try:
        new_train_x_ft = tsf_features.target_feature_gen(
            lag_values,
            new_cleaned_data,
            train_x.shape[0],
            n_periods,
            target_col,
            date_frequency,
        )
        new_data_ft = pd.concat(
            [
                new_train_x_ft.reset_index(drop=True),
                pd.DataFrame(
                    pd.concat([df[target_col], df[target_col][-n_periods:]]),
                    columns=[target_col],
                ).reset_index(drop=True),
            ],
            axis=1,
        )
        new_train_x_ft = new_train_x_ft[
            tsf_features.feature_selection(new_data_ft, target_col)
        ]

    except:
        new_train_x_ft = pd.DataFrame()

    time_ft_data_new = time_ft_data_new.drop(columns=target_col)
    new_train_x_ft = pd.concat(
        [
            new_train_x_ft.reset_index(drop=True),
            time_ft_data_new.reset_index(drop=True),
        ],
        axis=1,
    )

    model_tasks = []
    for model_name in model_list:
        model_params = {"scaling_method": "MinMaxScaler"}

        if model_name in [
            "seasonal_naive_forecast",
            "auto_arima_forecast",
            "simple_exponential_smoothing",
            "double_exponential_smoothing",
            "holt_winters_forecast",
            "croston_tsb_forecast",
            "tbats_forecast",
            "prophet_forecast",
        ]:
            model_params.update(
                {
                    "season_length": seasonal_period,
                    "smoothening_parameter": alpha,
                    "level_smoothening_parameter": alpha,
                    "trend_smoothening_parameter": beta,
                    "seasonal_smoothening_parameter": gamma,
                    "demand_smoothening_parameter": alpha,
                    "period_length_smoothening_parameter": beta,
                    "seasonal_periods": tbats_seasonal_periods,
                    "date_col": date_col,
                    "freq": freq,
                    "n_periods": n_periods,
                }
            )

        if mode == "backtest":

            test_x = new_train_x[-backtest_periods:]
            train_x_bt = new_train_x[:-backtest_periods]
            train_y_bt = train_y[:-backtest_periods]
            dates_bt = df[date_col].values[-backtest_periods:]

            task = delayed(model_forecast)(
                model_name, train_x_bt, train_y_bt, test_x, model_params
            )
            model_tasks.append((task, dates_bt, model_name))
        else:

            test_x_ft = new_train_x_ft[-n_periods:]
            train_x_ft = new_train_x_ft[:-n_periods]

            task = delayed(model_forecast)(
                model_name, train_x_ft, train_y, test_x_ft, model_params
            )
            model_tasks.append((task, future_dates, model_name))

    computed_results = compute(*[t[0] for t in model_tasks], scheduler="threads")

    for preds, (task, dates, model_name) in zip(computed_results, model_tasks):
        results.extend(
            [
                {"date": date, "prediction": pred, "model": model_name}
                for date, pred in zip(dates, preds)
            ]
        )

    logging.info(f"{mode.capitalize()} process completed")
    return pd.DataFrame(results)


# Example Usage
if __name__ == "__main__":
    logging.info("Script started")

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

    logging.info(f"Forecast Results:\n{forecast_results}")
    logging.info(f"Backtest Results:\n{backtest_results}")
