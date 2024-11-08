################################################################################
# Name: FeatureTransformer.py
# Purpose: Generate Features for Univariate Data
# Date                          Version                Created By
# 8-Nov-2024                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import warnings
import numpy as np
import pandas as pd
from forecast.external.vest.preprocess.embedding import embed_with_target
from forecast.external.vest.models.univariate import UnivariateVEST
from pmdarima.arima import auto_arima
from forecast.external.vest.config.aggregation_functions import SUMMARY_OPERATIONS_FAST
from forecast.external.datetime.datetime_features import add_datepart


from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    Normalizer,
)

from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

class FeatureGen:
    def __init__(
        self,
        p_max=3,
        d_max=1,
        q_max=3,
        scaling_method="RobustScaler",
        feature_selection_method="Boruta",
    ):
        """
        Automatically build features if featuers are not provided
        """
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.scaling_method = scaling_method
        self.feature_selection_method = feature_selection_method

    def date_feature_gen(self, data, date_column):
        """
        Return the date-time features
        Raise exception when data is not in the right format
        """

        # TODO: Add more logical exceptions
        if data is None:
            warnings.warn("data not avialable in the required format")

        else:
            # Extract date time features
            transformed_data = add_datepart(data, date_column, drop=True)
        return transformed_data

    def target_feature_gen(
       self, lag_values, data, train_rows, test_rows, actuals_col, date_frequency
    ):
        """
        Return the lagged values of target variable with time series features like
        central tendency, skewness and kurtosis after discrete fourier transformation
        """
        
        # TODO: add more logical exceptions
        if data is None:
            warnings.warn("data not avialable in the required format")

        # TODO: use of any model for forward forecast
        # Extract the static training data features and dynamic predicted features
        else:
            train = data.iloc[:train_rows]
            test = data.iloc[-test_rows:]
            model = self._fit( train, actuals_col, date_frequency)

            # arima prediction
            prediction = self._predict( model, test)
            # bind
            transformed_data = train[actuals_col].tolist() +  prediction[0].tolist()
            ##vest features           
            series = transformed_data
            X, y = embed_with_target(series, lag_values)
            col_names = ["t-" + str(i) for i in list(reversed(range(lag_values)))]
            col_names[-1] = "t"
            X = pd.DataFrame(X, columns=col_names)
            model = UnivariateVEST()
            model.fit(
                X=X.values,
                correlation_thr=0.9,
                apply_transform_operators=True,
                summary_operators=SUMMARY_OPERATIONS_FAST,
            )
            training_features = model.dynamics
            X = pd.DataFrame(
                np.nan, index=list(range(lag_values)), columns=X.columns
            ).append(X)
            training_features = pd.DataFrame(
                np.nan, index=list(range(lag_values)), columns=training_features.columns
            ).append(training_features)
            X_tr = pd.concat([X, training_features], axis=1)
            X_tr = X_tr.replace([np.nan , 'None'],0)   
            return X_tr

    def _fit(self, train, actuals_col, date_frequency):
            if date_frequency == "Weekly":
                m = 52
            elif date_frequency == "Monthly":
                m = 12
            elif date_frequency == "Daily":
                m = 7

            model = auto_arima(
                train[actuals_col].values,
                start_p=0,
                start_q=0,
                max_p= self.p_max,
                max_d=self.d_max,
                max_q=self.q_max,
                start_P=0,
                start_Q=0,
                max_P=self.p_max,
                max_D=self.d_max,
                max_Q=self.q_max,
                m=m,
                seasonal=True,
                error_action="warn",
                trace=True,
                supress_warnings=True,
                stepwise=True,
                random_state=20,
            )
            return model

    def _predict(self, model, test):
        predict_data = pd.DataFrame(
            model.predict(n_periods=test.shape[0]), index=test.index
            ).reset_index(drop = True)       
        return predict_data
    
    def feature_selection(self, data, target_col):
        X = data.drop([target_col], axis=1)
        X_columns = X.columns.tolist()
        y = data[target_col]
        if self.feature_selection_method == "Boruta":
            rf_model = RandomForestRegressor(n_jobs=4, oob_score=True)
            feat_selector = bp(rf_model, n_estimators="auto", verbose=0, max_iter=100,random_state=1)
            X = X.copy().values
            feat_selector.fit(X, y)
            selected_features = np.array(X_columns)[np.array(feat_selector.support_)].tolist()  
            
        elif self.feature_selection_method == "SequentialFeatureSelector":
            sfs_model = SFS(
                RandomForestRegressor(),
                k_features=10,
                forward=True,
                floating=False,
                verbose=2,
                scoring="r2",
                cv=3,
            )
            sfs_model = sfs_model.fit(np.array(X), y)
            selected_features = [ X_columns[i] for i in list(sfs_model.k_feature_idx_)]

        else:
            sel = VarianceThreshold(threshold=0)
            sel.fit(X)
            selected_features = np.array(X_columns)[np.array(sel.get_support())].tolist()  
        return selected_features

    def normalize(self, data):
        if self.scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif self.scaling_method == "RobustScaler":
            scaler = RobustScaler()
        elif self.scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif self.scaling_method == "Normalizer":
            scaler = Normalizer()
        else:
            scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data),columns = data.columns)
        return scaled_data

