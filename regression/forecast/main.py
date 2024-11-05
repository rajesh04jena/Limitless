import calendar
import configparser
import copy
import logging
from datetime import datetime
from pyper import *
import numpy as np
import pandas as pd
from scipy import stats
from seasonal import fit_seasons, adjust_seasons
from forecast.cortex_be import dataQuality
from forecast.cortex_be import dataForecastability
from forecast.utility.constant import Constant
from forecast.common import application_constants as app_const
from forecast import FeatureTransformer as ft

from forecast.cortex_be.cortex_all_algos_list import cortex_algo_univariate_multivariate as ca


import json
import re
from forecast.cortex_be.check_dates import CheckDates
from forecast.utility.common_utility import CommonUtility
# import traceback
props = configparser.ConfigParser()
config_file = CommonUtility().get_config_file(module_name=__file__)
props.read(config_file)

class PythonFunctionalities(CheckDates):

    def __init__(self,df, all_params):
        self.data = df
        self.all_params=all_params
        self.original_message = copy.deepcopy(self.all_params)
        self.isError=False
        self.forecastable = True
        self.preprocessingError= " "
        self.jobId = job_id
        self.failure_reason = {}
        self.checkDataQualityTests = False
        self.checkForecastabilityTests = False
        self.checkForecastValidationTests = False
        self.convert_to_date()
        self.orig_date = self.data[self.all_params['processing_parameters']['dataParams']['datetime_column_name']]
## Define datatype of each column.
    def define_datatype(self):
        """
        :return: define data type for each of the column in the dataframe.
        """
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})

        try:
            self.datetype_column_type = self.all_params['processing_parameters']['dataParams']['datetime_column_format']
            if self.all_params['processing_parameters']['dataParams']['datatypes_of_the_columns']==None:
                return self.data
            else:
                self.data['converted_date'] = self.converted_date.dt.date
                return self
        except Exception as e:
            logger.warning(Constant.DEFINE_DATATYPE_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.DEFINE_DATATYPE)
            logger.error(self.preprocessingError)
            # traceback.print_exc()
        return self

    def set_value_(self, val, default_val):
        return default_val if val is None else val

    def create_generalized_constants(self):
        """
        :return: returns a object with all the required parameter defined in job
        """
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})
        try:
            
            self.input_frequency=self.all_params['dataParams']['input_frequency']
            self.output_frequency=self.all_params['dataParams']['output_frequency']
            self.holdout_date=self.all_params['dataParams']['holdout_date']
            self.last_date=self.all_params['dataParams']['last_date']
            self.datetype_column_type=self.all_params['dataParams']['datetime_column_format']
            self.datetime_column_name=self.all_params['dataParams']['datetime_column_name']
            self.forecast_column=self.all_params['dataParams']['forecast_column_name']
            self.predictor_column=self.all_params['dataParams']['predictor_columns']
            self.train_test_split_ratio=self.all_params['dataParams']['train_test_split_ratio']
            self.error_metrics=self.all_params['modelParams']['error_metrics']
            self.ranking_metric=self.all_params['modelParams']['ranking_metric']
            self.nested_algorithms=self.all_params['modelParams']['nested_algorithm']
             self.includeDatetimeFeatures =  False
            self.includeTargetFeatures =  False
            self.includeLagFeaturesThreshold = 5
            self.enableFeatureSelection = False
           
            
            
            
            self.algorithms=list(self.nested_algorithms.keys())
            # self.algorithms=self.all_params['processing_parameters']['modelParams']['algorithms']
            self.outlier_value = self.all_params['dataParams']['outlier_value']
            self.outlier_threshold = self.all_params['dataParams']['outlier_threshold']
            self.outlier_method = self.all_params['dataParams']['outlier_method']
            self.grain=self.all_params['grain']
            self.allCols = self.all_params['processing_parameters']['dataParams']['name_of_the_input_columns_in_the_file']
            self.useExistingStoredModels = self.all_params['processing_parameters']['modelParams']['useExistingStoredModels']
            self.storeModels = self.all_params['processing_parameters']['modelParams']['storeModels']
            self.jobIdOfStoredModels = self.all_params['processing_parameters']['modelParams']['jobIdOfStoredModels']
            self.dayOfWeek = [0, 1, 2, 3, 4, 5, 6]
            self.includeHolidayEffects = self.all_params['processing_parameters']['modelParams']['includeHolidayEffects']
            self.countryCodeColumnName = self.all_params['processing_parameters']['modelParams']['countryCodeColumnName']
            self.seasonalComparisonPeriod=self.all_params['processing_parameters']['modelParams']['seasonalComparisonPeriod']
            self.computeTrendSeasonality = self.all_params['processing_parameters']['modelParams']['computeTrendSeasonality']
            self.holidayColnames = []
            self.generatedDummyColumns=[]
            self.requested_ranks = self.all_params['processing_parameters']['modelParams']['requested_ranks']
            self.categorical_columns = self.all_params['processing_parameters']['dataParams']['categorical_column_names']
            self.isEnsemble =  self.all_params['processing_parameters']['modelParams']['isEnsemble']
            self.negative_value_impute_rule = self.set_value_(self.all_params['processing_parameters']['dataParams']['negative_value_impute_rule'], None)
            self.checkDataQualityTests = self.set_value_(self.all_params['processing_parameters']['modelParams']['dataQualityTest'], False)
            self.checkForecastabilityTests = self.set_value_(self.all_params['processing_parameters']['modelParams']['forecastabilityTest'], False)
            self.checkForecastValidationTests = self.set_value_(self.all_params['processing_parameters']['modelParams']['forecastValidationTest'], False)
            self.min_training_period = self.all_params['processing_parameters']['modelParams']['min_training_period']
            self.naive_method = self.all_params['processing_parameters']['modelParams']['naive_method']
            self.isLTF = self.all_params.get('processing_parameters', {}).get('modelParams', {}).get('longTermForecast', False)
            self.isFFORMA = self.all_params.get('processing_parameters', {}).get('modelParams', {}).get('isFFORMA', False)
            # self.enable_shap_values = self.all_params['processing_parameters']['modelParams'].get('enableShapValues', False) # be default will be False
            self.retain_best_algo_period = self.all_params.get('processing_parameters', {}).get('modelParams', {}).get('retain_best_algo_period', 6) # defaults to 6 months
            self.config_params_file_path = self.all_params.get('processing_parameters', {}).get('modelParams', {}).get('config_params_file_path')
            self.local_meta_model_file_path =  self.all_params.get('processing_parameters', {}).get('modelParams', {}).get('local_meta_model_file_path')
            self.LTF_exclude_list = []
            if self.isLTF:
                self.LTF_algorithm_list = ltf_algos.LTF_Algorithm
                self.LTF_exclude_list = list(set(ltf_algos.LTF_Algorithm) - (set(self.algorithms) & set(ltf_algos.LTF_Algorithm)))
                self.algorithms = list(set(self.algorithms + ltf_algos.LTF_Algorithm))

            if self.isFFORMA:
                ml_metadata_path = self.config_params_file_path
                self.algorithms  = list(set(self.algorithms).union(set(fforma_algos.get_ml_metadata(ml_metadata_path))))


            if self.naive_method == None:
                self.naive_method = "last_observed"
            self.n_period = self.all_params.get('processing_parameters', {}).get('modelParams',
                                                                                  {}).get('n_moving_avg',
                                                                                          Constant.N_PERIOD)

            # By default set dummy columns flag to False
            # If the user has not provided any predictor_columns, then generate dummy columns
            self.create_dummies_datatime_column = False
            if self.predictor_column == None or not self.predictor_column:
                self.create_dummies_datatime_column = True

            # Check for edge condition: By default datetime column is added as predictor column.
            # If datetime columns is the only predictor column provided then create dummy columns
            if len(self.predictor_column) == 1 and self.datetime_column_name in self.predictor_column:
                self.create_dummies_datatime_column = True

            #setting dummies columns as False for vest features and date-time features generation
            self.create_dummies_datatime_column = False

            path = props['r-properties']['rFilePath']
            path = path + "Params.json"
            with open(path) as f:
                self.config = json.load(f)


            if self.min_training_period is None:
                if self.output_frequency == "Weekly":
                    self.min_training_period = 26
                elif self.output_frequency == "Monthly":
                    self.min_training_period = 12
                elif self.output_frequency == "Daily":
                    self.min_training_period = 182

            if self.output_frequency == "Weekly":
                self.forecast_frequency = 52
            elif self.output_frequency == "Monthly":
                self.forecast_frequency = 12
            else:
                self.forecast_frequency = 365

        except Exception as e:
            logger.warning(Constant.GENERALIZED_CONSTANTS_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.GENERALIZED_CONSTANT)
            logger.error(self.preprocessingError)
            self.isError = True
            pass
        ## This datetime column is what we are getting from user, for ex: a column with intrger value 201701,201702 cannot
        # be considered as date column automatically by program, We have to tell it externally to the program that it is datetime column
        # and value 201701 is not integer, it is 1st week of 2017.

        return self


    def format_datetime_cols(self):
        """
        :return: depending of format of datatime_column, we do processing. if datetime_column_name list has only one column then convert datetime_column_name list to string
        for ex: if type is YYYY-WM --> we convert it to the YYYYMM or YYYYWW
         if type is YYYY and WM --> we join those to column and create a new column with name YYYYWM and also change datetime_column_name to YYYYWM (which automatically converts list to string)
        """
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})
        try:
            if self.datetype_column_type in ["YYYY-WW", "YYYY-MM"]:
                self.data[self.datetime_column_name] = self.data[self.datetime_column_name].str.replace("-", "")
                self.data[self.datetime_column_name] = self.data[self.datetime_column_name].astype(int)
            #
            # elif self.datetype_column_type=="WM and YYYY":
            #     self.week_col=self.datetime_column_name[0]
            #     self.data[self.week_col]=self.data[self.week_col].apply(lambda x: ('0'+str(x) if len(str(x))==1 else str(x)))
            #     self.year_col=self.datetime_column_name[1]
            #     self.data['YYYYWM']=self.data[self.year_col].astype(str)+self.data[self.week_col].astype(str)
            #     self.data['YYYYWM']=self.data['YYYYWM'].astype(int)
            #     self.datetime_column_name='YYYYWM'
            #     self.predictor_column.remove(self.week_col)
            #     self.predictor_column.remove(self.year_col)
            #     self.predictor_column.append('YYYYWM')
            #     self.data.drop([self.week_col,self.year_col], axis=1, inplace=True)
            #
            # elif self.datetype_column_type=="YYYYWM":
            #     self.data[self.datetime_column_name]=self.data[self.datetime_column_name].astype(int)
            #
            # elif self.datetype_column_type=="DATE":
            #     self.datetime_column_name=self.datetime_column_name
            #     #self.data[self.datetime_column_name] = self.data[self.datetime_column_name].astype(int)

        except Exception as e:

            logger.warning(Constant.FORMAT_DATETIME_COLS_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.FORMAT_DATETIME_COLS)
            logger.error(self.preprocessingError)
            self.isError = True
            pass

        return self

    def customQuantileOutlierTreatment(self):

        outlier_value_upper = float(self.outlier_value) / 100
        outlier_value_lower = np.round(1 - outlier_value_upper, 2)
        column = self.forecast_column
        max_value = self.data[column].quantile(outlier_value_upper)
        min_value = self.data[column].quantile(outlier_value_lower)
        self.data.loc[(self.data[column] > max_value)|(self.data[column] < min_value), column] = self.data[column].mean()
        return self

    def zscoreOutlierTreatment(self):
        column = self.forecast_column
        threshold = int(self.outlier_threshold)
        z = np.abs(stats.zscore(self.data[column]))
        self.data[column][pd.Series(z > threshold)] = self.data[column].mean()

        return self

    def residOutlierTreatment(self):

        column = self.forecast_column
        frq = self.forecast_frequency
        threshold = int(self.outlier_threshold)
        seasons, trend = fit_seasons(self.data[column], period=frq)
        if seasons is not None:

            adjusted = adjust_seasons(self.data[column], seasons=seasons)
            resid = adjusted - trend
        else:
            resid = self.data[column] - trend
        resid_q = resid.quantile([0.05, 0.95])
        iqr = np.diff(resid_q)

        limits = resid_q + (1.5 * iqr * [-1, 1])
        diffL = (resid - list(limits)[0]) / iqr
        diffU = (resid + list(limits)[1]) / iqr
        # print list(np.minimum([(resid - list(limits)[1])/iqr], 0))
        pMin = np.array([diffL, np.zeros(len(diffL))]).min(axis=0)
        pMax = np.array([diffU, np.zeros(len(diffU))]).max(axis=0)
        score = pMin + pMax

        self.data[column][pd.Series(score > threshold)] = self.data[column].mean()
        return self

    ## Clean data by removing outliers. Outlier value is given by user
    def apply_outlier_treatment(self):
        """
        :return: self
        replace outliers above nth percentile with mean values.
        """
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})
        logger.info("self.outlier_method is -----> %s ", self.outlier_method)
        logger.info("inside apply_outlier_treatment")
        try:

            if not self.outlier_method:
                return self

            # self.outlier_method = self.outlier_method[0]

            if self.outlier_value is None:
                return self

            elif (self.outlier_value is None) or (self.outlier_value == "0") or (self.outlier_value == ""):
                if self.outlier_method == "z-score":
                    self.zscoreOutlierTreatment()
                    return self
                elif self.outlier_method == "resid":
                    self.residOutlierTreatment()
                    return self
                else:
                    return self
            elif self.outlier_method == "custom-quantile":
                self.customQuantileOutlierTreatment()
                return self
            else:
                return self
        except Exception as e:
            logger.warning(Constant.OUTLIER_TREATMENT_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.OUTLIER_TREATMENT)
            logger.error(self.preprocessingError)
            self.isError = True
            pass
        return self

    ## It creates a dataframe with date,week,month mapping.
    ## DATEVALUE=date, YYYYMM == Year Month corresponding to that date, YYYYWW == Year Week corresponding to that date.
    ## We will use this dataframe in 2 applications:
    ## 1. Converting date to week, date to month for aggregation and 2. fiiling in missing values.

    def create_date_Week_Month_mapping(self, agg_freq):
        all_days = pd.date_range(Constant.DAILY_DATA_MIN, Constant.DAILY_DATA_MAX, freq='D')
        daily_Data=pd.DataFrame()
        single_date=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        daily_Data['DATEVALUE']=all_days
        if agg_freq == "Weekly":
            daily_Data['week'] = daily_Data.apply(lambda x: ('0' + str(x['DATEVALUE'].isocalendar()[1])) if (x['DATEVALUE'].isocalendar()[1] in (single_date)) else str(x['DATEVALUE'].isocalendar()[1]), axis=1)
            daily_Data['YWMS'] = daily_Data.apply(lambda x: (str(x['DATEVALUE'].isocalendar()[0]) + str(x['week'])), axis=1)
        elif agg_freq == "Monthly":
            daily_Data['DATEVALUE'] = pd.to_datetime(daily_Data['DATEVALUE'])
            daily_Data['YWMS'] = daily_Data['DATEVALUE'].dt.strftime('%Y%m').astype(str)
        elif agg_freq == "Daily":
            daily_Data['DATEVALUE'] = pd.to_datetime(daily_Data['DATEVALUE'])
            daily_Data['YWMS'] = pd.to_datetime(daily_Data['DATEVALUE'].astype(str)).apply(lambda x: x.strftime("%Y%m%d"))
        return daily_Data

    ## Below function is used for aggregation and imputation.
    def data_aggregation_datewise(self):
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})
        try:
            #self.data = puiadsfb
            columns_we_want= [self.forecast_column ] + self.predictor_column
            # Aggegration
            if self.input_frequency != self.output_frequency:

                if (self.input_frequency=="Daily" and self.output_frequency=="Weekly"):
                    daily_Data=self.create_date_Week_Month_mapping("Weekly")
                    daily_Data['DATEVALUE'] = pd.to_datetime(daily_Data['DATEVALUE'].astype(str)).apply(lambda x: x.strftime("%Y%m%d"))
                elif (self.input_frequency=="Daily" and self.output_frequency=="Monthly"):
                    daily_Data=self.create_date_Week_Month_mapping("Monthly")
                    daily_Data['DATEVALUE'] = pd.to_datetime(daily_Data['DATEVALUE'].astype(str)).apply(lambda x: x.strftime("%Y%m%d"))

                self.data=daily_Data.merge(self.data, left_on='DATEVALUE', right_on=self.datetime_column_name, how='inner')
                self.data[self.datetime_column_name]=self.data['YWMS']
                self.data=self.data[columns_we_want]

            ## Aggregate the data
            if self.input_frequency !=self.output_frequency and (self.output_frequency == "Weekly" or self.output_frequency == "Monthly"):
                agg_colums = list(set(columns_we_want) ^ set([self.forecast_column]))
                self.data = self.data.groupby(agg_colums, as_index=False).agg({self.forecast_column: 'sum'})
                self.data[self.datetime_column_name] = self.data[self.datetime_column_name].astype(int)

        except Exception as e:
            logger.warning(Constant.DATA_AGGREGATION_DATEWISE_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.DATA_AGGREGATION_DATEWISE)
            logger.error(self.preprocessingError)
            self.isError = True
            pass
        return self

    def negative_data_imputation(self):
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})

        try:
            columns_we_want = self.forecast_column

            if not self.negative_value_impute_rule:
                pass

            elif self.negative_value_impute_rule == "Replace with 0":
                self.data[columns_we_want][self.data[columns_we_want] < 0] = 0

            elif self.negative_value_impute_rule == "Replace with 1":
                self.data[columns_we_want][self.data[columns_we_want] < 0] = 1

            elif self.negative_value_impute_rule == "Replace with NaN and Interpolate Linearly":
                self.data[columns_we_want][self.data[columns_we_want] < 0] = np.nan
                self.data[columns_we_want] = self.data[columns_we_want].interpolate()

        except Exception as e:
            logger.warning(Constant.NEGATIVE_DATA_IMPUTATION_WARNING)
            self.preprocessingError = (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.NEGATIVE_DATA_IMPUTATION)
            logger.error(self.preprocessingError)
            self.isError = True
            pass

        return self

    def data_imputation(self):
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})
        try:
            columns_we_want= ([self.forecast_column]
                              + self.predictor_column
                              + ['converted_date'])

            if self.output_frequency != "Daily":
                stdt = self.data[self.datetime_column_name].min()
                start_date = int(stdt)
                end_date = self.last_date
                all_ywms = list(self.make_dummy_dates_data(start_date, self.last_date, self.output_frequency))

            elif self.output_frequency == "Daily":
                stdt = self.data.converted_date.min()
                start_date = stdt
                end_date = datetime.strptime(str(self.last_date), "%Y%m%d").date()
                all_ywms = list(pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").astype('int'))
                self.data[self.datetime_column_name] = pd.to_datetime(self.data.converted_date).dt.strftime("%Y%m%d").astype(int)

            complete_YWMS=pd.DataFrame()
            complete_YWMS['YWMS']=all_ywms
            complete_YWMS['YWMS']=complete_YWMS['YWMS'].astype(int)
            self.data=complete_YWMS.merge(self.data, left_on='YWMS', right_on=self.datetime_column_name, how='left')
            self.data[self.datetime_column_name]=self.data['YWMS'].astype(int)
            self.data=self.data[columns_we_want]
            self.data.loc[:, self.data.select_dtypes(include=["object"]).columns] = self.data.select_dtypes(include=['object']).\
                apply(lambda x: x.fillna(x.mode()[0]))
            self.data.fillna(0,inplace=True)   ### For any missing week, it fills sales, causal variables values with 0
        except Exception as e:
            logger.warning(Constant.DATA_IMPUTATION_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.DATA_IMPUTATION)
            logger.error(self.preprocessingError)
            self.isError = True
            pass
        return self

    # Output of this function is a list with values between start end.
    # Ex: Let's suppose you call make_dummy_dates_data(self,201701, 201810, "Weekly")
    # Output will be: [201701, 201702, 201703.......201751, 201752, 201801, 201802,....201810]
    # For Monthly output will be [201701, 201702, 201703.......201711, 201712, 201801, 201802,....201810]
    def make_dummy_dates_data(self,start_date, end_date, freq):
        all_ywms = self.create_date_Week_Month_mapping(freq)['YWMS'].astype(int).unique()
        all_ywms = all_ywms[(all_ywms >= start_date) & (all_ywms <= end_date)]
        all_ywms = all_ywms.tolist()
        return all_ywms

    def is_forecastable(self):
        '''
        :return bool: Bast on the train_x and output frequency, it will return if it is forecastable
        '''
        threshold = 0.8
        if self.output_frequency == "Weekly":
            if self.train_x.shape[0] <= self.min_training_period:
                self.forecastable = False
                self.data_failure_reason = app_const.data_quality_data_completeness_test_failure_reason
            # if ((self.train_y[-52:] == 0).values.sum()/52.0) > threshold:
            #     self.forecastable = False
            #     self.data_failure_reason = app_const.data_quality_data_intermittent_test1_failure_reason
            # if ((self.train_y[-2 * 52:] == 0).values.sum()/(2 * 52.0)) > threshold:
            #     self.forecastable = False
            #     self.data_failure_reason = app_const.data_quality_data_intermittent_test2_failure_reason
        elif self.output_frequency == "Monthly":
            if self.train_x.shape[0] <= self.min_training_period:
                self.forecastable = False
                self.data_failure_reason = app_const.data_quality_data_completeness_test_failure_reason
            # if ((self.train_y[-12:] == 0).values.sum()/12.0) > threshold:
            #     self.forecastable = False
            #     self.data_failure_reason = app_const.data_quality_data_intermittent_test1_failure_reason
            # if ((self.train_y[-2 * 12:] == 0).values.sum()/(2 * 12.0)) > threshold:
            #     self.forecastable = False
            #     self.data_failure_reason = app_const.data_quality_data_intermittent_test2_failure_reason
        elif self.output_frequency == "Daily":
            if self.train_x.shape[0] <= self.min_training_period:
                self.forecastable = False
                self.data_failure_reason = app_const.data_quality_data_completeness_test_failure_reason
            # if ((self.train_y[-365:] == 0).values.sum()/365.0) > threshold:
            #     self.forecastable = False
            #     self.data_failure_reason = app_const.data_quality_data_intermittent_test1_failure_reason
            # if ((self.train_y[-2 * 365:] == 0).values.sum()/(2 * 365.0)) > threshold:
            #     self.forecastable = False
            #     self.data_failure_reason = app_const.data_quality_data_intermittent_test2_failure_reason

    def data_quailty_tests(self):
        '''
        :return bool and failure reason: Based on the train_x, train_y and output frequency, it will return data quality check
        '''
        return dataQuality.data_quality_tests(self.output_frequency, self.train_x, self.train_y, self.min_training_period)

    def data_forecastability(self):
        """
        :return bool and failure reason: Based on train_y and output frequency, it will return data forecastability flag
        """
        return dataForecastability.data_forecastability_tests(self.output_frequency, self.train_x, self.train_y)

    def mandatory_dataQuality_tests(self):
        '''
        :return bool: Bast on the train_x and output frequency, it will return if it is forecastable
        '''

        forecast_col = self.forecast_column
        if (self.train_y[forecast_col] == 0).all():
            self.forecastable = False
            self.data_failure_reason = app_const.AllZeroSales

        if self.output_frequency == "Weekly":
            if self.train_x.shape[0] <= self.min_training_period:
                self.forecastable = False
                self.data_failure_reason = app_const.InsufficientData

        elif self.output_frequency == "Monthly":
            if self.train_x.shape[0] <= self.min_training_period:
                self.forecastable = False
                self.data_failure_reason = app_const.InsufficientData

        elif self.output_frequency == "Daily":
            if self.train_x.shape[0] <= self.min_training_period:
                self.forecastable = False
                self.data_failure_reason = app_const.InsufficientData

        return self



    def train_test_split(self):

        """
        :return: split the data into train and test. Aldo modifies. Predictor_column
        """
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})

        try:
            #self.data = self.data.sort_values(by='converted_date')
            self.data = self.data.sort_values(by=self.datetime_column_name)
            self.dummy_colname_out()
            if self.input_frequency == "Daily" and self.output_frequency == "Daily":
                if (self.includeHolidayEffects):
                    HolidayDataFile = props['holiday-data']['dataPath']
                    hdatM = pd.read_csv(HolidayDataFile)
                    hdatM['ds'] = pd.to_datetime(hdatM['ds']).dt.strftime("%Y%m%d").astype(int)
                    self.data = pd.merge(self.data, hdatM, how="left", left_on=[self.countryCodeColumnName, self.datetime_column_name], right_on=[self.countryCodeColumnName, "ds"])
                    self.create_holiday_dummies()
                col_dayofwk=self.generatedDummyColumns
                self.data[col_dayofwk[0]] = pd.Categorical(
                    pd.to_datetime(self.data[self.datetime_column_name].astype(str)).dt.weekday_name, \
                    categories=list(calendar.day_name), ordered=True)

                predictor_columns = self.predictor_column + col_dayofwk
                if self.includeHolidayEffects:
                    predictor_columns = predictor_columns + self.holidayColnames

            elif not (self.input_frequency == "Daily" and self.output_frequency == "Daily"):
                self.data['dummy_column_wm'] = self.data[self.datetime_column_name].astype(str).str[4:]
                self.data['dummy_column_year'] = self.data[self.datetime_column_name].astype(str).str[:4]
                self.data[self.datetime_column_name] = self.data[self.datetime_column_name].astype(int)
                self.predictor_column = self.all_params['processing_parameters']['dataParams']['predictor_columns']
                if self.datetime_column_name not in self.predictor_column:
                    self.predictor_column.append(self.datetime_column_name)
                #predictor_columns = self.predictor_column + ['dummy_column_wm'] + ['dummy_column_year']
                predictor_columns = self.predictor_column + self.generatedDummyColumns

            else:
                predictor_columns = self.predictor_column

            predictor_columns = pd.unique(predictor_columns).tolist()

            # If the user has not provided any predictor_columns and selected multivariate algos, then generate date-time and target columns

            multivariate_algorithms_subset = {key: value for key, value in ca.items() if value == 'Multivariate'}
            multivariate_algorithms = list(multivariate_algorithms_subset.keys())
            multivariate_algorithms_selected = len(set(self.algorithms).intersection(set(multivariate_algorithms)))
            
            if self.predictor_column == None or not self.predictor_column and (multivariate_algorithms_selected > 0):
                self.includeDatetimeFeatures = True
                self.includeTargetFeatures = True

            # Check for edge condition: By default datetime column is added as predictor column.
            if (len(self.predictor_column) == 1) and (self.datetime_column_name in self.predictor_column) and (multivariate_algorithms_selected > 0):
                self.includeDatetimeFeatures = True
                self.includeTargetFeatures = True

            ## Currently we don't do forecast lesser than weekly, so we can use dummy_column line otherwise we have to comeup with some logic here

            if self.train_test_split_ratio !=[0,0] and self.train_test_split_ratio != None:
                training = int(self.train_test_split_ratio[0]*(self.data.shape[0]))
                testing = self.data.shape[0]-training#int(self.train_test_split_ratio[1]*(self.data.shape[0]))
                self.train_x=self.data[:training][predictor_columns]
                self.train_y=self.data[:training][[self.forecast_column]]
                self.test_x=self.data[training:training+testing][predictor_columns]
                self.test_y=self.data[training:training+testing][[self.forecast_column]]
            else:
                self.train_x=self.data.loc[self.data[self.datetime_column_name] <= self.holdout_date][predictor_columns]
                self.train_y=self.data.loc[self.data[self.datetime_column_name] <= self.holdout_date][[self.forecast_column]]
                self.train_date=self.data.loc[self.data[self.datetime_column_name] <= self.holdout_date][self.datetime_column_name]
                self.test_x=self.data.loc[(self.data[self.datetime_column_name] > self.holdout_date) & (self.data[self.datetime_column_name] <= self.last_date)][predictor_columns]
                self.test_y=self.data.loc[(self.data[self.datetime_column_name] > self.holdout_date) & (self.data[self.datetime_column_name] <= self.last_date)][[self.forecast_column]]
                self.test_date=self.data.loc[(self.data[self.datetime_column_name] > self.holdout_date) & (self.data[self.datetime_column_name] <= self.last_date)][self.datetime_column_name]

            ## Addition of date-time features and target variable(vest) features for univariate data

            if (self.includeDatetimeFeatures is True) and (self.includeTargetFeatures is False):
                feature_model = ft.FeatureGen()
                univ_data = self.data[[self.datetime_column_name, self.forecast_column, 'converted_date']]
                train_rows = self.train_y.shape[0]
                test_rows = self.test_y.shape[0]
                datetime_feature_data = univ_data[['converted_date']]
                datetime_feature_data = feature_model.date_feature_gen(datetime_feature_data, 'converted_date')
                #normalization
                datetime_feature_data = feature_model.normalize(datetime_feature_data)
                #Feature Selection
                if self.enableFeatureSelection:
                    train_df =  pd.concat([ datetime_feature_data.iloc[:train_rows].reset_index(drop=True),univ_data[self.forecast_column].iloc[:train_rows].reset_index(drop=True)] , axis=1)
                    predictor_columns = feature_model.feature_selection(train_df,self.forecast_column)
                else :
                    predictor_columns = datetime_feature_data.columns.tolist()
                datetime_feature_data = datetime_feature_data[predictor_columns]
                self.train_x = datetime_feature_data.iloc[:train_rows]
                self.test_x = datetime_feature_data.iloc[-test_rows:]
                self.predictor_column  = predictor_columns

            elif (self.includeTargetFeatures is True) and (self.includeDatetimeFeatures is False):
                feature_model = ft.FeatureGen()
                univ_data = self.data[[self.datetime_column_name, self.forecast_column]]
                train_rows = self.train_y.shape[0]
                test_rows = self.test_y.shape[0]
                target_feature_data = feature_model.target_feature_gen(self.includeLagFeaturesThreshold, univ_data, train_rows, test_rows, self.forecast_column, self.output_frequency)
                #normalization
                target_feature_data = feature_model.normalize(target_feature_data)
                #Feature Selection
                if self.enableFeatureSelection:
                    train_df =  pd.concat([ target_feature_data.iloc[:train_rows].reset_index(drop=True),univ_data[self.forecast_column].iloc[:train_rows].reset_index(drop=True)] , axis=1)
                    predictor_columns = feature_model.feature_selection(train_df,self.forecast_column)
                else :
                    predictor_columns = target_feature_data.columns.tolist()
                target_feature_data = target_feature_data[predictor_columns]
                self.train_x= target_feature_data.iloc[:train_rows]
                self.test_x= target_feature_data.iloc[-test_rows:]
                self.predictor_column  = predictor_columns

            elif (self.includeDatetimeFeatures is True) and (self.includeTargetFeatures is True) :
                feature_model = ft.FeatureGen()
                univ_data = self.data[[self.datetime_column_name, self.forecast_column,'converted_date']]
                train_rows = self.train_y.shape[0]
                test_rows = self.test_y.shape[0]
                #date time features
                try :
                    datetime_feature_data = feature_model.date_feature_gen(univ_data, 'converted_date')
                except :
                    datetime_feature_data =  self.data[[self.datetime_column_name]]
                #vest features
                try :
                    target_feature_data= feature_model.target_feature_gen(self.includeLagFeaturesThreshold, univ_data, train_rows, test_rows, self.forecast_column, self.output_frequency)
                    all_feature_data = pd.concat([datetime_feature_data.reset_index(drop=True), target_feature_data.reset_index(drop=True)] , axis=1)
                except :
                    all_feature_data = datetime_feature_data
                #normalization
                all_feature_data = feature_model.normalize(all_feature_data)
                #feature selection
                if self.enableFeatureSelection is True:
                    train_df = pd.concat([ all_feature_data.iloc[:train_rows].reset_index(drop=True),univ_data[self.forecast_column].iloc[:train_rows].reset_index(drop=True)] , axis=1)
                    predictor_columns = feature_model.feature_selection(train_df,self.forecast_column)
                    # if after selction predictor columns list is empty add datetime_column name as predictor    
                    if bool(predictor_columns):
                        predictor_columns = [self.datetime_column_name]
                    else:
                        pass
                else :
                    predictor_columns = all_feature_data.columns.tolist()
                #train test split
                all_feature_data = all_feature_data[predictor_columns]
                self.train_x= all_feature_data.iloc[:train_rows]
                self.test_x= all_feature_data.iloc[-test_rows:]
                self.predictor_column  = predictor_columns
            else :
                self.train_x=self.data.loc[self.data[self.datetime_column_name] <= self.holdout_date][predictor_columns]
                self.test_x=self.data.loc[(self.data[self.datetime_column_name] > self.holdout_date) & (self.data[self.datetime_column_name] <= self.last_date)][predictor_columns]

            # drop the datetime column from train_x and test_x
            self.train_x.drop(self.datetime_column_name, axis = 1, inplace=True)
            self.test_x.drop(self.datetime_column_name, axis = 1, inplace=True)
            # self.is_forecastable()

            self.forecastable = True
            self.data_failure_reason = ''

            self.mandatory_dataQuality_tests()

            if self.train_x.shape[0] == 0:
                self.forecastable = False
                logger.error("No training data")
            if self.test_x.shape[0] == 0:
                self.forecastable = False
                logger.error("No testing data")

            self.testing_data_length_actuals_y=self.train_y.append(self.test_y)
            self.testing_data_length_actuals_y.reset_index(drop=True, inplace=True)
            self.testing_data_length_actuals_date=self.train_date.append(self.test_date)
            self.testing_data_length_actuals_date.reset_index(drop=True, inplace=True)
            self.testing_data_length=len(self.test_x)

            if self.testing_data_length_actuals_date.empty:
                dummy_data=self.create_date_Week_Month_mapping(self.output_frequency)
                dummy_data["YWMS"] = dummy_data["YWMS"].astype(int)
                self.testing_data_length_actuals_date = dummy_data.loc[(dummy_data["YWMS"]>self.holdout_date) &
                                                                       (dummy_data["YWMS"]<=self.last_date)]["YWMS"].unique().tolist()
            # logger.info("self.train_x %s", self.train_x)
            # logger.info("self.train_y %s", self.train_y)
            # logger.info("self.test_x %s", self.test_x)
            # logger.info("self.test_y %s", self.test_y)

            # print "self.train_x.shape", self.train_x.shape
            # print "self.test_x.shape", self.test_x.shape

        except Exception as e:
            logger.warning(Constant.TRAIN_TEST_SPLIT_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.TRAIN_TEST_SPLIT)
            logger.error(self.preprocessingError)
            self.isError = True
            pass

        return self

    def dummy_colname_out(self):
        if self.input_frequency == "Daily" and self.output_frequency == "Daily":
            self.generatedDummyColumns=["dummy_column_dayofwk"]
        else:
            # catgorical
            if self.create_dummies_datatime_column:
                if self.categorical_columns:
                    self.generatedDummyColumns = self.categorical_columns + ['dummy_column_wm']
                else:
                    self.generatedDummyColumns = ['dummy_column_wm']

            else:
                if self.categorical_columns:
                    self.generatedDummyColumns = self.categorical_columns
                else:
                    self.generatedDummyColumns = []


    def create_holiday_dummies(self):

        '''
        Parameters
        ----------
        self: reference to the instance of the class

        creates holiday dummy variables in the dataset
        '''

        tmp = copy.deepcopy(self.data)
        cNames = self.data.columns
        dat1 = self.data[cNames]
        dat1['Is_Holiday'] = (~pd.isnull(self.data.holiday)) * 1
        subDat = self.data[["lower_window", "upper_window", "category"]].dropna()

        subUpper = subDat.reindex(subDat.index.repeat((subDat.upper_window).astype(int)))
        subUpper.index = subUpper.index - subUpper.groupby(level=0).cumcount(ascending=False) - 1

        subUtmp = subUpper['category'].astype(str).str.get_dummies().sum(level=0). \
            reindex(tmp.index, fill_value=0)

        subLower = subDat.reindex(subDat.index.repeat((subDat.lower_window).astype(int)))
        subLower.index = subLower.index + subLower.groupby(level=0).cumcount() + 1

        subLtmp = subLower['category'].astype(str).str.get_dummies().sum(level=0). \
            reindex(tmp.index, fill_value=0)

        dat1['Is_Holiday_after'] = ((subLtmp.T == 1).any()) * 1
        dat1['Is_Holiday_before'] = ((subUtmp.T == 1).any()) * 1

        subUpLowDat = pd.concat([subUpper, subLower, subDat])['category'].astype(int).astype(str) + "_day_cat"
        hDummies = subUpLowDat.str.get_dummies().sum(level=0).reindex(tmp.index, fill_value=0)

        self.data = pd.concat([dat1, hDummies], axis=1)

        self.holidayColnames = ['Is_Holiday', 'Is_Holiday_after', 'Is_Holiday_before'] + list(hDummies.columns)
        return self

    def rm_cols(self):
        if self.input_frequency == "Daily" and self.output_frequency == "Daily" and self.includeHolidayEffects:
            self.train_x.drop([self.countryCodeColumnName], axis=1, inplace=True)
            self.test_x.drop([self.countryCodeColumnName], axis=1, inplace=True)
        return self

    def create_dummy_var(self):
        """
        :return: create dummy variables.
        """
        logger_instance = logging.getLogger(__name__)
        logger = logging.LoggerAdapter(logger_instance, {'jobId': self.jobId})
        try:
            catg_cols = self.train_x.select_dtypes(include=['object']).columns.tolist() + self.generatedDummyColumns
            catg_cols = pd.unique(catg_cols)
            if (self.input_frequency == "Daily") and (self.output_frequency == "Daily") and self.includeHolidayEffects:
                catg_cols.remove(self.countryCodeColumnName)

            self.train_x=pd.get_dummies(self.train_x, columns=catg_cols)
            self.test_x=pd.get_dummies(self.test_x, columns=catg_cols)
            self.rm_cols()
            for elm in set(self.test_x.columns)-set(self.train_x.columns): self.train_x[elm]=0
            for elm in set(self.train_x.columns)-set(self.test_x.columns): self.test_x[elm]=0
        except Exception as e:
            logger.warning(Constant.CREATE_DUMMY_VAR_WARNING)
            self.preprocessingError += (Constant.PREPROCESSING_ERROR % (str(e))).format(function_specific=Constant.CREATE_DUMMY_VAR)
            logger.error(self.preprocessingError)
            self.isError = True
            pass

        return self
