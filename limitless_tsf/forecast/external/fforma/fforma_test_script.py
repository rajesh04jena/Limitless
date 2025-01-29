import pandas as pd
from ESRNN import ESRNN
from ESRNN.m4_data import *

from forecast.external.fforma.fforma import FFORMA
from functools import partial
import multiprocessing as mp
import glob

from forecast.external.fforma.meta_model  import (
    MetaModels,
    temp_holdout,
    calc_errors,
    get_prediction_panel
)

from forecast.models import *


X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data( dataset_name= 'Daily' , 
                                                               directory = './data',
                                                             num_obs = 100  )



monthly_train = pd.read_csv('data/m4/Train/Monthly-train.csv' , sep = ',')


info_file = pd.read_csv('data/m4/M4-info.csv' , sep = ',')


temp = info_file[info_file['M4id'] == monthly_train['V1'].iloc[0] ]



temp.to_csv('monthly_train_sample.csv' , index = False)






#Generate predictions on univariate algorithms
#seasonal naive

X_train_df.to_csv('X_train_df.csv' , index = False)



model_params = {'season_length' : 12 }
# Using kwargs to pass train_x, test_x, and season_length

train_x.columns

final_predicted  = pd.DataFrame()

for ts_id in train_x['unique_id'].unique().tolist() :
    
    train_x = X_train_df[X_train_df['unique_id'] == ts_id ]
    
    train_y =  
    test_x =  
    test_y  = 
    
    predicted = seasonal_naive_forecast(train_x= train_x , test_x=test_x , test_y = test_y, 
                                       train_y =  train_y, model_params= model_params )
    
    final_predicted = pd.concat([final_predicted , predicted ])









y_panel = y_panel_df[['unique_id', 'ds', 'y']]
y_hat_panel_fun = lambda model_name: y_panel_df[['unique_id', 'ds', model_name]].rename(columns={model_name: 'y_hat'})

model_names = set(y_panel_df.columns) - set(y_panel.columns)

errors_smape = y_panel[['unique_id']].drop_duplicates().reset_index(drop=True)
errors_mase = errors_smape.copy()

for model_name in model_names:
    errors_smape[model_name] = None
    errors_mase[model_name] = None
    y_hat_panel = y_hat_panel_fun(model_name)

    errors_smape[model_name] = evaluate_panel(y_panel, y_hat_panel, smape)
    errors_mase[model_name] = evaluate_panel(y_panel, y_hat_panel, mase, y_insample_df, seasonality)

mean_smape_benchmark = errors_smape[benchmark_model].mean()
mean_mase_benchmark = errors_mase[benchmark_model].mean()

errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

errors = errors_smape/mean_mase_benchmark + errors_mase/mean_smape_benchmark
errors = 0.5*errors


