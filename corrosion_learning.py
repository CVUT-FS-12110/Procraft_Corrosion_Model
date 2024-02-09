import pandas as pd
pd.options.plotting.backend = "plotly"
import numpy as np
import plotly
import sklearn
from pycaret.regression import RegressionExperiment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("https://zenodo.org/records/10640939/files/procraft_corrosion_kbely_data_v02.csv?download=1", index_col=0)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')

features_cols = ['PM10_holesovice',
                'PM2_5_holesovice',
                'temp_in',
                'hum_in',
                'dew_in',
                'SO2',
                'windspeed']

target = ["corrosion_diff"]

x = data[features_cols + target]

x = x.rolling(16).mean()

x.dropna(inplace=True)

custom_et = ExtraTreesRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1)
custom_knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
custom_gp = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, random_state=None)

models_to_compare = [custom_et, custom_knn, custom_gp]

limit = 24*360
x_test = x[49:limit]
exp = RegressionExperiment()
exp.setup(x_test, target="corrosion_diff", session_id = 123, fold=3,
        normalize = True, normalize_method="minmax", preprocess=True,  train_size=0.8, data_split_shuffle=True)
best = exp.compare_models(include=models_to_compare, n_select=3, verbose=True)

def holdout_prediction(exp, best):
    x_holdout = x[49:]
    total_predict = pd.DataFrame()
    
    for i in range(len(best)):
        pred = exp.predict_model(best[i], data=x_holdout)
        
        total_predict["corrosion_diff"] = pred["corrosion_diff"]
        total_predict["corr"] = pred["corrosion_diff"].cumsum()
        
        total_predict[f"corr_diff_{str(best[i])[:20]}"] = pred["prediction_label"]
        total_predict[f"corr_{str(best[i])[:20]}"] = pred["prediction_label"].cumsum()
        
    return total_predict
       
total_predict = holdout_prediction(exp, best)
total_predict.plot()
        
