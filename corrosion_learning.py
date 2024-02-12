import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly
from pycaret.regression import RegressionExperiment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
import webbrowser
import os


# load data from Zenodo
data = pd.read_csv("https://zenodo.org/records/10640939/files/procraft_corrosion_kbely_data_v02.csv?download=1", index_col=0)

# transform datetime string index into datetime type of pandas dataframe
data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')

# select features for modelling
features_cols = ['PM10_holesovice',
                'PM2_5_holesovice',
                'temp_in',
                'hum_in',
                'dew_in',
                'SO2',
                'windspeed']

# select target for modelling
target = ["corrosion_diff"]

# create datafrem of selected features and target
x = data[features_cols + target]

# apply moving average filter
window_length = 16
x = x.rolling(window_length).mean()

# delete rows with NaN values
x.dropna(inplace=True)


# Prepare models and their parameters
custom_et = ExtraTreesRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1)
custom_knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
custom_gp = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, random_state=None)

models_to_compare = [custom_et, custom_knn, custom_gp]

# the range for training data
limit = 24*360 # 1 year 
x_test = x[49:limit] # at 49 starts the first corrosion incerement

# Instatiate the Pycaret experiment
exp = RegressionExperiment()
# Setup the experiment. Chose the parameters for training like normaliyation, training data size etc
exp.setup(x_test, target="corrosion_diff", session_id = 123, fold=3,
        normalize = True, normalize_method="minmax", preprocess=True,  train_size=0.8, data_split_shuffle=True)

# compare chosen models
best = exp.compare_models(include=models_to_compare, n_select=3, verbose=True)

# function for holdout prediction
def holdout_prediction(exp, best):
    # choose the whole dataset
    x_holdout = x[49:]

    # prepare empty dataframe
    total_predict = pd.DataFrame()
    
    # go through best models and predict the corrosion
    for i in range(len(best)):
        pred = exp.predict_model(best[i], data=x_holdout)
        
        total_predict["corrosion_diff"] = pred["corrosion_diff"]
        total_predict["corr"] = pred["corrosion_diff"].cumsum()
        
        # add the prediction of the model to the dataframe
        total_predict[f"corr_diff_{str(best[i])[:20]}"] = pred["prediction_label"]
        # add the cummulative sum of the prediction to the dataframe
        total_predict[f"corr_{str(best[i])[:20]}"] = pred["prediction_label"].cumsum()
        
    return total_predict

# call the function for holdout prediction      
total_predict = holdout_prediction(exp, best)

# plot the results
fig = total_predict.plot()

# Save the figure to an HTML file
file_path = 'plot.html'
fig.write_html(file_path)
        
# Open the HTML file in the default browser
webbrowser.open('file://' + os.path.realpath(file_path))