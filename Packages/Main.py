from Packages import JobController
from Packages.ModelDev import model_list
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import numpy as np
import random
random.seed(42)
np.seterr(divide='ignore', invalid='ignore')
"""
SETTINGS
"""
# Import available models and select which to train
avail_models = model_list()
selected_models = [
	"ARIMA Forecaster",
	"Daily Split Forecaster",
	"Exponential Smoothing Daily Split Forecaster",
	"Gradient Boosting Machine Auto Regression Forecaster",
	"Gradient Boosting Machine Auto Regression Select",
	"Gradient Boosting Machine Impute Forecaster",
	"Gradient Boosting Machine Split Impute Forecaster",
	"Historical Post Averaging",
	"Historical Pre Averaging",
	"Holt Winters Additive Forecaster",
	"Holt Winters Multiplicative Forecaster",
	"Importance Averaging",
	"Lasso Daily Split Forecaster",
	"Moving Averages Daily Split Forecaster",
	"OLS Daily Split Forecaster",
	"Random Forest Auto Regression Forecast Tune Select",
	"Random Forest Auto Regression Tune Select",
	"Random Forest Impute Tune Forecaster",
	"Trend Exponential Smoothing Daily Split Forecaster",
	"XGBoost Auto Regression Daily Split Select Forecaster",
	"XGBoost Auto Regression Select Forecaster",
	"XGBoost Impute Forecaster"
]
selected_models = [
	"Gradient Boosting Machine Auto Regression Select",
	"Gradient Boosting Machine Impute Forecaster",
	"Gradient Boosting Machine Split Impute Forecaster",
	"Historical Post Averaging",
	"Historical Pre Averaging",
	"Importance Averaging",
	"Random Forest Auto Regression Tune Select",
	"Random Forest Impute Tune Forecaster",
	"XGBoost Auto Regression Select Forecaster",
	"XGBoost Impute Forecaster"
]
selected_units = [12679, 12687, 12743, 12747, 12759,
				  12795, 12855, 12879, 12883, 12903,
				  13087, 13111, 13171, 13211, 13227,
				  13255, 13259, 13271, 13319, 13371,
				  13387, 13407, 13411, 13567, 13639]

selected_models = [selected_models] if type(selected_models) != list else selected_models
# Select models for meta model
selected_base_models = selected_models
# Choose which meta model types to train
meta_models = {'LR': (LinearRegression(), dict()),
			   'RF': (RandomForestRegressor(), dict(n_estimators=1000, max_features='auto')),
			   'RF (hyperparam. tuning)': (RandomForestRegressor(), dict(n_estimators=[100, 1000, 10000],
																		  max_features=['auto', 'sqrt'],
																		  max_depth=[10, 20, 50, 100])),
			   'Lin. SVR': (LinearSVR(), dict(C=100, fit_intercept=True, loss='squared_epsilon_insensitive', dual=False)),
			   'Lin. SVR (hyperparam. tuning)': (LinearSVR(), dict(C=list(np.logspace(-2, 2, 5)),
																   fit_intercept=[False],
																   loss=['squared_epsilon_insensitive'],
																   dual=[False])),
			   'XGBoost': (XGBRegressor(), dict(booster='dart',
												verbosity=1,
												n_estimators=999,
												eta=0.2)),
			   'XGBoost (hyperparam. tuning)': (XGBRegressor(), dict(booster=['dart'],
																	 verbosity=[1],
																	 n_estimators=[999],
																	 eta=[0.2],
																	 gamma=[1, 2, 4],
																	 max_depth=[3, 7, 10],
																	 colsample_bytree=[1/3, 2/3, 1])),
			   'Shallow FNN': (MLPRegressor(), dict(hidden_layer_sizes=(len(selected_base_models)),
													max_iter=10000)),
			   'Deep FNN': (MLPRegressor(), dict(hidden_layer_sizes=(len(selected_base_models), ) * 3 + (10, ) * 2,
												 max_iter=10000))}
meta_models = {'RF (hyperparam. tuning)': (RandomForestRegressor(), dict(n_estimators=[100, 1000, 10000],
																		  max_features=['auto', 'sqrt'],
																		  max_depth=[10, 20, 50, 100])),
			   'XGBoost (hyperparam. tuning)': (XGBRegressor(), dict(booster=['dart'],
																	 verbosity=[1],
																	 n_estimators=[999],
																	 eta=[0.2],
																	 gamma=[1, 2, 4],
																	 max_depth=[3, 7, 10],
																	 colsample_bytree=[1/3, 2/3, 1])),
			   }

team_id = "005befa4-efad-49b3-8c0c-b574bd6f25e3"  # customer to use
max_datasets = 1  # nr of datasets to use
min_length = 17500
time_min = None  # retrieve data from specific date-time
time_max = "2020-03-01T00:00:00.000"  # retrieve data up until specific date-time
test_size = 'last 5 months'  # test set fraction or "last"/"first" + time
o_folder = "C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_stationary/"
conventional = False  # Turn on/off training of conventional model(s)

model_output = JobController.model_training_testing(selected_models, meta_models, team_id,
													train_size=0.7, test_size=test_size, max_datasets=max_datasets, min_length=min_length, time_min=time_min, time_max=time_max,
													o_folder=o_folder, conventional=conventional, feat_selection=False, selected_units=selected_units,
													stationarity=False)
