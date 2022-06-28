from copy import deepcopy
from datetime import date
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from algorithms4004.forecasters import get_forecaster_class
from sklearn.preprocessing import scale
from Packages.DataProcessing import *
from openpyxl import load_workbook
from meta_model import Model
import numpy as np

"""
Models to use
"""
models = [
	"ARIMA Forecaster",
	"Daily Split Forecaster",
	"Exponential Smoothing Daily Split Forecaster",
	"Gradient Boosting Machine Auto Regression Forecast",
	"Gradient Boosting Machine Auto Regression Forecaster",
	"Gradient Boosting Machine Auto Regression Select",
	"Gradient Boosting Machine Auto Regression",
	"Gradient Boosting Machine Forecaster",
	"Gradient Boosting Machine Impute Forecaster",
	"Gradient Boosting Machine Split Forecaster",
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
	"Random Forest Auto Regression Forecast Tune",
	"Random Forest Auto Regression Forecaster",
	"Random Forest Auto Regression Forest Select",
	"Random Forest Auto Regression Select",
	"Random Forest Auto Regression Tune Select",
	"Random Forest Auto Regression Tune",
	"Random Forest Auto Regression",
	"Random Forest Forecaster",
	"Random Forest Impute Forecaster",
	"Random Forest Impute Tune Forecaster",
	"Random Forest Tune Forecaster",
	"Trend Exponential Smoothing Daily Split Forecaster",
	"XGBoost Auto Regression Daily Split Forecaster",
	"XGBoost Auto Regression Daily Split Select Forecaster",
	"XGBoost Auto Regression Forecaster",
	"XGBoost Auto Regression Select Forecaster",
	"XGBoost Forecaster",
	"XGBoost Impute Forecaster"
]


"""
Score functions
"""


def scoring_metrics():
	"""Returns a list of available scoring metrics for easy checking"""
	return ['R2', 'MAPE', 'MedAPE', 'MPE', 'RMSE', 'SMAPE', 'InvSMAPE', 'InvMADP', 'RMSLE']


def SMAPE(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	mask = y_true != 0
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
		return (np.fabs(y_pred - y_true) / (np.fabs(y_pred) + np.fabs(y_true))/2)[mask].mean() * 100


def MAPE(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	mask = y_true != 0
	return (np.fabs(y_true - y_pred) / y_true)[mask].mean() * 100


def MedAPE(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	mask = y_true != 0
	return np.median((np.fabs(y_true - y_pred) / y_true)[mask]) * 100


def MPE(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	mask = y_true != 0
	return ((y_true - y_pred) / y_true)[mask].mean() * 100


def RMSE(y_true, y_pred):
	rmse = sqrt(mean_squared_error(y_true, y_pred))
	return rmse


def InvSMAPE(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	mask = y_true + y_pred != 0
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
		return 100 * (1 - (np.fabs(y_true - y_pred) / (y_pred + y_true))[mask].mean() * 2)


def InvMADP(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	return 100 * (1 - np.mean(np.abs(y_pred - y_true)) / np.mean(np.abs(y_true)))


def RMSLE(y_true, y_pred):
	rmsle = 1 - np.square(np.log10([i+1 for i in y_pred]) - np.log10([i+1 for i in y_true])).mean() ** 0.5
	return rmsle


def WB_Scores(y_true, y_pred):
	result = {
		'R2': r2_score(y_true, y_pred),
		'MAPE': MAPE(y_true, y_pred),
		'MedAPE': MedAPE(y_true, y_pred),
		'MPE': MPE(y_true, y_pred),
		'RMSE': RMSE(y_true, y_pred),
		'SMAPE': SMAPE(y_true, y_pred),
		'InvSMAPE': InvSMAPE(y_true, y_pred),
		'InvMADP': InvMADP(y_true, y_pred),
		'RMSLE': RMSLE(y_true, y_pred)
	}
	return result


def model_list():
	return models


def check_finished_meta(unit, o_path):
	"""
	Checks if meta models have already been trained for a specific unit
	:param unit: str
	:param o_path: str, folder to look in
	:return: bool
	"""
	for filename in listdir(o_path):
		if unit in filename and filename.endswith(".xlsx"):
			return True
	return False


def train_models(datadict, model_output,
				 s_models=tuple(model_list()), meta_pipeline=False, train_fold=0.7, test_fold=0.1, bestmodel_score='RMSE',
				 save_output=True, save_training_data=False, o_folder=".\\Output\\",
				 verbose=False):
	"""
	:param datadict: nested dictionary: {<attribute Id>: {'data': <time series data>,
	 		 									  		  'timezone': <string tz>}}
	:param model_output: list of dictionaries (see <return>)
	:param s_models: list of models to be trained
	:param meta_pipeline: boolean. If True, level 1 models are trained.
	:param train_fold: only used if meta_pipeline==True. Size of first training fold
	:param test_fold: Size of test set. Can be fraction, or string: "first/last X year(s)/month(s)/week(s)/day(s)/hour(s)/minute(s)/second(s)/microsecond(s)"
	:param bestmodel_score: string. Scoring metric to be used to determine the best performing model
	:param save_output: boolean. If True, output is saved to o_folder
	:param save_training_data: boolean. If True, training data and model preds are saved. Makes output larger.
	:param o_folder: path to save scoring metrics for all models
	:return: List of dictionaries, one for each dataset
			{'models': dict of all models {'model name': model},
			 'model_preds': pd DF of test set + cols for model preds,
			 'model_scores': pd DF of all performance metrics
			 'train_set': json of training set,
			 'test_set': json of actual independent test set (different from model_preds in case of meta,
			 'data_info': dict of strings for file naming convention {'customer': customer, 'unit': str(attId)}
			}
	"""

	"""
	Input section
	"""

	daymap = None
	peak_boost = False
	save_prefix = "Performance"  # output data naming convention prefix

	# Checks
	if bestmodel_score not in scoring_metrics():
		raise ValueError(f"Performance metric '{bestmodel_score}' not recognised. Available metrics: {', '.join(scoring_metrics())}")
	if datadict is None:
		raise ValueError("Parameter 'datadict' cannot be None")

	customer = model_output[0]['data_info']['customer']
	for i, attId in enumerate(datadict):
		# Check if attribute exists in model_output
		attribute_model_output = [j for j in model_output if j['data_info']['unit'] == str(attId) and j['model_scores'] is not None]

		if len(attribute_model_output) > 0:
			models_to_train = [j for j in s_models if j not in attribute_model_output[0]['model_scores']['model'].values]
		else:
			models_to_train = s_models
		print(f"Nr. of models already trained ({attId}): {len(s_models) - len(models_to_train)} ({len(models_to_train)} models left to train)")

		"""
		Data read and prep
		"""
		if len(models_to_train) > 0:
			#### Data read
			df_all = datadict[attId]['df_all']
			initial_cols = datadict[attId]['initial_cols']
			bucket_size = int(df_all.at[0, 'bucketSize'])

			# Split train/test, and split train again for the meta level 1 models
			train, test = transform.split_by_time(df_all, test_fold)
			test_final = test  # test set to be used to validate meta model

			if meta_pipeline:
				train, test = transform.split_by_time(deepcopy(train), 1 - train_fold)
			df_train = pd.DataFrame(train)  # df
			df_test = pd.DataFrame(test)  # testdf

			train_adjust = df_train[initial_cols].to_dict('records')

			if verbose:
				print('length train_adjust: ', len(train_adjust))
				print('length test: ', len(df_test.index))

			""" 
			Modeling
			"""
			allscores = []
			allmodels = {}

			#### 1 original scores and model definition
			bestscore = -11111111111
			start_time = datetime.now()
			for mdl in models_to_train:
				print('starting with: ', mdl)
				mdl_start_time = datetime.now()

				fnc = get_forecaster_class(mdl)
				mdlres = fnc(deepcopy(train_adjust), bucket_size=bucket_size, peak_boost=peak_boost, fi_daymap=deepcopy(daymap))
				train_time = (datetime.now() - mdl_start_time).total_seconds()

				# Forecast
				df_test[mdl] = mdlres.forecast(deepcopy(test))

				if verbose:
					print('lengths of test and forecast set: ', len(df_test['value']), len(df_test[mdl]))
				np.seterr(divide='ignore', invalid='ignore')
				result_scores = WB_Scores(df_test['value'], df_test[mdl])
				forecast_time = (datetime.now() - mdl_start_time).total_seconds() - train_time
				result_scores['train time'], result_scores['validation time'], result_scores['model'] = train_time, forecast_time, mdl
				allscores.append(result_scores)
				allmodels[mdl] = mdlres

				if result_scores[bestmodel_score] > bestscore:
					bestscore = result_scores[bestmodel_score]
					bestmethod = mdl

			if verbose:
				print('best:', bestmethod, 'after', (datetime.now() - start_time).seconds, 'seconds')

			"""
			Saving results
			"""
			if save_output:
				df_all['day'] = df_all['time'].dt.date
				df_all['month'] = df_all['time'].dt.isocalendar().week
				df_all['month'] = df_all['time'].dt.month
				df_all['year'] = df_all['time'].dt.year
				df_all['dayofweek'] = df_all['time'].dt.dayofweek
				df_all['daytime'] = df_all['time'].dt.time
				df_test['day'] = df_test['time'].dt.date
				df_test['time'] = df_test['time'].dt.tz_localize(None)

				# Join model scores with previously logged scores (if present)
				allscores = pd.DataFrame(allscores)
				allscores = pd.merge(allscores, model_output[i]['model_scores'], on=list(allscores.columns), how="outer") \
					if model_output[i]['model_scores'] is not None else allscores
				# Join model predictions on test set with previously logged preds (if present)
				df_test['time'] = df_test['time'].dt.tz_localize(None)
				df_test['attributeId'] = df_test['attributeId'].astype('int64')
				df_test['day'] = df_test['day'].astype('datetime64[ns]')
				df_test = pd.merge(df_test, model_output[i]['model_preds'], on=[c for c in df_test.columns if c in model_output[i]['model_preds'].columns], how="outer") \
					if model_output[i]['model_preds'] is not None else df_test

				default_save_name = o_folder + save_prefix + f"{' lvl1 ' if meta_pipeline else ' '}" + customer + " " + \
									str(attId) + "_" + date.today().strftime('%Y-%m-%d') + ".xlsx"

				writer = pd.ExcelWriter(default_save_name, engine='xlsxwriter')

				# Save performance metrics
				allscores.to_excel(writer, sheet_name="scores")
				# Save training data
				if save_training_data:
					df_train.to_excel(writer, sheet_name="training data")
				# Save independent test data
				df_test.to_excel(writer, sheet_name="train fold 2" if meta_pipeline else "testing data")

				try:
					writer.save()
					print("Saved file:", default_save_name)
				except Exception as e:

					warnings.warn(f"Could not save file: {e}")

			model_output[i]['models'] = allmodels
			model_output[i]['model_preds'] = df_test
			model_output[i]['model_scores'] = allscores
			model_output[i]['train_set'] = train_adjust
			model_output[i]['test_set'] = test_final

	return model_output


def conventional_pipeline(datadict, customer, selected_models, test_size, o_folder):
	# Load previously run conventional models
	conv_output = load_previous_data(customer, datadict, o_folder + "Conventional/", conventional=True)
	start_time = datetime.now()
	# Train models
	conv_output = train_models(datadict, conv_output,
							   s_models=selected_models, o_folder=o_folder + 'Conventional/', test_fold=test_size, save_training_data=True)
	print('Runtime conv. model training:', timedelta(seconds=(datetime.now() - start_time).seconds))

	return conv_output


def meta_pipeline(datadict, customer, selected_models,
				  feat_selection, direction, n_features,
				  meta_models, test_size, o_folder,
				  stationarity):
	# Load previously run level 1 model output
	meta_output = load_previous_data(customer, datadict, o_folder + "Meta/Level1/", conventional=False)

	# Initialise meta model results DF
	meta_results = {}

	# Train level 1 models on first training fold
	start_time = datetime.now()
	if stationarity:
		for k, v in datadict.items():
			datadict[k]['df_all'] = difference(v['df_all'], 'value', 24)
	meta_output = train_models(datadict, meta_output,
							   meta_pipeline=True, s_models=selected_models, o_folder=o_folder + 'Meta/Level1/', test_fold=test_size, save_training_data=True)
	print('Runtime level 1 training:', timedelta(seconds=(datetime.now() - start_time).seconds))

	# Validate level 1 models on independent test set
	for run in meta_output:
		print(f"> DATASET UNIT '{run['data_info']['unit']}'")
		# Check if results are already present
		if check_finished_meta(unit=str(run['data_info']['unit']), o_path=o_folder + 'Meta'):
			print(f"Meta models have already been trained for unit {run['data_info']['unit']}. Continuing...")
			# continue

		mdl_preds = run['model_preds']
		test_df = pd.DataFrame(run['test_set'])
		if run['models'] is not None:  # if not all models had been trained before
			print("\n\nLEVEL 1 MODELS TEST SET FORECASTING")
			models_lvl1 = run['models']

			for i, mdl in enumerate(models_lvl1):
				start_time = datetime.now()
				test_df[mdl] = models_lvl1[mdl].forecast(run['test_set'])
				run['model_scores'].loc[i, 'validation time'] += (datetime.now() - start_time).total_seconds()

			# Save level 1 test set forecasts
			o_path = f'{o_folder}Meta/Level1/Performance lvl1 {run["data_info"]["customer"]} {run["data_info"]["unit"]}_{datetime.now().strftime("%Y-%m-%d")}.xlsx'
			excel_book = load_workbook(o_path)
			with pd.ExcelWriter(o_path, engine='openpyxl') as writer:
				writer.book = excel_book
				writer.sheets = {
					worksheet.title: worksheet
					for worksheet in excel_book.worksheets
				}
				test_df.to_excel(writer, sheet_name="testing data")
				writer.save()
				print("Saved file:", o_path)
				writer.close()
			run['test_set'] = test_df

		# Standardise level 1 train fold 2 forecasts
		run['model_preds'][selected_models] = scale(run['model_preds'][selected_models])

		# Standardise level 1 test set forecasts
		run['test_set'][selected_models] = scale(run['test_set'][selected_models])

		# Forward / Backward feature selection
		feat_sel_time = 0
		if feat_selection:
			print(f"Starting {direction} feature selection...")
			start_time = datetime.now()
			sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features, direction=direction, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
			sfs.fit(mdl_preds[selected_models], mdl_preds['value'])
			selected_features = [col for mask, col in zip(sfs.get_support(), selected_models) if mask]
			feat_sel_time = (datetime.now() - start_time).total_seconds()
			print(f"Selected features ({len(selected_models)}):\n{selected_models}")
		else:
			selected_features = selected_models

		# Train meta models
		meta_mdl_scores = pd.DataFrame(columns=scoring_metrics() + ['model'])
		for i, meta_model in enumerate(meta_models):
			model_obj = meta_models[meta_model][0]
			model_hyperparams = meta_models[meta_model][1]
			print(f"\nTRAINING META MODEL '{meta_model}' ({i + 1}/{len(meta_models)})")
			start_time = datetime.now()
			# Initialise model
			meta_mdl = Model(model_obj, model_hyperparams)
			# Train model (automatically implements 5 fold GridSearchCV when required)
			meta_mdl.train(x_train=mdl_preds[selected_features],
						   y_train=mdl_preds['value'])
			train_time = (datetime.now() - start_time).total_seconds() + feat_sel_time

			# Run model on train set
			meta_pred = meta_mdl.test(mdl_preds[selected_features])
			run['model_preds'][f'Meta Forecaster ({meta_model})'] = list(meta_pred[0])

			"""
			VALIDATE META MODEL
			"""
			print(f"VALIDATING META MODEL '{meta_model}' ({i + 1}/{len(meta_models)})")
			start_time = datetime.now()
			meta_pred = meta_mdl.test(test_df[selected_features])
			y_pred = list(meta_pred[0])
			y_true = list(test_df['value'])
			mdl_score = WB_Scores(y_true, y_pred)
			test_time = (datetime.now() - start_time).total_seconds()
			run['test_set'][f'Meta Forecaster ({meta_model})'] = y_pred

			mdl_score['model'] = f'Meta Forecaster ({meta_model})'
			meta_mdl_scores = meta_mdl_scores.append(mdl_score, ignore_index=True)
			meta_mdl_scores.loc[i, 'train time'], meta_mdl_scores.loc[i, 'validation time'] = train_time, test_time

		# Add model scores to previously ran scores
		run['model_scores'] = pd.concat([run['model_scores'], meta_mdl_scores], ignore_index=True)

		meta_results[run['data_info']['unit']] = run['model_scores']

		# Save meta test set forecasts and scores
		print("Saving results...")
		o_path = f'{o_folder}Meta/Meta/Performance Meta {run["data_info"]["unit"]}_{datetime.now().strftime("%Y-%m-%d-%H%M")}.xlsx'
		with pd.ExcelWriter(o_path, engine='openpyxl') as writer:
			run['model_scores'].to_excel(writer, sheet_name="scores")
			run['model_preds'].to_excel(writer, sheet_name="training data")
			run['test_set'].to_excel(writer, sheet_name="testing data")
			writer.save()
			print("Saved file:", o_path)
			writer.close()

	"""
	# Save performance data
	o_path = f'{o_folder}Meta/Meta/Performance Meta_{datetime.now().strftime("%Y-%m-%d-%H%M")}.json'
	with open(o_path, 'w') as outfile:
		outfile.write(json.dumps([meta_results[k].to_dict() for k in meta_results]))
	print("Saved file:", o_path)
	"""

	return meta_output
