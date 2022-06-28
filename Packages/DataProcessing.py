import random
import pytz
import warnings
import pandas as pd
import numpy as np
from Packages.Results import Visualise
from datetime import datetime, timedelta
from os import listdir
from dateutil.parser import parse
from alfacore import transform
from statsmodels.tsa.stattools import adfuller
pd.options.display.width = 0


def check_stationarity(data, split_by="month", window='24H', plot_rolling=False, fig_path=None):
	"""
	Checks for presence of stationarity in a time series
	:param data: pandas dataframe with single column
	:param split_by: splits dataset to compare summary statistics. "month" to split by month, integer to sep by number of observations

	"""
	stationary = False
	y_col = data.columns[0]
	# Plot series
	if plot_rolling:
		Visualise.rolling_mean(data, window=window, o_path=fig_path)
	# Augmented Dickey-Fuller test
	dftest = adfuller(data[y_col].values, autolag='t-stat')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
	for key, value in dftest[4].items():
		dfoutput['Critical Value (%s)' % key] = value
	if max(dftest[4].values()) > dftest[0]:
		stationary = True
	print(dfoutput)
	print(f">>> Time series is {'' if stationary else 'NOT '}stationary ({max(dftest[4].values())} {'>' if stationary else '<'} {dftest[0]})")


def difference(data, target, lag=24, order=1):
	for _ in range(order):
		diff = data[target].diff()
		data = data.iloc[lag:, :]
		for i in diff.index:
			data.at[i, target] = diff[i]
		data.reset_index(inplace=True, drop=True)
	return data


def normalise(arr, low=1, high=101):
	width = high - low
	return (arr - arr.min()) / (arr.max() - arr.min()) * width + low


def retrieve_data(alfa_session=None, teamId=None, unitId=None, selected_units=None, att_names=None, max_units=1, min_length=0, time_min=None, time_max=None):
	"""
	Retrieves time series data from a specific customer-attribute combination
	When unitId is specified, teamId will be retrieved through an API call with that unitId and the teamId will not be used
	unitId is the external database ID from one of units in customer
	:param alfa_session: API session (from alfa_sdk.common import session)
	:param teamId: (ALFA) teamId specifying customer
	:param unitId: (Pythia) external database unitId used in retrieving the teamId
	:param att_names: list of attribute names to retrieve for each unit
	:param max_units: max number of units to retrieve
	:param min_length: minimum length of each time series
	:return: (str) customer name,
	 		 nested dictionary: {<attribute Id>: {'df_all': <time series data>,
	 		 									  'initial_cols': <list of strings>,
	 		 									  'timezone': <string tz>}
	 		 					}
	"""
	# Checks and warnings
	if teamId is None and unitId is None:
		raise ValueError("No Team ID or Unit ID specified")
	if teamId is not None and unitId is not None:
		warnings.warn("Both Team ID and Unit ID have been passed. Specified Team ID will be ignored")
	if att_names is None:
		att_names = []
		warnings.warn("No Attribute name specified. All attributes will be retrieved")
	if alfa_session is None:
		from alfa_sdk.common import session
		alfa_session = session.Session()

	# Retrieve teamId
	if unitId is not None:
		unit = alfa_session.request(method="GET", service="pythia", path="/api/Units/getUnitByExternalDatabaseId",
									json={"externalDatabaseId": unitId})[0]
		teamId = unit["customer"]["teamId"]

	# Retrieve customer info
	data_dict = {}
	cust = alfa_session.request(method="GET", service="pythia", path="/api/Organisations/getCustomerUnitHierarchy",
									  json={"teamId": teamId})[0]

	# Loop through units
	if selected_units is not None:
		all_units = [unit for unit in cust['units'] for att in unit['attributes'] if att['id'] in selected_units]
	else:
		all_units = [unit for unit in cust['units']]
	for unit in random.sample(all_units, len(all_units)):
		for att in unit['attributes']:
			if att['name'].lower() in [n.lower() for n in att_names]:
				# Retrieve time series
				qs = {
					"attributeId": att['id'],
					"toReturnReference": False
				}

				# Read data
				df_alfa = alfa_session.request(method="POST", service="pythia", path="/api/readings/gettrainingdata", json=qs)

				# Skip if time series is empty
				if len(df_alfa) == 0:
					continue

				bucket_size = df_alfa[0]['bucketSize']

				df_all = pd.DataFrame(preprocess_data(df_alfa, bucket_size, unit['tz']))
				initial_cols = df_all.columns.values

				#### Data transformations
				loadeddata = df_all.to_dict('records')
				loadeddata = transform.to_datetime(loadeddata)

				#### timezone transformation & splitting data
				df_raw = pd.DataFrame(loadeddata)
				df_raw['time'] = df_raw['time'].dt.tz_localize(None)
				for entry in loadeddata:
					entry['time'] = entry['time'].replace(tzinfo=None)
				df_all = pd.DataFrame(loadeddata)
				df_all['time'] = df_all['time'].dt.tz_localize(None)

				#### Time filters
				if time_min:
					df_all = df_all[df_all['time'] > time_min]
				if time_max:
					df_all = df_all[df_all['time'] < time_max]

				# Only add data if at least of min length
				if len(df_all) >= min_length:
					data_dict[att['id']] = {'df_all': df_all, 'initial_cols': initial_cols, 'loadeddata': loadeddata}
				break
		if len(data_dict) == max_units:
			break
	return cust['name'], data_dict


def preprocess_data(data, bucket_size, tz):
	"""
	Transforms the string timestamps into datetime objects, applies the timezone offset to the data,
	removes the timezone, and removes entries with duplicate timestamps.
	"""
	_to_dt(data)
	for entry in data:
		entry["time"] = offset_to_timezone(entry["time"], pytz.timezone(tz))

	data.sort(key=lambda entry_: entry_["time"])
	return _filter_duplicate_date_times(data, bucket_size)


def offset_to_timezone(dt, tz):
	"""Offsets the datetime object to the specified timezone and removes the timzone information"""
	return tz.normalize(pytz.utc.localize(dt.replace(tzinfo=None))).replace(tzinfo=None)


def _to_dt(data):
	"""
	Returns the data same data object, with dt_strings replaced with datetime objects.
	"""
	for entry in data:
		dt = parse(entry["time"])
		if dt.tzinfo is None:
			pytz.utc.localize(dt)
		entry["time"] = dt


def _filter_duplicate_date_times(raw_data, bucket_size):
	"""
	Filters data from duplicate date-times. The first entry is always stored, unless the first
	entry's value equals zero and another is non-zero. The non-zero value is stored in that case.
	"""
	bucket_timedelta = timedelta(milliseconds=bucket_size)
	bucket_size_sec = bucket_size / 1000

	previous_entry = raw_data[0]
	data = raw_data[:1]

	for index in range(1, len(raw_data)):
		this_entry = raw_data[index]
		if previous_entry["time"] + bucket_timedelta == this_entry["time"]:
			data.append(this_entry)
		elif previous_entry["time"] == this_entry["time"]:
			if previous_entry["value"] == 0:
				data[-1] = this_entry
		elif (this_entry["time"] - previous_entry["time"]).total_seconds() % bucket_size_sec != 0:
			print(this_entry, previous_entry)
			raise ValueError(f"Step size is not multiple of bucket-size {bucket_size}")
		elif previous_entry["time"] > this_entry["time"]:
			raise ValueError(f"Data is not sorted.")
		else:
			warnings.warn(
				f"A gap is observed between: {previous_entry['time']} and {this_entry['time']}."
			)
		previous_entry = this_entry

	return data


def handle_duplicate_times(data):
	"""
	Data may contain duplicate timestamps or even complete duplicates.
	These we want to get rid of, returning a dataset with unique timestamps.
	"""
	now = datetime.now()
	seen = {}
	new_list = []
	for entry in data:
		if entry['time'] not in seen:
			# Add time/value combination of unprecedented time
			seen[entry['time']] = entry['value']
			new_list.append(entry)
		else:
			if seen[entry['time']] != entry['value']:
				if not seen[entry['time']] > 0:
					# Delete existing entry if it has 0, add the new entry.
					# It cannot be 0, those have been filtered out already.
					# Implies we always take the first non-zero value.
					new_list = [i for i in new_list if i['time'] != entry['time']]
					new_list.append(entry)
	new_list.sort(key=lambda x: x["time"])
	print(datetime.now() - now)
	return new_list


def load_previous_data(customer, datadict, o_path, conventional):
	model_output = [{'models': None, 'model_preds': None, 'model_scores': None, 'train_set': None, 'test_set': None, 'data_info': {'customer': customer, 'unit': str(unit)}}
					for unit in datadict]

	for n, unit in enumerate(datadict):
		# Loop through folders to find pre-existing model output data
		start_string = f"Performance{' ' if conventional else ' lvl1 '}{customer} {unit}"
		for i in [x for x in listdir(o_path) if x.startswith(start_string)][::-1]:
			xlsx = pd.read_excel(o_path + i, sheet_name=None, index_col=0, engine='openpyxl')
			if 'training data' in xlsx.keys():
				# Check if data has runtimes saved
				if 'train time' in xlsx['scores'].columns:
					if conventional:
						# Load conventional model training data
						model_output[n]['model_preds'] = xlsx['testing data']
						print("Loaded independent test data for conventional models")
					else:
						# Load level 1 training + test data
						model_output[n]['model_preds'] = xlsx['train fold 2']
						print("Loaded train fold 2 data for level 1 models and unit", unit)
						model_output[n]['test_set'] = xlsx['testing data']
						print("Loaded independent test data for level 1 models and unit", unit)
					model_output[n]['model_scores'] = xlsx['scores']
					break
	return model_output


def input_new_data(o_path, selected_units, models=('Meta Forecaster (Lin. SVR)', 'Meta Forecaster (Lin. SVR (hyperparam. tuning))'),
				   old_date='2022-05', new_date='2022-06'):
	""" Replaces data from meta models with new runs"""
	for unit in selected_units:
		for basef in [x for x in listdir(o_path) if x.startswith(f"Performance Meta {unit}_{old_date}")]:
			base_file = pd.read_excel(o_path + basef, sheet_name=['scores', 'training data', 'testing data'], index_col=0, engine='openpyxl')
			base_scores, base_test, base_training = base_file['scores'], base_file['testing data'], base_file['training data']

			# Add SMAPE col
			if 'SMAPE' not in base_scores.columns:
				base_scores.insert(5, 'SMAPE', np.NaN)

			for newf in [x for x in listdir(o_path) if x.startswith(f"Performance Meta {unit}_{new_date}")]:
				new_file = pd.read_excel(o_path + newf, sheet_name=['scores', 'testing data'], index_col=0, engine='openpyxl')
				new_scores, new_test = new_file['scores'], new_file['testing data']

				for row, model in enumerate(base_scores['model'].values):
					if model not in models:
						# Calculate SMAPE for all models
						smape = SMAPE(base_test['value'], base_test[model])
						base_scores.at[row, 'SMAPE'] = smape
					else:
						# Copy values from new Lin. SVR models to old file
						for col in new_scores.columns:
							base_scores.at[row, col] = new_scores[new_scores['model'] == model][col].values[0]

				for model in models:
					for row in new_test[model].index:
						base_test.at[row, model] = new_test.at[row, model]

			writer = pd.ExcelWriter(o_path + basef, engine='openpyxl')
			base_scores.to_excel(writer, 'scores')
			base_test.to_excel(writer, 'training data')
			base_test.to_excel(writer, 'testing data')
			writer.save()
			print("Saved file:", o_path + basef)
