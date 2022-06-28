import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots
# from Packages.ModelDev import RMSE, SMAPE
from datetime import datetime, date as dte
from os import listdir
from os.path import splitext
from dateutil.relativedelta import relativedelta
from scipy.stats import ttest_rel


def _savedf(o_path, df, **kwargs):
	# Print or save data
	if o_path is None:
		print(df)
	else:
		# Convert dictionaries in each cell to strings: mean (sd)
		o_data = stringify_dicts(df.copy(deep=True))
		if o_path.endswith(".csv"):
			o_data.to_csv(o_path, **kwargs)
		elif o_path.endswith(".xlsx"):
			o_data.to_excel(o_path, engine='xlsxwriter', **kwargs)
		print("Saved file:", o_path)


def _savefig(o_path, fig, **kwargs):
	if o_path is None:
		print("Plotting offline...")
		offline.plot(fig)
	elif o_path.endswith(".html"):
		fig.write_html(o_path, **kwargs)
		print("Saved file:", o_path)
	elif o_path.endswith((".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf")):
		fig.write_image(o_path, **kwargs)
		print("Saved file:", o_path)
	else:
		raise ValueError(f'File extension "{splitext(o_path)}" not supported. Supported extensions:\n[".html", ".png", ".jpg", ".webp", ".svg", ".pdf"]')


def _colour_palette():
	brand_colours = {'Happy Pink': '#f2d4d7', 'Petrol Green': '#004851'}
	accent_colours = {'Teal': '#3591A0', 'Aqua': '#BADDDA', 'Berry': '#B23D59', 'Rouge': '#D0858B'}
	extended_palette = {'Warm Grey': '#f0e6e1', 'Deep Berry': '#ba2a50', 'Poppy': '#ff8533', 'Heritage Purple': '#c3c5f1',
						'Deep Rouge': '#ef736f', 'Springy': '#cddcd2', 'Sunny': '#fff2a2', 'Earthy': '#c59c79'}
	return {'brand': brand_colours, 'accent': accent_colours, 'extended': extended_palette}


def _load_data(f_path, prefix, filter_units=None, unit_len=5, sheet_name='scores'):
	file_dict = {}
	filter_units = [] if filter_units is None else filter_units
	f_path = f_path + "/" if not f_path.endswith("/") else f_path
	for fname in listdir(f_path):
		if not fname.endswith(".xlsx") or not fname.startswith(prefix):  # or not any(x in fname for x in ['12679', '13371', '13407']):
			continue
		unit = fname[len(prefix):len(prefix) + unit_len]
		if len(filter_units) == 0 or unit in filter_units:
			file_dict[unit] = pd.read_excel(f_path + fname, sheet_name=sheet_name, index_col=0, engine='openpyxl')
	return file_dict


def _aggregated_scores(folder, prefix, unit_len=5, sheet_test='testing data', sheet_scores='scores',
					   window='24H', on='time', y='value'):
	"""
	Generates scores at a granular level (e.g. for windows of 24 hours)
	:param folder: folder path where scores and predictions can be found
	:param prefix: name prefix untill unit code
	:param unit_len: length of unit code
	:param sheet_test: name of sheet holding predictions
	:param sheet_scores: name of sheet holding scores
	:param window: pd offset window for aggregation
	:param on: column name of timestamps to aggregate on
	:return: dictionary of pd DF's
	"""
	# Load predictions
	preds_dict = _load_data(folder, prefix, unit_len=unit_len, sheet_name=sheet_test)
	# Load scores
	scores_dict = _load_data(folder, prefix, unit_len=unit_len, sheet_name=sheet_scores)
	# Aggregate
	preds_dict = _aggregate(preds_dict, window, on)
	# Calculate RMSE, SMAPE, train time, validation time
	for unit, pred_data in preds_dict.items():
		for row in scores_dict[unit]['model'].index:
			model = scores_dict[unit].at[row, 'model']
			rmse = RMSE(pred_data[y], pred_data[model])
			smape = SMAPE(pred_data[y], pred_data[model])
			scores_dict[unit].at[row, 'RMSE'] = rmse
			scores_dict[unit].at[row, 'SMAPE'] = smape
	return scores_dict


def _aggregate(data_dict, window, on='time'):
	for unit, dataset in data_dict.items():
		dataset = dataset.resample(window, on=on).mean().reset_index(drop=True)
		data_dict[unit] = dataset
	return data_dict


def _link_data(conv_dict, lvl1_dict, meta_dict):
	linked_dict = {}
	for conv_unit, conv_data in conv_dict.items():
		linked_dict[conv_unit] = {'conv': conv_data, 'lvl1': lvl1_dict[conv_unit], 'meta': meta_dict[conv_unit]}
	return linked_dict


def _relative_performance(data_dict):
	units = list(data_dict.keys())
	models = ['Reference Model'] + [i for i in data_dict[units[0]]['meta']['model'].values if i.startswith("Meta Forecaster")]
	o_df = pd.DataFrame(index=['SMAPE', 'RMSE', 'Execution time', 'RMSE difference', 'Execution time difference'], columns=models)

	# Get scores from best reference model for each dataset
	o_df.at['SMAPE', 'Reference Model'] = {unit: scores['conv'][scores['conv']['RMSE'] == scores['conv']['RMSE'].min()]['SMAPE'].values[0]
										   for unit, scores in data_dict.items()}
	o_df.at['RMSE', 'Reference Model'] = {unit: scores['conv']['RMSE'].min()
										   for unit, scores in data_dict.items()}
	o_df.at['Execution time', 'Reference Model'] = {unit: (scores['conv']['train time'] + scores['conv']['validation time']).max()
													for unit, scores in data_dict.items()}
	o_df.at['RMSE difference', 'Reference Model'] = {unit: 100.0 for unit, scores in data_dict.items()}
	o_df.at['Execution time difference', 'Reference Model'] = {unit: 100.0 for unit, scores in data_dict.items()}

	# Get scores from all meta models for each dataset
	for meta_model in models[1:]:
		o_df.at['SMAPE', meta_model] = {unit: scores['meta'][scores['meta']['model'] == meta_model]['SMAPE'].values[0]
										for unit, scores in data_dict.items()}
		o_df.at['RMSE', meta_model] = {unit: scores['meta'][scores['meta']['model'] == meta_model]['RMSE'].values[0]
										for unit, scores in data_dict.items()}
		o_df.at['Execution time', meta_model] = {unit: scores['meta'][scores['meta']['model'] == meta_model]['train time'].values[0] +
													   scores['meta'][scores['meta']['model'] == meta_model]['validation time'].values[0] +
													   (scores['lvl1']['train time'] + scores['lvl1']['validation time']).max()
												 for unit, scores in data_dict.items()}
		o_df.at['RMSE difference', meta_model] = {unit: scores['conv']['RMSE'].min() /
														scores['meta'][scores['meta']['model'] == meta_model]['RMSE'].values[0] * 100
												  for unit, scores in data_dict.items()}
		o_df.at['Execution time difference', meta_model] = {unit: (max(o_df.at['Execution time', 'Reference Model'][unit], o_df.at['Execution time', meta_model][unit]) /
																   min(o_df.at['Execution time', 'Reference Model'][unit], o_df.at['Execution time', meta_model][unit]) - 1)
																   * (100 if o_df.at['Execution time', 'Reference Model'][unit] < o_df.at['Execution time', meta_model][unit] else -100)
															for unit, scores in data_dict.items()}
		o_df.at['Execution time difference', meta_model] = {unit: o_df.at['Execution time', meta_model][unit] /
																  o_df.at['Execution time', 'Reference Model'][unit] * 100
															for unit, scores in data_dict.items()}

	return o_df


def _meanify(data):
	"""
	Calculates mean and SD for each cell
	Replaces cell value with dict with "mean" and "sd" keys
	"""
	data = data.copy(deep=True)
	for col in data.columns:
		for row in data.index:
			if type(data.at[row, col]) == dict:
				values = list(data.at[row, col].values())
			elif type(data.at[row, col]) == list:
				values = data.at[row, col]
			else:
				raise TypeError(f"Data type {type(data.at[row, col])} not supported. Only type 'data' and type 'list' can be passed.")
			cell_mean = np.mean(values).round(1)
			cell_sd = np.std(values).round(1)
			data.at[row, col] = {'mean': cell_mean, 'sd': cell_sd}
	return data


def _cpi(data):
	data = data.copy(deep=True).T
	data['CPI'] = 1.0
	for row in data.index:
		if row == 'Reference Model':
			continue
		else:
			data.at[row, 'CPI'] = round(data.at[row, 'RMSE difference']['mean'] / data.at[row, 'Execution time difference']['mean'], 1)
	return data.T


def _todate(date):
	if type(date) == np.datetime64:
		return pd.to_datetime(date).date()
	elif isinstance(date, dte):
		return date
	else:
		return date.date()


def _filter_best_model(scores, preds):
	for unit, pred_data in preds.items():
		score_data = scores[unit]
		best_model = score_data[score_data['RMSE'] == score_data['RMSE'].min()]['model'].values[0]
		preds[unit] = pred_data[best_model]
	return preds


def load_predictions(conv_folder, meta_folder,
					 conv_prefix="Performance Swarovski North America ", meta_prefix="Performance Meta ",
					 unit_len=5, sheet_test='testing data', sheet_scores='scores',
					 filter_units=None, load_lvl1=False):
	"""
	:return: Dictionary of unit: pd DF where the DF holds meta predictions and a single column of (best) conv predictions
	"""
	# Load data
	print("loading conventional model predictions...")
	conv_preds = _load_data(conv_folder, conv_prefix, unit_len=unit_len, sheet_name='testing data', filter_units=filter_units)
	print("loading conventional model scores...")
	conv_scores = _load_data(conv_folder, conv_prefix, unit_len=unit_len, sheet_name=sheet_scores, filter_units=filter_units)
	print("loading meta model predictions...")
	meta_preds = _load_data(meta_folder, meta_prefix, unit_len=unit_len, sheet_name=sheet_test, filter_units=filter_units)
	# Filter out only best conv model in every dataset
	conv_preds = _filter_best_model(conv_scores, conv_preds)
	# Combine conv preds to meta preds
	filter_cols = ('time', 'value', 'Meta Forecaster') if not load_lvl1 else \
		tuple([i for i in meta_preds[list(meta_preds.keys())[0]].columns if not i.startswith(('attributeId', 'seconds', 'bucketSize', 'day', 'month', 'year', 'dayofweek', 'daytime'))])
	for unit, data in meta_preds.items():
		data = data[[i for i in data.columns if i.startswith(filter_cols)]]
		data.loc[:, f"Reference Model ({conv_preds[unit].name})"] = conv_preds[unit].values
		meta_preds[unit] = data

	if filter_units is None:
		return meta_preds
	else:
		if len(filter_units) == 1:
			return meta_preds[filter_units[0]]
		return {k: v for k, v in meta_preds.items() if k in filter_units}


def load_lvl1_predictions(lvl1_folder, lvl1_prefix="Performance lvl1 Swarovski North America ",
						  unit_len=5, sheet_test='testing data',
						  filter_units=None):
	"""
	:return: Dictionary of unit: pd DF where the DF holds meta predictions and a single column of (best) conv predictions
	"""
	# Load data
	print("loading base model predictions...")
	meta_preds = _load_data(lvl1_folder, lvl1_prefix, unit_len=unit_len, sheet_name=sheet_test, filter_units=filter_units)
	# Combine conv preds to meta preds
	filter_cols = tuple([i for i in meta_preds[list(meta_preds.keys())[0]].columns if not i.startswith(
		('attributeId', 'seconds', 'bucketSize', 'day', 'month', 'year', 'dayofweek', 'daytime'))])
	# Filter out columns
	for unit, data in meta_preds.items():
		data = data[[i for i in data.columns if i.startswith(filter_cols)]]
		meta_preds[unit] = data

	return meta_preds if filter_units is None else meta_preds[filter_units]


def stringify_dicts(data):
	"""Takes a pd DF and for each cell where there is a dict, it unpacks it into a string with mean (sd)"""
	data = data.copy(deep=True)
	for col in data.columns:
		for row in data.index:
			if type(data.at[row, col]) == dict:
				data.at[row, col] = f"{round(data.at[row, col]['mean'], 2)} ({round(data.at[row, col]['sd'], 2)})"
	return data


def create_data_dict(conv_folder, lvl1_folder, meta_folder,
					 conv_prefix, lvl1_prefix, meta_prefix,
					 unit_len, sheet_scores, sheet_test,
					 base_forecasters, window=None):
	"""Creates a data_dict of absolute values {unit: data}"""
	# Load data
	if window is None:
		conv_scores = _load_data(conv_folder, conv_prefix, unit_len=unit_len, sheet_name=sheet_scores)
		meta_scores = _load_data(meta_folder, meta_prefix, unit_len=unit_len, sheet_name=sheet_scores)
	else:
		conv_scores = _aggregated_scores(conv_folder, conv_prefix, unit_len=unit_len, sheet_test=sheet_test, sheet_scores=sheet_scores, window=window)
		meta_scores = _aggregated_scores(meta_folder, meta_prefix, unit_len=unit_len, sheet_test=sheet_test, sheet_scores=sheet_scores, window=window)
	lvl1_scores = _load_data(lvl1_folder, lvl1_prefix, unit_len=unit_len, sheet_name=sheet_scores)

	# Filter out base forecasters
	for unit, scores in lvl1_scores.items():
		lvl1_scores[unit] = scores[scores['model'].isin(base_forecasters)]

	# Link conv to meta data
	return _link_data(conv_scores, lvl1_scores, meta_scores)


class Tablefy:
	@staticmethod
	def monthly_performance(data_dict, y_col='value', date_col='time', o_path=None):
		"""
		Generates a table with performances for each month in the data (also when month is not fully available)
		Expects a dictionary of unit:DataFrame for every dataset available, where the DF contains both reference and meta model predictions
		:param data_dict: dict of unit: pd DataFrame
		:return: pd DataFrame
		"""
		months = None
		conv_model, meta_models = None, None
		table = None
		for unit, data in data_dict.items():
			data[date_col] = [_todate(i) for i in data[date_col].values]
			if table is None:
				# Find months in data
				firstdate = data.iloc[0, data.columns.get_loc(date_col)]
				lastdate = data.iloc[-1, data.columns.get_loc(date_col)]
				months = pd.date_range(firstdate, lastdate, freq='MS').strftime("%b %Y").tolist()
				# Find meta model names
				meta_models = [i for i in data.columns if i.startswith("Meta")]
				# Initialise output table with double column headers
				table = pd.DataFrame(columns=[list(np.repeat(months, 2)), list(np.tile(['SMAPE', 'RMSE'], len(months)))], index=['Reference Model'] + meta_models)
				for col in table.columns:
					for row in table.index:
						table.at[row, col] = []

			for i, month in enumerate(months, start=1):
				start_month = datetime.strptime(month, "%b %Y").date()
				next_month = datetime(2099, 1, 1).date() if i == len(months) else datetime.strptime(months[i], "%b %Y").date()
				actual_values = data[(data[date_col] >= start_month) & (data[date_col] < next_month)][y_col]

				# Calculate values for conventional model
				conv_model = [i for i in data.columns if not i.startswith(("Meta", y_col, date_col))][0]
				pred_values = data[(data[date_col] >= start_month) & (data[date_col] < next_month)][conv_model]
				table.at['Reference Model', (month, 'RMSE')].append(100.0)
				table.at['Reference Model', (month, 'SMAPE')].append(round(SMAPE(actual_values, pred_values), 1))
				ref_RMSE = RMSE(actual_values, pred_values)

				for meta_model in meta_models:
					# Calculate RMSE for window
					pred_values = data[(data[date_col] >= start_month) & (data[date_col] < next_month)][meta_model]
					# RMSE relative to that of reference model
					table.at[meta_model, (month, 'RMSE')].append(round(ref_RMSE / RMSE(actual_values, pred_values) * 100, 1))
					# SMAPE
					table.at[meta_model, (month, 'SMAPE')].append(round(SMAPE(actual_values, pred_values), 1))

		table = _meanify(table)

		# Print or save data
		if o_path is None:
			print(table)
		else:
			# Convert dictionaries in each cell to strings: mean (sd)
			o_data = stringify_dicts(table.copy(deep=True))
			o_data.to_csv(o_path)
			print("Saved file:", o_path)
		return table

	@staticmethod
	def weekday_performance(data_dict, y_col='value', date_col='time', o_path=None):
		"""
		Generates a table with performances for each weekday in the data
		Expects a dictionary of unit:DataFrame for every dataset available, where the DF contains both reference and meta model predictions
		:param data_dict: dict of unit: pd DataFrame
		:return: pd DataFrame
		"""
		weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
		conv_model, meta_models = None, None
		table = None
		for unit, data in data_dict.items():
			data[date_col] = [_todate(i) for i in data[date_col].values]
			if table is None:
				# Find meta model names
				meta_models = [i for i in data.columns if i.startswith("Meta")]
				# Initialise output table with double column headers
				table = pd.DataFrame(columns=[list(np.repeat(weekdays, 2)), list(np.tile(['SMAPE', 'RMSE'], len(weekdays)))], index=['Reference Model'] + meta_models)
				for col in table.columns:
					for row in table.index:
						table.at[row, col] = []

			for weekday_i, weekday in enumerate(weekdays):
				actual_values = [data.at[i, y_col] for i in data.index if data.at[i, date_col].weekday() == weekday_i]

				# Calculate values for conventional model
				conv_model = [i for i in data.columns if not i.startswith(("Meta", y_col, date_col))][0]
				pred_values = [data.at[i, conv_model] for i in data.index if data.at[i, date_col].weekday() == weekday_i]
				table.at['Reference Model', (weekday, 'RMSE')].append(round(RMSE(actual_values, pred_values), 1))
				table.at['Reference Model', (weekday, 'SMAPE')].append(round(SMAPE(actual_values, pred_values), 1))
				ref_RMSE = RMSE(actual_values, pred_values)

				for meta_model in meta_models:
					# Calculate RMSE for window
					pred_values = [data.at[i, meta_model] for i in data.index if data.at[i, date_col].weekday() == weekday_i]
					# RMSE relative to that of reference model
					table.at[meta_model, (weekday, 'RMSE')].append(round(ref_RMSE / RMSE(actual_values, pred_values) * 100, 1))
					# SMAPE
					table.at[meta_model, (weekday, 'SMAPE')].append(round(SMAPE(actual_values, pred_values), 1))

		table = _meanify(table)

		# Save or print data
		_savedf(o_path, table)
		return table

	@staticmethod
	def forecast_horizon(data_dict, horizons, y_col='value', date_col='time', o_path=None):
		"""
		Generates a table with performances for several forecasting windows
		:param data_dict: dict of unit: pd DataFrame
		:param horizons: list of pd offset strings
		:return: pd DataFrame
		"""
		headers = [f"{list(d.values())[0]} {list(d.keys())[0]}" for d in horizons]
		conv_model, meta_models = None, None
		table = None

		for unit, data in data_dict.items():
			data[date_col] = [_todate(i) for i in data[date_col].values]
			if table is None:
				# Find meta model names
				meta_models = [i for i in data.columns if i.startswith("Meta")]
				# Initialise output table with double column headers

				table = pd.DataFrame(columns=[list(np.repeat(headers, 2)), list(np.tile(['SMAPE', 'RMSE'], len(horizons)))],
									 index=['Reference Model'] + meta_models)
				for col in table.columns:
					for row in table.index:
						table.at[row, col] = []

			for i, horizon in enumerate(horizons):
				actual_values = data[data[date_col] < (data.at[0, date_col] + relativedelta(**horizon))][y_col]

				# Calculate values for conventional model
				conv_model = [i for i in data.columns if not i.startswith(("Meta", y_col, date_col))][0]
				pred_values = data[data[date_col] < (data.at[0, date_col] + relativedelta(**horizon))][conv_model]
				table.at['Reference Model', (headers[i], 'RMSE')].append(100.0)
				table.at['Reference Model', (headers[i], 'SMAPE')].append(round(SMAPE(actual_values, pred_values), 1))
				ref_RMSE = RMSE(actual_values, pred_values)

				for meta_model in meta_models:
					# Calculate RMSE for window
					pred_values = data[data[date_col] < (data.at[0, date_col] + relativedelta(**horizon))][meta_model]
					# RMSE relative to that of reference model
					table.at[meta_model, (headers[i], 'RMSE')].append(round(ref_RMSE / RMSE(actual_values, pred_values) * 100, 1))
					# SMAPE
					table.at[meta_model, (headers[i], 'SMAPE')].append(round(SMAPE(actual_values, pred_values), 1))

		table = _meanify(table)

		# Save or print data
		_savedf(o_path, table)

		return table

	@staticmethod
	def mean_performance(conv_folder, lvl1_folder, meta_folder, o_path=None,
						 conv_prefix="Performance Swarovski North America ", lvl1_prefix="Performance lvl1 Swarovski North America ", meta_prefix="Performance Meta ",
						 unit_len=5, sheet_scores='scores', sheet_test='testing data',
						 base_forecasters=None, window=None):
		"""
		Generates a table with mean performance (SMAPE, rel. RMSE, CPI etc.)
		:param conv_folder: path str
		:param lvl1_folder: path str
		:param meta_folder: path str
		:param o_path: path str
		:param conv_prefix: filename prefix until unit code (str)
		:param lvl1_prefix: filename prefix until unit code (str)
		:param meta_prefix: filename prefix until unit code (str)
		:param unit_len: length of unit code (int)
		:param sheet_scores: sheet name of scores data to access (str)
		:param sheet_scores: sheet name of scores data to access (str)
		:param sheet_test: sheet name of test set predictions data to access (str)
		:param base_forecasters: if not None, then this filters out specified base models out of the level1 model scores. List of strings.
		:param window: for aggregating predictions in a time window
		:return: pd DF
		"""

		# Load and link data
		data_dict = create_data_dict(conv_folder, lvl1_folder, meta_folder,
									 conv_prefix, lvl1_prefix, meta_prefix,
									 unit_len, sheet_scores, sheet_test,
									 base_forecasters, window)
		# Clear memory
		conv_scores, lvl1_scores, meta_scores = None, None, None
		# Calculate all absolute and relative values and put them in a DF
		data_dict = _relative_performance(data_dict)
		for meta_model in data_dict.columns[1:]:
			ttest = ttest_rel(list(data_dict.at['RMSE', meta_model].values()), list(data_dict.at['RMSE', 'Reference Model'].values()))
			print(meta_model, ttest)
		data_dict.drop(labels=['RMSE'], axis=0, inplace=True)
		print(data_dict)
		exit()
		# Calculate averages
		mean_data_dict = _meanify(data_dict)
		# Calculate CPI's
		cpi_data = _cpi(mean_data_dict)
		mean_data_dict = None
		cpi_data.columns = [i[len("Meta Forecaster ("):-1] if i.startswith("Meta") else i for i in cpi_data.columns]

		# Save or print data
		_savedf(o_path, cpi_data.T)

		return cpi_data


class Visualise:
	@staticmethod
	def predictions_line(data, y_hats, y='value',
						 window='24H', show_minmax=False, leading_hours=240, dashify=True,
						 o_path=None, **kwargs):
		"""
		Plots a line graph with actual data and predictions
		:param data: pandas DataFrame
		:param y: string, actual values
		:param y_hats: list of strings, columns with predictions
		:param leading_hours: number of leading hours to show before predictions are available
		:param dashify: plots dashed lines for lines other than meta models
		:param o_path: path to save file to
		:param window: window for grouping values https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
		:param show_minmax: show faint min and max predictions next to grouped means. Only used when window is not None
		"""

		if leading_hours > 0:
			data = data.iloc[data[pd.isna(data[y_hats[0]])].index[-leading_hours]:, :]

		# grouped means
		if window is not None:
			if show_minmax:
				data_min = data.resample(window, on='time', origin='start').min().reset_index(drop=True)
				data_max = data.resample(window, on='time', origin='start').max().reset_index(drop=True)
				# Some weird bug causes the data_max time col to start and end 23 hours later.. Insert hacky fix here
				data_max['time'] = data_min['time']
			data = data.resample(window, on='time', origin='start').mean().reset_index()

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['time'], y=data[y], name='Target', marker=dict(color=_colour_palette()['accent']['Teal'])))
		if show_minmax:
			fig.add_trace(go.Scatter(x=data_min['time'], y=data_min[y], name='Target Min.', marker=dict(color=_colour_palette()['accent']['Teal'])))
			fig.add_trace(go.Scatter(x=data_max['time'], y=data_max[y], name='Target Max.', marker=dict(color=_colour_palette()['accent']['Teal'])))
		for mdl in y_hats:
			if mdl.startswith('Meta Forecaster'):
				fig.add_trace(go.Scatter(x=data['time'], y=data[mdl], name=mdl, marker=dict(color=_colour_palette()['accent']['Berry'])))
				if show_minmax:
					fig.add_trace(go.Scatter(x=data_min['time'], y=data_min[mdl], name=mdl + " Min.", opacity=0.5, marker=dict(color=_colour_palette()['accent']['Berry'])))
					fig.add_trace(go.Scatter(x=data_max['time'], y=data_max[mdl], name=mdl + " Max.", opacity=0.5, marker=dict(color=_colour_palette()['accent']['Berry'])))
			else:
				fig.add_trace(go.Scatter(x=data['time'], y=data[mdl], name=mdl, opacity=.33,
										 line=dict(dash='dash' if dashify else 'solid'), marker=dict(color=_colour_palette()['extended']['Deep Rouge'])))

		# Update layout with default parameters
		fig.update_layout(
			xaxis_title="Time",
			yaxis_title="Store traffic",
			template='ggplot2',
			plot_bgcolor=_colour_palette()['brand']['Happy Pink'],
			autosize=False,
			width=1400,
			height=600,
			font={'size': 16, 'family': 'Poppins'},
		)

		# Update layout with kwargs
		fig.update_layout(**kwargs)

		# Save figure or show temporarily in browser
		_savefig(o_path, fig)

	@staticmethod
	def predictions_line_zoom(data, y_hats, y='value',
							  windows=('24H', '1H'), show_minmax=False, leading_hours=240, dashify=True,
							  o_path=None, **kwargs):
		"""
		Plots a line graph with actual data and predictions
		:param data: pandas DataFrame
		:param y: string, actual values
		:param y_hats: list of strings, columns with predictions
		:param leading_hours: number of leading hours to show before predictions are available
		:param dashify: plots dashed lines for lines other than meta models
		:param o_path: path to save file to
		:param windows: windows for grouping values https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
		:param show_minmax: show faint min and max predictions next to grouped means. Only used when window is not None
		:param **kwargs: used in fig.update_layout()
		"""

		if leading_hours > 0:
			data = data.iloc[data[pd.isna(data[y_hats[0]])].index[-leading_hours]:, :]

		fig = make_subplots(1, 2, column_widths=[3/5, 2/5])

		for col, window in enumerate(windows, start=1):
			# grouped means
			if show_minmax:
				data_min = data.resample(window, on='time', origin='start').min().reset_index(drop=True)
				data_max = data.resample(window, on='time', origin='start').max().reset_index(drop=True)
				# Some weird bug causes the data_max time col to start and end 23 hours later.. Insert hacky fix here
				data_max['time'] = data_min['time']
			data_aggr = data.resample(window, on='time', origin='start').mean().reset_index()

			fig.add_trace(go.Scatter(x=data_aggr['time'], y=data_aggr[y], name='Target', marker=dict(color=_colour_palette()['accent']['Teal'])), row=1, col=col)
			if show_minmax:
				fig.add_trace(go.Scatter(x=data_min['time'], y=data_min[y], name='Target Min.', marker=dict(color=_colour_palette()['accent']['Teal'])), row=1, col=col)
				fig.add_trace(go.Scatter(x=data_max['time'], y=data_max[y], name='Target Max.', marker=dict(color=_colour_palette()['accent']['Teal'])), row=1, col=col)
			for mdl in y_hats:
				if mdl.startswith('Meta Forecaster'):
					fig.add_trace(go.Scatter(x=data_aggr['time'], y=data_aggr[mdl], name=mdl, marker=dict(color=_colour_palette()['accent']['Berry'])), row=1, col=col)
					if show_minmax:
						fig.add_trace(go.Scatter(x=data_min['time'], y=data_min[mdl], name=mdl + " Min.", opacity=0.5, marker=dict(color=_colour_palette()['accent']['Berry'])),
									  col=col)
						fig.add_trace(go.Scatter(x=data_max['time'], y=data_max[mdl], name=mdl + " Max.", opacity=0.5, marker=dict(color=_colour_palette()['accent']['Berry'])),
									  col=col)
				else:
					fig.add_trace(go.Scatter(x=data_aggr['time'], y=data_aggr[mdl], name=mdl,  # opacity=.33,
											 line=dict(dash='dash' if dashify else 'solid'), marker=dict(color=_colour_palette()['extended']['Deep Rouge'])), row=1, col=col)

		# Update layout with default parameters
		fig.update_layout(
			xaxis_title="Time",
			yaxis_title="Store traffic",
			template='ggplot2',
			plot_bgcolor=_colour_palette()['brand']['Happy Pink'],
			autosize=False,
			width=1400,
			height=600,
			font={'size': 16, 'family': 'Poppins'},
		)

		# Update layout with kwargs
		fig.update_layout(**kwargs)

		# Save figure or show temporarily in browser
		_savefig(o_path, fig)

	@staticmethod
	def cost_performance_scatter(data, o_path=None):
		"""
		Plots a scatter plot with execution time relative to reference on X, and performance relative to reference on Y
		:param data: pandas DF
		:param o_path: path to save file to
		"""
		for c in ['RMSE difference', 'Execution time difference', 'CPI']:
			if c not in data.columns:
				raise KeyError(f"Expected column {c} in DataFrame, but none was found")

		data = data[~pd.isna(data['CPI'])]  # Filter out reference model
		data['size'] = 1  # Dummy column used for sizing the points
		data.index.name = "Model"
		data.reset_index(inplace=True, drop=False)
		data['RMSE difference'] = [data.at[i, 'RMSE difference']['mean'] for i in data.index]
		data['Execution time difference'] = [data.at[i, 'Execution time difference']['mean'] for i in data['Execution time difference'].index]
		data.sort_values(by='RMSE difference', inplace=True, ascending=False)
		fig = px.scatter(data, x='Execution time difference', y='RMSE difference', color='Model', size='size', size_max=20, log_x=True)
		fig.update_traces(textposition='top center')

		fig.update_layout(
			xaxis_title="Execution time relative to reference model (%)",
			yaxis_title="Mean RMSE improvement<br>relative to reference model (%)",
			template='ggplot2',
			plot_bgcolor=_colour_palette()['brand']['Happy Pink'],
			autosize=False,
			showlegend=True,
			width=1500,
			height=600,
			font={'size': 24, 'family': 'Poppins'}
		)

		# Save figure or show temporarily in browser
		_savefig(o_path, fig)

	@staticmethod
	def cost_performance_scatter_splitaxis(data, o_path=None):
		"""
		Plots a scatter plot with execution time relative to reference on X, and performance relative to reference on Y
		:param data: pandas DF
		:param o_path: path to save file to
		"""
		for c in ['RMSE difference', 'Execution time difference', 'CPI']:
			if c not in data.columns:
				raise KeyError(f"Expected column {c} in DataFrame, but none was found")

		data = data[~pd.isna(data['CPI'])]  # Filter out reference model
		data['size'] = 1  # Dummy column used for sizing the points
		data.index.name = "Model"
		data.reset_index(inplace=True, drop=False)
		data['RMSE difference'] = [data.at[i, 'RMSE difference']['mean'] for i in data.index]
		data['Execution time difference'] = [data.at[i, 'Execution time difference']['mean'] for i in data['Execution time difference'].index]
		data.sort_values(by='RMSE difference', inplace=True, ascending=False)

		fig = make_subplots(1, 2, column_widths=[5/6, 1/6], shared_yaxes=True)
		subdata = data[data['Execution time difference'] < data['Execution time difference'].max()]

		for model in subdata.Model.values:
			model_data = subdata[subdata['Model'] == model]
			fig.add_trace(go.Scatter(x=model_data['Execution time difference'], y=model_data['RMSE difference'], mode='markers', name=model,
									 marker=dict(size=30, opacity=0.7, line=(dict(color='white', width=2)))), row=1, col=1)
		subdata = data[data['Execution time difference'] == data['Execution time difference'].max()]
		fig.add_trace(go.Scatter(x=subdata['Execution time difference'], y=subdata['RMSE difference'], mode='markers', name=subdata['Model'].values[0],
								 marker=dict(size=30, opacity=0.7, color=subdata.index, line=(dict(color='white', width=2)))), row=1, col=2)

		fig.update_xaxes(range=[4800, 5000], row=1, col=2, tick0=4800, dtick=100)
		fig.update_layout(
			xaxis_title="Execution time relative to reference model (%)",
			yaxis_title="Mean RMSE improvement<br>relative to reference model (%)",
			template='ggplot2',
			plot_bgcolor=_colour_palette()['brand']['Happy Pink'],
			autosize=False,
			showlegend=True,
			width=1500,
			height=600,
			font={'size': 24, 'family': 'Poppins'}
		)

		# Save figure or show temporarily in browser
		_savefig(o_path, fig)

	@staticmethod
	def cost_performance_bar(conv_folder, lvl1_folder, meta_folder, best_model,
							 conv_prefix="Performance Swarovski North America ", lvl1_prefix="Performance lvl1 Swarovski North America ", meta_prefix="Performance Meta ",
							 unit_len=5, sheet_scores='scores', sheet_test='testing data',
							 base_forecasters=None, window='1H',
							 o_path=None):
		"""
		Plots both a bar and scatter plot with:
			each run on X
			execution time relative to reference on Y (left)
			performance relative to reference on Y (right)
		:param data: pandas DF
		:param o_path: path to save file to
		"""

		# Load and link data
		data_dict = create_data_dict(conv_folder, lvl1_folder, meta_folder,
									 conv_prefix, lvl1_prefix, meta_prefix,
									 unit_len, sheet_scores, sheet_test,
									 base_forecasters=base_forecasters, window=window)
		# Calculate all absolute and relative values and put them in a DF
		data_dict = _relative_performance(data_dict)

		# Make dataframe from original data
		runtime_col, perf_col = [], []
		best_meta_model = {}

		for unit, execution_time in data_dict.at['Execution time', best_model].items():
			runtime_col.append(execution_time)
			perf_col.append(data_dict.at['RMSE difference', best_model][unit])
			best_meta_model[unit] = (perf_col[-1], )

		data_df = pd.DataFrame({'runtime': runtime_col, 'performance': perf_col})
		data_df.sort_values('runtime', inplace=True, ignore_index=True)

		# Create figure with secondary y-axis
		fig = make_subplots(specs=[[{"secondary_y": True}]])
		fig.add_trace(go.Bar(x=[i for i in data_df.index], y=data_df['runtime'], name='Relative execution time',
							 marker=dict(color=_colour_palette()['brand']['Petrol Green'])),
					  secondary_y=False)
		fig.add_trace(go.Scatter(x=[i for i in data_df.index], y=data_df['performance'], mode='markers', name='Relative RMSE',
								 marker=dict(size=15, color=_colour_palette()['accent']['Rouge'])),
					  secondary_y=True)

		# Configure visuals
		fig['layout']['yaxis2']['showgrid'] = False
		fig.update_layout(
			template='ggplot2',
			plot_bgcolor=_colour_palette()['brand']['Happy Pink'],
			autosize=False,
			width=1400,
			height=600,
			font={'size': 16, 'family': 'Poppins'},
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=1.02,
				xanchor="right",
				x=1
			)
		)

		# Set y-axes titles
		fig.update_yaxes(title_text="Mean RMSE relative to reference model (%)", range=[0, 100], secondary_y=True)
		fig.update_yaxes(title_text="Execution time relative to reference model (%)", secondary_y=False)
		fig.update_xaxes(visible=False)

		# Save figure or show temporarily in browser
		_savefig(o_path, fig)

	@staticmethod
	def rolling_mean(data, y='value', window='24H', o_path=None, **kwargs):
		"""
		Plots a rolling mean and standard deviation
		:param data: pandas DF
		:param y: name of column with actual target values
		:param window: rolling window size
		"""
		data_mean = data.resample(window, on='time', origin='start').mean().reset_index()
		data_sd = data.resample(window, on='time', origin='start').std().reset_index()

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data_mean['time'], y=data_mean[y], name='Mean',
								 marker=dict(color=_colour_palette()['accent']['Teal'])))
		fig.add_trace(go.Scatter(x=data_sd['time'], y=data_sd[y], name='SD',
								 marker=dict(color=_colour_palette()['extended']['Deep Rouge'])))

		# Update layout with default parameters
		fig.update_layout(
			xaxis_title="Time",
			yaxis_title="Store traffic",
			template='ggplot2',
			plot_bgcolor=_colour_palette()['brand']['Happy Pink'],
			autosize=False,
			width=1400,
			height=600,
			font={'size': 16, 'family': 'Poppins'},
		)

		# Update layout with kwargs
		fig.update_layout(**kwargs)

		# Save figure or show temporarily in browser
		_savefig(o_path, fig)