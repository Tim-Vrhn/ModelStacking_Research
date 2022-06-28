from Packages.Results import Tablefy, Visualise, load_predictions, load_lvl1_predictions, stringify_dicts
import pandas as pd

base_models = [
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

lvl1_trainfold2 = False
meta_base_forecasts = False
mean_perf_window = False
mean_perf = True
cpi_scatter = False
cpi_bar = False
horizon = False
monthly = False
weekday = False
meta_forecasts = False
meta_forecasts_zoom = False

# Forecasts of meta model + level 1 base models
if meta_base_forecasts:
	train_df = pd.read_excel("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta/Performance Meta 13227_2022-05-26-1602.xlsx",
								 sheet_name='training data', usecols=['time', 'value'], engine='openpyxl')
	preds = load_predictions("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Conventional",
							 "C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta",
							 filter_units='13227', load_lvl1=True)
	all_df = pd.concat([train_df, preds], ignore_index=True)
	all_df = all_df[[i for i in all_df.columns if not i.startswith("Reference")]]
	Visualise.predictions_line(all_df, [i for i in all_df.columns if not i.startswith(('time', 'value', 'Meta')) or i == 'Meta Forecaster (RF)'], height=1400)

# Forecasts of mlevel 1 base models on train fold 2
if lvl1_trainfold2:
	preds = load_lvl1_predictions("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Level1", sheet_test="train fold 2",
								  filter_units='13227')
	Visualise.predictions_line(preds, [i for i in preds.columns if not i.startswith(('time', 'value', 'Meta'))], leading_hours=0, height=1200)


if meta_forecasts or meta_forecasts_zoom:
	train_df = pd.read_excel("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta/Performance Meta 13227_2022-05-26-1602.xlsx",
							 sheet_name='training data', usecols=['time', 'value'], engine='openpyxl')
	preds = load_predictions("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Conventional",
							 "C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta", filter_units='13227')
	all_df = pd.concat([train_df, preds], ignore_index=True)
	ref_model_name = [i for i in all_df.columns if i.startswith("Reference")][0]
	all_df.rename(columns={ref_model_name: 'Reference Model'}, inplace=True)

	if meta_forecasts:
		Visualise.predictions_line(all_df, ['Meta Forecaster (Lin. SVR)', 'Reference Model'],
								    # o_path="C:/Users/timvn/OneDrive/Quinyx/Results/predictions_24hr_TEMP.html",
								   showlegend=False, width=840)
	if meta_forecasts_zoom:
		Visualise.predictions_line_zoom(all_df, ['Meta Forecaster (Lin. SVR)', 'Reference Model'], leading_hours=1000,
										)  # o_path="C:/Users/timvn/OneDrive/Quinyx/Results/predictions_TEMP_zoom.html")

if mean_perf_window:
	for window in ['4H', '8H', '12H', '24H']:
		agg_perf = Tablefy.mean_performance("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Conventional",
											"C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Level1",
											"C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta",
											base_forecasters=base_models, window=window,
											o_path=f"C:/Users/timvn/OneDrive/Quinyx/Results/{window}_performance.csv")

if mean_perf or cpi_scatter:
	cpi_data = Tablefy.mean_performance("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Conventional",
										"C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Level1",
										"C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta",
										base_forecasters=base_models, window='1H',
										) # o_path="C:/Users/timvn/OneDrive/Quinyx/Results/mean_performance.csv")
	print(stringify_dicts(cpi_data).T)

	if cpi_scatter:
		Visualise.cost_performance_scatter(data=cpi_data.T.iloc[1:, :], o_path="C:/Users/timvn/OneDrive/Quinyx/Results/CPI_scatter.png")

if cpi_bar:
	Visualise.cost_performance_bar("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Conventional",
								   "C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Level1",
								   "C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta",
								   best_model='Meta Forecaster (RF)', base_forecasters=base_models,
								   )

if horizon or monthly or weekday:
	preds = load_predictions("C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Conventional",
							 "C:/Users/timvn/OneDrive/Quinyx/PyCharm/model_output_no_featselection/Meta/Meta")

	if monthly:
		monthly_perf = Tablefy.monthly_performance(preds, o_path="C:/Users/timvn/OneDrive/Quinyx/Results/monthly_performance.csv")
	if weekday:
		weekday_perf = Tablefy.weekday_performance(preds, o_path="C:/Users/timvn/OneDrive/Quinyx/Results/weekday_performance.csv")
	if horizon:
		horizon_perf = Tablefy.forecast_horizon(preds, horizons=[{'weeks': 1}, {'weeks': 2}, {'months': 1}, {'months': 3}, {'months': 5}],
												o_path="C:/Users/timvn/OneDrive/Quinyx/Results/horizon_performance.csv")
