from Packages.ModelDev import *


def model_training_testing(selected_models, meta_models, team_id,
                           train_size=0.7, test_size=0.1, max_datasets=1, min_length=17500, time_min=None, time_max=None, selected_units=None,
                           o_folder="", conventional=True,
                           feat_selection=False, direction='forward', n_features=2/3,
                           stationarity=False):
    """
    Trains and tests a meta model and its level 1 models, and optionally trains and tests conventional models
    A specified folder is checked for previously run models, and their output is loaded to reduce runtimes
    :param selected_models: list of models strings to run for conventional and level 1 models.
    :param feat_selection: bln, if set to true
    :param direction: 'forward' or 'backward'. Only used if feat_selection == True
    :param n_features: number of features so select. Can be int or fraction (float). Only used if feat_selection == True
    :param stationarity: make datasets stationary
    :param meta_models: list of model objects to generate meta models from
    :param team_id: string, customer identifier
    :param train_size: fraction, level 1 first training fold size
    :param test_size: can be fraction, or string: "first/last X year(s)/month(s)/week(s)/day(s)/hour(s)/minute(s)/second(s)/microsecond(s)"
    :param max_datasets: max number of units to retrieve
    :param min_length: minimum length of each time series
    :param time_min: min. starting time for each time series
    :param time_max: cutoff time for each time series
    :param selected_units: load data from only a limited set of units, if not selected_unit=None
    :param o_folder: path to retrieve data from, and save data to
    :param conventional: if set to True, conventional models will also be trained and validated
    :return:
    """
    function_output = []

    # Load datasets
    customer, datadict = retrieve_data(teamId=team_id, att_names=['traffic'], selected_units=selected_units,
                                       max_units=max_datasets, min_length=min_length, time_min=time_min, time_max=time_max)

    # Train conventional models
    if conventional:
        print("DATA PREPROCESSING AND TRAINING CONVENTIONAL MODELS")
        conv_output = conventional_pipeline(datadict, customer, selected_models, test_size, o_folder)
        function_output.append(conv_output)

    # Train level 1 and meta models
    meta_output = meta_pipeline(datadict, customer, selected_models,
                                feat_selection, direction, n_features,
                                meta_models, test_size, o_folder,
                                stationarity)
    function_output.append(meta_output)

    return function_output

