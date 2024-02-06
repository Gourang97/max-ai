maxaimonitoring
===============


DataDriftChecker
****************
Data drift checker module to compare drifts between 2 Pandas or PySpark Dataframes.

Args:
    - ``reference_data (pandas.DataFrame or pyspark.sql.DataFrame)`` - Reference Dataset
    - ``current_data (pandas.DataFrame or pyspark.sql.DataFrame)`` - Current Dataset. Data to be compared
    - ``drifted_features_threshold (Union[int,float])`` - Threshold value for drift to be True. If int, it denotes the number of columns that has to be drifted for the whole dataset to be drifted as True. If float, it denotes the percentage of columns that has to be drifted for the whole dataset to be drifted as True. Defaults to ``0.33``.
    - ``save_report_filepath (str, optional)`` - save the report to a html file in the given location
    - ``sample_size_reference (int)`` - number of rows of data to be used for reference dataset. Defaults to ``500000``.
    - ``sample_size_current (int, optional)`` - number of rows of data to be used for current dataset. Defaults to ``500000``.
    - ``pre_process_spark_function (Callable, optional)`` - A function which processes the given spark_dataframe. If this argument is passed then the above samples won't be applied. Defaults to ``None``.
    - ``column_mapping_dictonary (dict, optional)`` - Columns to be mapped for numerical features. Accepted keys in the dictionary are ``['numerical_features','categorical_features']``. Defaults to ``{}``.
    - ``drift_detection_for_num_cols (str, optional)`` - Drift detection method to be used for Numerical features. Accepted Methods are ``['ks','wasserstein','kl_div','psi','jensenshannon','anderson']``. Defaults to ``None``.
    - ``drift_detection_for_cat_cols (str, optional)`` - Drift detection method to be used for Categorical features. Accepted Methods are ``['chisquare','z','kl_div','psi','jensenshannon']``. Defaults to ``None``.
    - ``return_complete_drift_details (bool, optional)`` - returns the result set of the Analyzer with all the details. Defaults to ``None``.
    
Returns:
    - ``dict`` with following keys:
        - ``is_drifted (bool)`` - True if Dataset is drifted else False
        - ``drift_detected_columns (list)`` - List of columns for which drift has been detected
        - ``ignored_null_columns (list)`` - List of columns where the percentage of null values is more than ``95%``.
        - ``drift_score (float)`` - Drift score metric
        - ``result (evidently.analyzers.data_drift_analyzer.DataFrameAnalyzer)`` - This result is only returned when return_complete_drift_details parameter is set to True
        
        
>>> from maxaimonitoring.data_drift.data_drift_checker import DataDriftChecker
>>> drift_checker = DataDriftChecker(
...     reference_data,
...     current_data,
...     drifted_features_threshold=0.5
...     save_report_filepath="report.html"
... )
>>> result = drift_checker.data_drift_check()


ModelDrift
**********
Model drift checker to detect drift between current model and reference model or current model with Dummy model

Args:
    - ``current_data (pandas.DataFrame or pyspark.sql.DataFrame)`` - Current Dataset. Data to be compared.
    - ``predicted_column_name (str)`` - Name of the Column which has predictions
    - ``problem_type (str)`` - Type of the problem. Supported problem types are ``["regression", "binary_classification", "multiclass_classification", "no_target_performance"]``.
    - ``reference_data (optional)`` - Reference Dataset. If reference dataset is not passed, then the model is evaluated against a dummy model. Defaults to ``None``.
    - ``target_column_name (str, optional)`` - Name of the Column which has ground truth. Defaults to ``None``.
    - ``column_mapping_dictonary (dict, optional)`` - Columns to be mapped for numerical features. Accepted keys in the dictionary are ``['numerical_features','categorical_features']``. Defaults to ``{}``.
    - ``prediction_type (str, optional)`` - Type of prediction in case of classification problem. Supported prediction types are ``["labels", "probas"]``. **Please do not pass raw predictions as currently it is not supported**. Defaults to ``'labels'``
    - ``sample_size_reference (int, optional)`` - number of rows of data to be used for reference dataset. Defaults to ``500000``.
    - ``sample_size_current (int, optional)`` - number of rows of data to be used for reference dataset. Defaults to ``500000``.
    - ``threshold (Union[int, float], optional)`` - If int, then the the threhold is number of tests to be passed to mark the model as drifted. If float, then the the threhold is percentage of tests to be passed to mark the model as drifted. Defaults to ``0.8``.
    - ``save_report_filepath (bool, optional)`` - save the report to a html file in the given location. Defaults to ``None``.
    
Returns:
    - ``dict`` with following keys:
        - ``is_model_drifted (bool)`` - True if Model is drifted else False
        - ``is_prediction_drifted (bool)`` - Returned only when reference data is passed.
        - ``test_result (dict)`` - returns complete test results
        
>>> from maxaimonitoring.model_drift.model_drift_checker import ModelDrift
>>> model_drift = ModelDrift(
...    current_data=current_data,
...    reference_data=reference_data,
...    target_column_name='label',
...    predicted_column_name='prediction',
...    problem_type='binary_classification',
...    prediction_type='labels',
...    threshold=0.8,
...    save_report_filepath = 'model_drift_report.html'
... )
>>> result_dict = model_drift.check_model_drift()
