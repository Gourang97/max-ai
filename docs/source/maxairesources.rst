maxairesources
==============

DataChecks
**********
The datachecks module is dedicated to perform quick checks on the data. 

SparkDataFrameAnalyser
^^^^^^^^^^^^^^^^^^^^^^
performs profiling a PySpark DataFrame. The aim of this class is to show all red flags in data for a given column(s). 

Args:
    - ``df (pyspark.sql.DataFrame)`` - DataFrame to be profiled.
    - ``in_scope_columns (list)`` - If passed, only considers the mentioned columns. If ``None``, all columns are considered in scope. Defaults to ``None``.
    - ``summary_only (bool)`` - (Not Implemented). If set to ``True``, save graphs to a given folder. Defaults to ``False``
    - ``save_report (bool)`` - If set to ``True``, saves the report. Defaults to ``True``
    - ``column_types (dict)`` - A ``dict`` that contains columns by their segregated by their type. Defaults to ``None``. The keys in dictionary are as follow:
        - ``numerical_cols (list)`` - list of numerical columns
        - ``bool_cols (list)`` - list of boolean columns with ``True`` or ``False`` values. ``0``s and ``1``s are not accepted.
        - ``categorical_cols (list)`` - list of categorical columns only. No text columns are allowed
        - ``free_text_cols (list)`` - (Not Implemented). list of text columns.
        - ``unique_identifier_cols (list)`` - unique ID columns (primary key etc.)
    - ``thresholds (dict)`` - Defaults to ``None``
    - ``behaviours (dict)`` - Defaults to None
    - ``sample_ratio (float)`` - Defaults to ``None``
    
>>> from maxairesources.datachecks.dataframe_analysis_spark import SparkDataFrameAnalyser
>>> col_types = {
...     "numerical_cols": [],
...     "bool_cols": [],
...     "categorical_cols": [],
...     "free_text_cols": [],
...     "unique_identifier_cols": []
... }
>>> df = spark.read.csv("path")
>>> analyser = SparkDataFrameAnalyser(df=df, column_types=col_types)   # create instance of analyser
>>> report = analyser.generate_data_health_report()
>>> analyser.save_analysis_report(report)


compare_reports
$$$$$$$$$$$$$$$
compare two report dictionaries coming from same class, key by key. If value increases or decreases by certain basis points then a warning raised.

Args:
    - ``old_report (dict)`` -  reference report to be compared with
    - ``new_report (dict)`` - newly generated report
    - ``threshold_change (int)`` - Deviation metric. If above this threshold, the metric is said to be deviated. Defaults to ``3``.

Returns
    - ``score (int)`` - change or deviation score based on number of places where change is observed.


generate_data_health_report
$$$$$$$$$$$$$$$$$$$$$$$$$$$
method to generate SparkDataFrameAnalyser report.

Args:
    - ``None``

Returns:
    - ``analysis_report (dict)``


Eval
****

ModelEvaluator
^^^^^^^^^^^^^^
runs multiple checks for diagnosis of a model.

Args:
    - ``train_data (pandas.core.frame.DataFrame or pyspark.sql.dataframe.DataFrame)`` - Reference Dataset
    - ``test_data (pandas.core.frame.DataFrame or pyspark.sql.dataframe.DataFrame)`` - Current Dataset, the data to be compared with Reference dataset.
    - ``model (maxaibase.model.model_base.BaseModel)`` - Spark model
    - ``label_col (str)`` - save the report to a html file in the given location
    - ``features (list)`` - list of features required in the training
    - ``cat_features (list)`` - list of categorical features
    - ``sample_ratio (int)`` - Sample size to convert the Spark-DataFrame to Pandas-DataFrame for reference dataset
    - ``pre_process_spark_function (callable)`` - A function which processes the given spark_dataframe. If this argument is passed then the above samples won't be applied anymore.
    
Returns:
    - ``model_evaluation_results (dict)`` - ``True`` if dataset is drifted else ``False``
    - ``train_test_validations (dict)`` - List of columns for which drift has been detected
    
>>> from maxairesources.eval.model_evaluator import ModelEvaluator
>>> evaluator = ModelEvaluator(
...     train_data,
...     test_data,
...     model
...     sample_ratio = 0.2
... )
>>> model_val, train_test_val = evaluator.evaluate()



Utilities
*********

DataFrame
^^^^^^^^^
``DataFrame`` is the data connector utility of Max.AI. It contains two primary methods, ``get()`` for reading the data and ``write()`` for writing the data. The ``DataFrame`` class is designed keeping in mind the config-driven nature of Max.AI modules. One can further refer to its method (listed below) for detailed overview.

get
$$$$
Function to read the data as a Spark or Pandas DataFrame.

Args:
    - ``input_data (dict)`` - Config dictionary container ``port``, ``type`` and ``sourceDetails`` information (or keys)
        - ``port (int)`` - identifier key in the ``input_data``
        - ``type (str)`` - Type of DataFrame. Accepts only two values, ``Pandas`` or ``Spark``
        - ``sourceDetails (dict)`` - a dictionary that captures datasource information. It should have following keys:
            - ``source (str)`` - identifier of the cloud provider. Accepted values: ``s3``, ``adls``.
            - ``fileFormat (str)`` - this parameter depends upon the ``type``. If the ``type=="Spark"``, then supported values are ``iceberg``, ``feast``, ``csv``, ``parquet`` and ``cassandra``. Where as if ``type=="Pandas"``, then supported values are ``csv``, ``parquet`` ``excel`` and ``json``.
            - ``filePath (str)`` - path of the file.
 
Returns:
    - ``output_dataframe (Union[pandas.core.frame.DataFrame, pyspark.sql.dataframe.DataFrame])`` - returns either ``pandas.core.frame.DataFrame`` or ``pyspark.sql.dataframe.DataFrame`` based on ``type`` defined in ``input_data``.
    
>>> from maxairesources.utilities.data_connectors import DataFrame
>>> config_data = [{
...     "port": 1,
...     "type": "pandas",
...     "sourceDetails": {
...          "source": "s3",
...          "fileFormat": "csv",
...          "filePath": "s3://zs-sample-datasets-ds/temp/examples/test.csv"
...     }
... }]
>>> df_obj = DataFrame()
>>> df = df_obj.get(config_data, port_number=1)
>>> df.head()

get_data_for_a_port
$$$$$$$$$$$$$$$$$$$
returns the port details

Args:
    - ``data (dict)``: config dictionary
    - ``port_number(int)``: port number for which details have to be fetched
    - ``connection_type(Optional[str])`` : *Deprecated*. Will be ignored if passed.

Returns:
    - ``port_details (dict)``: port details in dictionary format

>>> from maxairesources.utilities.data_connectors import DataFrame
>>> input_data = [{
...     "port": 1,
...     "type": "pandas",
...     "sourceDetails": {
...         "source": "s3",
...         "fileFormat": "csv",
...         "filePath": "s3://zs-sample-datasets-ds/temp/examples/test.csv"
...     }
... }]
>>> df_obj = DataFrame()
>>> port_details = df_obj.get_data_for_a_port(input_data,port_number=1)
>>> print(port_details)
    
get_default_mandatory_arguments
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
Function to get the default arguments and mandatory arguments for particular DataFrame ``type``, ``format`` and operation (``get`` or ``write``). 

Args:
    - ``df_type (str)`` - Type of DataFrame. It can be either ``'Pandas'`` or ``'Spark'``
    - ``df_format (str)`` - format of all the data. One can use ``get_supported_formats()`` to get the list of available data formats supported by the ``DataFrame``.
    - ``operation (str)`` - either ``'get'`` or ``'write'``
    
>>> from maxairesources.utilities.data_connectors import DataFrame
>>> df_type = 'spark'
>>> operation = 'write'
>>> df_obj = DataFrame()
>>> df_obj.get_default_mandatory_arguments(
...     df_type,
...     df_format,
...     operation
... )

get_supported_formats
$$$$$$$$$$$$$$$$$$$$$
Returns the dictionary of the supported formats.

Args:
    - ``None``
    
Returns:
    - ``dict`` - Dictionary of all the supported formats with their keys
    
>>> from maxairesources.utilities.data_connectors import DataFrame
>>> df_obj = DataFrame()
>>> df_obj.get_supported_formats()

write
$$$$$$
Function to write the data in the declared file-format.

Args:
    - ``df (Union[pandas.core.frame.DataFrame, pyspark.sql.dataframe.DataFrame])`` - DataFrame to be written
    - ``output_data (dict)`` - Config dictionary container ``port``, ``type`` and ``sourceDetails`` information (or keys)
        - ``port (int)`` - identifier key in the ``input_data``
        - ``type (str)`` - Type of DataFrame. Accepts only two values, ``Pandas`` or ``Spark``
        - ``sourceDetails (dict)`` - a dictionary that captures datasource information. It should have following keys:
            - ``source (str)`` - identifier of the cloud provider. Accepted values: ``s3``, ``adls``.
            - ``fileFormat (str)`` - this parameter depends upon the ``type``. If the ``type=="Spark"``, then supported values are ``iceberg``, ``feast``, ``csv``, ``parquet`` and ``cassandra``. Where as if ``type=="Pandas"``, then supported values are ``csv``, ``parquet`` ``excel`` and ``json``.
            - ``filePath (str)`` - path of the file.
 
Returns:
    - ``status (boolean)`` - returns ``True`` if the data is written.
    
>>> from maxairesources.utilities.data_connectors import DataFrame
>>> config_data = [{
...     "port": 1,
...     "type": "pandas",
...     "sourceDetails": {
...          "source": "s3",
...          "fileFormat": "csv",
...          "filePath": "s3://zs-sample-datasets-ds/temp/examples/test/"
...     }
... }]
>>> df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
>>> df_obj = DataFrame()
>>> status = df_obj.write(df,config_data,port_number=1)
>>> print(status)


SparkDistributor
^^^^^^^^^^^^^^^^

A PySpark wrapper module to distribute Python functions which are mainly written using Pandas. SparkDistributor converts the Python functions to PandasUDF and runs them at scale.

Args:
    - ``python_function (Callable)`` - A user defined function that should take Pandas Dataframe as input and return Pandas Dataframe as output.
    - ``spark_dataframe (pyspark.sql.DataFrame)`` - The Dataframe which needs to be processed using the ``python_function``.
    - ``sample_size (int, optional)`` - The number of sample records to be used to call the ``python_function`` directly. The call to ``python_function`` using sample of a ``Pandas.DataFrame`` is used to infer the schema for the final dataframe. *Increase the sample size if the python function is not able to execute with the given sample size*. Defaults to ``100``.
    - ``output_schema (optional)`` - schema of the output dataframe. If None the function tries to infer the schema by using sample of data. The size of the sample is specified by sample size. Defaults to ``None``.
    - ``group_key`` - Name of the column to do grouby on. If None then spark partition id is used as a ``group_key``. Defaults to ``None``.
    - ``parallelism`` - Specifies the number of partitions. If none then no repartition is performed. Defaults to ``None``.
    - ``args`` - Arguments to ``python_function``.
    - ``kwargs`` - Keyword Arguments to ``python_function``.
    
>>> from maxairesources.utilities import SparkDistributor
>>> spark_wrapper = SparkDistributor(python_function=python_function, spark_dataframe=spark_df)
>>> result = spark_wrapper.pandas_to_spark_wrapper()
>>> result.show(5)
