maxaifeaturization
==================

The Featurization Module is used for feature engineering workloads. This module has following submodules:

- Aggregation
- Time Series
- Transformation

Aggregation
***********
The aggregation submodule performs the ``.groupBy().agg()`` operation on the dataframe.
It supports various PySpark in-built transformations and custom transformations.

Args:
    - ``df (pyspark.sql.DataFrame)`` - dataframe on which operations are to be performed
    - ``arguments (dict)`` - a dictionary that captures all the aggregation operations to be performed
        - ``entity_column (str)`` - column on which ``groupBy`` operation is to be performed
        - ``aggregation_ops (list(dict))`` - a a list of dictionaries capturing aggregation operations. The dictionary will have following elements:
            - ``aggregation (int)`` - identifier of the operation to be performed, as defined below
            - ``feature (list)`` - column on which operation is to be performed
            - ``output_column_name (list)`` - name for the output column

Please refer to the following list for aggregation-to-encoder mapping:
    - 1: ``sum``
    - 2: ``mean``
    - 3: ``stddev``
    - 4: ``max``
    - 5: ``min``
    - 6: ``count``
    - 7: ``count_distinct``
    - 8: ``variance``
    - 9: ``percentile``
    - 10: ``quantile``
    - 11: ``median``
    - 12: ``most_frequent``

>>> from maxaifeaturization.aggregation import Aggregation
>>> df = spark.read.csv(filepath)    # file on which aggregations are to performed
>>> agg_dict = {
...    "entity_column": "customer_id",          # GroupBy Column
...    "aggregation_ops": [                     # list of dictionaries
...      {
...         "aggregation": 2,                   # which aggregation to be performed
...         "feature": ["total_revenue"],       # column on which aggregation is to be performed
...         "output_column_name": ["mean_rev"]  # name of aggregated column
...     },
...     {
...         "aggregation": 4,
...         "feature": ["total_revenue"],
...         "output_column_name": ["max_rev"]
...     },
...   ]
... }
>>> agg_obj = Aggregation(df=df, arguments=agg_dict)
>>> agg_df = agg_obj.execute()

----------

Time-Series
***********
The time-series transformations are the set of transformations that are to be performed on temporal data. Currently, only the following univariate transformations are supported in this module:

- autocorrelation
- time series decompositions

This module is built with the belief that majority of time-series datasets are combinations of small (often independent) time-series. These time-series are distinguishable from each other by some identifier. For instance, a large warehouse will store thousand of items that have a unique identifier. This column is defined as ``groupby_col``. This feature will have a univariate time-series defined against it, i.e., no duplicates exist on the datetime column. 

For end-to-end functioning of time-series module, please refer to this `example notebook <https://dev.azure.com/personalize-ai/max.ai/_git/max.ai.ds.core?path=/documents/Time-Series-E2E.ipynb&_a=preview>`_.

autocorrelation
^^^^^^^^^^^^^^^
Computes the Pearson correlation between the Series and its shifted self. 

>>> from maxaifeaturization.timeseries.univariate import autocorrelation
>>> abdf = autocorrelation(
...    spark_df, 
...    groupby_col="machine_id", 
...    datetime_col="date", 
...    value_col="sensor_reading", 
...    nlags=2, 
...    partial=True
... )


time_series_decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^
performs ``statsmodels`` style decomposition on all the time-series in a dataframe. In this method, ``moving_average`` and ``loess`` are supported and can be passed in the function call with the ``method`` argument.

>>> from maxaifeaturization.timeseries.univariate import time_series_decomposition
>>> ddf = time_series_decomposition(
...    spark_df, 
...    groupby_col="machine_id", 
...    datetime_col="date", 
...    value_col="sensor_reading", 
...    method="loess"
... )

----------

Transformation
**************
Defines simple transforms that don't change the shape of the dataframe (as opposed to ``Aggregation`` defined above).

Transformation
^^^^^^^^^^^^^^
performs columnar transformation on the PySpark DataFrame.

Args:
    - ``df (pyspark.sql.DataFrame)``: Dataframe on which transformation operations are to be performed
    - ``arguments (dict)``: a dictionary that captures all the transformation operations to be performed
        - ``transform_ops (list(dict))`` - a list of dictionaries capturing transform operations. The dictionary will have following 
            - ``feature (list)`` - column on which transformation is to be performed
            - ``transformation (int)`` - identifier for a transformation. Reference list is provided below.
            - ``rules (dict)`` - *will be removed in future*.
            - ``rule_expression (str)`` - *will be removed in future*.
            - ``output_column_name (str)`` - name for the transformed column
            - ``retain_original (bool)`` - if ``True``, original column will be retained, otherwise dropped.

Transformation available are defined as below. 
The indentifier number added against ``transformation`` will execute that particular transformation.
    - 2: ``z-score``
    - 3: ``exp``
    - 4: ``log``
    - 5: ``reciprocal``
    - 6: ``box-cox``
    - 7: ``binning``
    - 8: ``string-indexer``
    - 9: ``one-hot-encoding``
    - 10: ``concat-with-delimiter``
    - 11: ``split``
    - 12: ``uppercase``
    - 13: ``lowercase``
    - 14: ``trim``
    - 15: ``timestring-to-iso8601``
    - 16: ``epoch-to-iso8601``

Methods
@@@@@@@

execute
$$$$$$$
driver method of the transform

Args
    - ``None``

Returns
    - ``pyspark.sql.DataFrame``

>>> from maxaifeaturization.transformation import Transformation
>>> transform_dict = {
...     "transform_ops": [
...         {
...             "feature": ["Weekly_Sales"],
...             "transformation": 2,
...             "rules": {},
...             "rule_expression": "",
...             "output_column_name": "Weekly_Sales_Z",
...             "retain_original": True
...         }
...    ]
... }
>>> trans_ops = Transformation(df, transform_dict)
>>> output_df = trans_ops.execute()

WindowOperations
^^^^^^^^^^^^^^^^
creates the rolling window features on a PySpark DataFrame.

Args:
    - ``df (pyspark.DataFrame)``: dataframe on which operations are to be performed
    - ``arguments (dict)``: a dictionary capturing the details of the operations to be performed
        - ``window_spec (dict)`` - a dictionary containing window-defining features
            - ``partition_cols (list(str))`` - a list of string defining columns on which to partition the datasets
            - ``order_col (str)`` - column by which to order the data
            - ``asc (bool)`` - if True, the data will be ordered in ascending order. Otherwise, in descending order.
            - ``window_size (int)`` - size of window
        - ``window_ops (list(dict))`` - a list of dictionary. Each dictionary instance should capture one window operation to be performed
            - ``feature (str)`` - name of the column on which operation is to be performed
            - ``operation (int)`` - identifier of the operation to be performed, as defined below
            - ``output_column_name (str)`` - name of the output column

Returns:
    - ``df (pyspark.DataFrame)``: dataframe with additional features columns

Please refer to the following list for rolling_window_transformation-to-encoder mapping:
    - 1: ``differencing``
    - 2: ``avg``
    - 3: ``median``
    - 4: ``sum``
    - 5: ``max``
    - 6: ``min``
    - 7: ``stddev``
    - 8: ``variance``
    - 9: ``lead``
    - 10: ``lag``
    - 11: ``cumulative_distribution``
    - 12: ``row_number``
    - 13: ``rank``
    - 14: ``dense_rank``
    - 15: ``percent_rank``

>>> from maxaifeaturization.transformation import WindowOperations   
>>> # define the arguments dictionary
>>> window_dict = {
...     "window_spec": {
...         "partition_cols": ["Dept"],
...         "order_col": "Date",
...         "asc": True,
...         "window_size": 7,
...     },
...     "window_ops": [
...         {
...             "feature": "Weekly_Sales",
...             "operation": 2,
...             "output_column_name": "Weekly_Sales_Avg"
...         }
...     ]
... }
>>> # initialize the WindowOperations class
>>> w_obj = window.WindowOperations(sales_df, window_dict)
>>> sales_df_updated = w_obj.execute()
