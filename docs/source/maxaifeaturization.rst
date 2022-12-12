maxaifeaturization
==================

The Featurization Module is used for feature engineering workloads. This module has following submodules:
- Aggregation
- Decomposition
- Time Series
- Transformation

Aggregation
***********

.. automodule:: maxaifeaturization.aggregation.Aggregation
    :members:
    :undoc-members:
    :show-inheritance:

The aggregation submodule performs the ``.groupBy().agg()`` operation on the dataframe. It supports various PySpark in-built transformations and custom transformations.

>>> from maxaifeaturization.aggregation import Aggregation
>>> df = spark.read.csv(filepath)    # file on which aggregations are to performed
>>> agg_dict = {
...    "entity_column": "customer_id",
...    "aggregation_ops": [
...      {
...         "aggregation": 2,
...         "feature": ["total_revenue"],
...         "output_column_name": ["mean_rev"]
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

As shown in the example snippet above, each aggregations are encoded. Please refer to the following list for aggregation-to-encoder mapping:

- 1: sum
- 2: mean
- 3: stddev
- 4: max
- 5: min
- 6: count
- 7: count_distinct
- 8: variance
- 9: percentile
- 10: quantile
- 11: median
- 12: most_frequent


Decomposition
*************


Time Series
***********


Transformation
**************