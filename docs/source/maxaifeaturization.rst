maxaifeaturization
==================


Aggregation
***********

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


Decomposition
*************


Time Series
***********


Transformation
**************