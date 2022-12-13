maxairesources
==============

Utilities
*********

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