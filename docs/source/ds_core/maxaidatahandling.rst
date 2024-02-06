Data Handler
=================
Wrappers to handle data in pipelines.

DataEvaluator
^^^^^^^^^^^^^
runs multiple checks for validating a dataset.

Args:
    - ``data (pandas.core.frame.DataFrame or pyspark.sql.dataframe.DataFrame)`` - reference dataset
    - ``label_col (str)`` - save the report to a **html** file in the given location
    - ``sample_ratio (Union[float], optional)`` - Sample size to convert the Spark-DataFrame to Pandas-DataFrame for reference dataset. Defaults to ``0.2``.
    - ``pre_process_spark_function (Callable, optional)`` - A function which processes the given spark-dataframe. If this argument is passed then the above ``sample_ratio`` won't be applied. Defaults to ``None``.
    
>>> from maxaidatahandling.data_evaluator import DataEvaluator
>>> evaluator = DataEvaluator(
...     data,
...     label_col='label',
...     sample_ratio=0.2
... )
>>> integrity_results = evaluator.evaluate()
    

MaxExpectations
^^^^^^^^^^^^^^^
a class that executes built in expectations of Great Expectations on data

Args:
    - ``df (pyspark.sql.Dataframe)``: dataframe on which we have to execute the expectations
    - ``expeconfig (dict[str, list])``: configuration containing parameter values for all the expectations
        - ``expectations (List[Dict[str, Union[Dict, list, str, int]]])`` - a list of dictionaries specifiying
        expectations to be ran.
            - ``expectations_id (int)`` - identifier of which expectation is to be ran. The list of all the
            available expectations is defined after this section.
            - ``kwargs (dict)`` - identifier specific parameters. Specified in the section below.

Returns:
    - ``dict``: returns a dictionary updated with result value given by every expectation executed

..  code-block:: python

    from maxaidatahandling.data_expectations import MaxExpectations

    # define config for data_expectations
    expeconfig = {
        "expectations": [
            {
                "expectations_id": 19,
                "function_name": expect_column_min_to_be_more_than,    # callable (custom expectation)
                "meta_attribute": "column_aggregate",
                "function_args": {
                    "column": "deadline",
                    "value": "2016-07-01",
                    "parse_strings_as_datetimes": True
                },
                "expectation_name": "",
                "result": {}
            },
            {
                "expectations_id": 1,
                "column_name": [
                    "id"
                ],
                "kwargs": {},
                "expectation_name": "",
                "result": {}
            },
            {
                "expectations_id": 2,
                "column_name": [
                    "goal_usd"
                ],
                "kwargs": {},
                "expectation_name": "",
                "result": {}
            }
        ]
    }
    expectations_ops = MaxExpectations(data, expeconfig)
    expeconfig = expectations_ops.execute()
    print(expeconfig)    # result
    
The expectations available currently are as follow:
    1. ``expect_column_values_to_be_unique``
    2. ``expect_column_values_to_not_be_null``
    3. ``expect_column_values_to_be_of_type``
    4. ``expect_column_values_to_be_between``
    5. ``expect_column_values_to_match_strftime_format``
    6. ``expect_column_values_to_be_json_parseable``
    7. ``expect_column_values_to_match_json_schema``
    8. ``expect_column_distinct_values_to_be_in_set``
    9. ``expect_column_distinct_values_to_contain_set``
    10. ``expect_column_mean_to_be_between``
    11. ``expect_column_median_to_be_between``
    12. ``expect_column_stdev_to_be_between``
    13. ``expect_column_pair_values_A_to_be_greater_than_B``
    14. ``expect_column_pair_values_to_be_in_set``
    15. ``expect_select_column_values_to_be_unique_within_record``
    16. ``expect_compound_columns_to_be_unique``
    17. ``expect_column_values_to_match_regex_list``
    18. ``expect_column_values_to_not_match_regex_list``
    19. ``custom_expectations``


MaxDataset
^^^^^^^^^^
Class to read and perform basic preprocessing on read data. It can also be used both as a data reader and a data writer. This class works well with existing input and output configurations in ``py_config.json`` files.

Args:
    - ``name (str)``: name of the dataset
    - ``dataset_config (dict, optional)``: configuration that captures the input and preprocessing details. Defaults to ``None``.
    - ``df (spark.sql.DataFrame, optional)``: Dataset on which the MaxDataset is to be run. Defaults to ``None``. If  declared as ``None``, then, this module looks reads the data from the ``sourceDetails`` defined in the ``dataset_config``.


**Read and preprocess example**

.. code-block:: python

    from maxaidatahandling.dataset import MaxDataset
    
    
    # dataset_name and data_config
    dataset_name = "sample_data"
    data_config = {
        "port": 1,
        "dataType": "dataframe",
        "sourceDetails": {
            "source": "s3",
            "fileFormat": "csv",
            "filePath": "dim_customer.csv",
        },
        "preprocess": {
            "rename_cols": {"dob": "date_of_birth", "is_employee": "is_employee_bool"},
            "select_cols": [
                "customer_id",
                "signup_date",
                "date_of_birth",
                "first_store_id",
                "is_employee_bool",
                "is_outlier",
                "gender",
                "language",
            ],
            "re_partition": {"on": ["customer_id"], "size": 1},
            "data_analysis": {
                "sample_ratio": 0.3,
                "col_types": {
                    "numerical_cols": [],
                    "bool_cols": ["is_employee_bool", "is_outlier"],
                    "categorical_cols": ["gender", "language"],
                    "free_text_cols": [],
                    "unique_identifier_cols": ["customer_id"],
                },
            },
            "cache": True,
        },
    }
    
    # create instance of Dataset and prepare datasets
    ds_read_obj = MaxDataset(name=dataset_name, dataset_config=data_config)
    ds_read_obj.prepare_dataset()
    
    
**Write Mode usage**

..  code-block:: python

    from maxaidatahandling.dataset import MaxDataset

    
    # output data_config
    op_data_config = {
        "port": 1,
        "dataType": "dataframe",
        "sourceDetails": {
            "source": "s3",
            "fileFormat": "csv",
            "filePath": "copy_dim_customer",
        },
        "preprocess": {
            "rename_cols": {"date_of_birth": "dob", "is_employee_bool": "is_employee"},
            "select_cols": [
                "customer_id",
                "signup_date",
                "dob",
                "first_store_id",
                "is_employee",
                "is_outlier",
                "gender",
                "language",
            ],
            "re_partition": {"on": ["customer_id"], "size": 1},
            "data_analysis": {
                "sample_ratio": 0.3,
                "col_types": {
                    "numerical_cols": [],
                    "bool_cols": ["is_employee", "is_outlier"],
                    "categorical_cols": ["gender", "language"],
                    "free_text_cols": [],
                    "unique_identifier_cols": ["customer_id"],
                },
            },
        },
    }
    # create instance of Dataset and store data
    ds_write_obj = MaxDataset(name=dataset_name, dataset_config=op_data_config)
    ds_write_obj.store_data()


To enable column masking feature, add the following key-values in the config under the ``sourceDetails`` key. These entities are entirely optional:-
    - ``"encryption_enabled": True`` - used to specify that encryption should be enabled when reading or writing the data
    - ``"encrypt_key": "myKey"`` - key used when encrypting the column name and the same key should be used when decrypting (Mandatory if ``encryption_enabled`` is set to True)
    - ``"encrypt_prefix":"f_"``- prefix to be used for the encrypted column names (Mandatory if ``encryption_enabled`` is True)
    
