maxaimarketplace
================
This document captures the flow to deploy DAGs and use-cases in the Max Marketplace repository. We encourage the use of conventions and structures laid down in this document for the following reasons:

- Standardization - when similar conventions are followed, it accelerates the debugging process.
- Efficiency - everytime a PR is accepted, the use-case drivers and handlers files should be packed and built into a Docker image. Following these steps, helps to automate that process and trigger build for only use-cases which have been updated (covered in detail in the later sections), rather than trigger build for all the use-cases.

Directory Structure
*******************
Every client deployment (dev and prod) should have three directories, one for *configs*, one for *DAGs* and one for *use-case*. These directories should be further divided into sub-directories based on *domain*, which further are divided into *usecase*. The *usecase* is further divided into *objectives*, which captures execution files and configs. For illustration:

.. code-block:: text

   maxaimarketplace
       ├──maxaiconfigs
       |     ├──domain_01
       |     │    └──usecase_01
       |     |       ├──spark_config.json
       |     |       └──obj0
       |     |            └──py_config.json
       |     └──domain_02
       |          └──usecase_02
       |               ├──kubepod_config.json
       |               └──obj0
       |                    └──py_config.json
       ├──maxaidags
       |     ├──domain_01
       |     │    └──usecase_01
       |     |       └──obj0
       |     |            └──dag.py
       |     └──domain_02
       |          └──usecase_02
       |               └──obj0
       |                    └──dag.py
       └──maxaiusecases
             ├──domain_01
             │    ├──usecase_01
             |    |    └──handlers
             |    |         ├──__init__.py
             |    |         ├──handler_01.py
             |    |         └──handler_02.py
             |    ├──Dockerfile
             |    ├──README.md
             |    ├──requirements.txt
             |    └──setup.py
             └──domain_02
                  ├──usecase_02
                  |    └──handlers
                  |         ├──__init__.py
                  |         ├──handler_01.py
                  |         └──handler_02.py
                  ├──Dockerfile
                  ├──README.md
                  ├──requirements.txt
                  └──setup.py


The structure of configs, DAGs and usecases is covered in the following sections.

Configs
*******
As **Max.AI** deployments are config-driven, this section talks about the structure of the configs, specifically if DagFactory is being used for purpose of DAG creation.

Currently, three types of configs are supported:
    1. ``spark_config.json`` - used by Airflow's *SparkSubmitOperator*.
    2. ``kubepod_config.json`` - used by Airflow's *KubernetesPodOperator*.
    3. ``py_config.json`` - used by Python handlers, which execute the driver code.
    
The structure of these configuration files is described in detail in the following sections:

spark_config
^^^^^^^^^^^^
The ``spark_config.json`` contains two type of arguments:
    1. SparkConf arguments - which are applicable across all the tasks, like `container.image`, `namespace` etc.
    2. Task-specific arguments - which are applicable only for a specific task, like `name`, `num-executors`, `driver-memory` etc
    
Hence, ``spark_config.json`` should have following args:
    - ``conf`` - key-values pairs of SparkConf arguments, with keys being spelled exactly as defined in `official Spark configurations document <https://spark.apache.org/docs/latest/configuration.html>`_ followed by their respective values.
    - ``tasks`` - a **list** of dicts, with three mandatory keys, ``name``, ``task_id`` and ``spark_submit_conf``.
        - ``name`` - a string with task name, same as defined in DAGFactory task definition.
        - ``task_id`` - integer that defines the order of task in the DAG.
        - ``spark_submit_conf`` - the keys here should be in accordance with `spark_submit_operator <https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/_api/airflow/providers/apache/spark/operators/spark_submit/index.html>`_.
        
.. code-block:: json

    {
        "conf": {...},
        "tasks": [
            {
                "name": "task_1",
                "task_id": 1,
                "spark_submit_conf": {...}
            },
            {
                "name": "task_2",
                "task_id": 2,
                "spark_submit_conf": {...}
            }
        ]
    }
        



