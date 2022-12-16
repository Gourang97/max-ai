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
    - ``tasks`` - a **list** of dicts, with three mandatory keys, ``name``, ``id`` and ``spark_submit_conf``.
        - ``name`` - a string with task name, same as defined in DAGFactory task definition.
        - ``id`` - integer that defines the order of task in the DAG.
        - ``spark_submit_conf`` - the key-value pairs, in accordance with `spark_submit_operator <https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/_api/airflow/providers/apache/spark/operators/spark_submit/index.html>`_.
        
.. code-block:: json

    {
        "conf": {...},
        "tasks": [
            {
                "name": "task_1",
                "id": 1,
                "spark_submit_conf": {...}
            },
            {
                "name": "task_2",
                "id": 2,
                "spark_submit_conf": {...}
            }
        ]
    }
    
If, for some reason, a particular task requires a different set of SparkConf, then in the ``spark_submit_conf``, the revised configurations can be defined against a ``conf`` argument. For instance, in the spark_config defined below, the first and third tasks will be executed by ``image_01:latest`` and second will be executed by ``image_02:latest``.

.. code-block:: json
    
    {
        "conf": {
            "spark.kubernetes.container.image": "image_01:latest",
            "spark.kubernetes.container.image.pullSecrets": "some_secret",
            "spark.kubernetes.container.image.pullPolicy": "Always"
            },
        "tasks": [
            {
                "id": 1,
                "name": "first_task",
                "spark_submit_conf": {
                    "application": "first_main.py",
                    "spark_binary": "spark-submit"
                }
            },
            {
                "id": 2,
                "name": "second_task",
                "spark_submit_conf": {
                    "conf": {
                        "spark.kubernetes.container.image": "image_02:latest"     # a different will be used to execute this task
                    },
                    "application": "second_main.py",
                    "spark_binary": "spark-submit"
                }
            },
            {
                "id": 3,
                "name": "third_task",
                "spark_submit_conf": {
                    "py_files": "",
                    "application": "third_main.py",
                    "spark_binary": "spark-submit"
                }
            }
        ]
    }
    

    

kubepod_config
^^^^^^^^^^^^^^
The ``kubepod_config.json`` is used to provide the configuration for KubernetesPodOperator. The structure is similar to spark_config:
    - ``id`` - integer that defines the order of task in the DAG.
    - ``name`` - a string with task name, same as defined in DAGFactory task definition.
    - ``conf`` - key-value pairs, in accordance with `KubernetedPodOperator <https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/_api/airflow/providers/cncf/kubernetes/operators/kubernetes_pod/index.html#airflow.providers.cncf.kubernetes.operators.kubernetes_pod.KubernetesPodOperator>`_.


.. code-block:: json

    {
        "tasks": [
            {
                "id": 1,
                 "name": "task_1",
                 "conf": {...}
            },
            {
                "id": 2,
                 "name": "task_02",
                 "conf": {...}
            },
        ]
    }
    
    
py_config
^^^^^^^^^
The PyConfig should contain a **task name** as a key and task specific configs as its value. The following example illustrates that:

.. code-block:: json

    {
        "task_01": {
            ...
        },
        "task_02": {
            ...
        }
    }


.. note::
    The "name" of task in all the configs should be same as the task_id in the DAGFactory's DAG definition.
    
How to access py_config.json in Python Driver file?
+++++++++++++++++++++++++++++++++++++++++++++++++++
The ConfigStore can be accessed in the Python scripts using the ``maxairesources.config_store.config`` module. One can decorate their ``execute`` function by ``config.pyconfig``. The code snippet looks somewhat like this:-

.. code-block:: python

    import sys
    from maxairesources.config_store import config
    from maxairesources.logging.logger import get_logger


    logger = get_logger(__name__)


    class ComponentHandler(object):
        def __init__(self):
            pass

        def execute_component(self, request_data):
            input_data = request_data["input"]
            output_data = request_data["output"]
            arguments = request_data["function"]["args"]

            logger.info("Execute component....")


    @config.main()
    def execute(**kwargs):
        access_credentials = kwargs["access_credentials"]
        py_config = kwargs["config"]        # imports the whole py_config
        task_config = py_config["task_id"]  # define which task this pyconfig belongs to

        ComponentHandler().execute_component(request_data=task_config)


    try:
        logger.info("Task Started")
        exit_code = 0
        execute(argument=input_argument)
        logger.info("Task Ended")
    except Exception as e:
        exit_code = 1
        raise Exception(e)
    finally:
        spark.stop()
        sys.exit(exit_code)
    
    
DAGFactory
**********
The DAGFactory is an abstraction layer built on top of Airflow, specifically for Max.AI. The motivation behind development of DAGFactory is to provide a platform to build DAG quickly by abstracting out all the unnecessary details like connections, reading configs etc. It also strives to standardize the DAGs.

.. code-block:: python

    from maxairesources import dagfactory as DG


    # initalize DAGFactory instance
    dgf = DG.DAGFactory(
        dag_id="obj0_customer360",
        domain_name="qsr",
        usecase="customer360",
        obj="obj0",
        schedule_interval=None,
        default_args={
            "depends_on_past": True,
            "retries": 0,
            "start_date": datetime(2022, 7, 14)
        }
    )

    dgf.add(
        task_id="start_task",
        operator="DummyOperator",
        parent=["root"]
    )

    dgf.add(
        task_id="process_data_with_spark",
        operator="SparkSubmitOperator",
        parent=["start_task"]
    )

    dgf.add(
        task_id="end_task",
        operator="DummyOperator", 
        parent=["process_data_with_spark"]
    )

    dag = dgf.create_dag()
    
    
.. warning::
    it is essesntial to assign returned object to a variable. This is because, if assignment is not done, the DAG won't be appear in ``global()`` scope. For further details, please check this `link <https://airflow.apache.org/docs/apache-airflow/1.10.3/concepts.html?highlight=variable#scope>`_.
    

The task dependencies in the DAGFactory are captured by ``parent`` argument in the ``add`` method. The ``parent`` accepts a list of tasks on which the current task is dependent on. If a task has no dependency, i.e. it is the first task, one should mention ``parent=["root"]``, signifying it is the root or first task and has no dependencies.


The DAGFactory currently supports following operators:
    1. `BashOperator <https://airflow.apache.org/docs/apache-airflow/1.10.13/_api/airflow/operators/bash_operator/index.html)>`_
    2. `BranchPythonOperator <https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.BranchPythonOperator>`_
    3. `DummyOperator <https://airflow.apache.org/docs/apache-airflow/1.10.12/_api/airflow/operators/dummy_operator/index.html>`_
    4. `KubernetedPodOperator <https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/_api/airflow/providers/cncf/kubernetes/operators/kubernetes_pod/index.html#airflow.providers.cncf.kubernetes.operators.kubernetes_pod.KubernetesPodOperator>`_
    5. `PostgresOperator <https://airflow.apache.org/docs/apache-airflow-providers-postgres/stable/_api/airflow/providers/postgres/operators/postgres/index.html>`_
    6. `PythonOperator <https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.PythonOperator>`_
    7. `ShortCircuitOperator <https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/python/index.html#airflow.operators.python.ShortCircuitOperator>`_
    8. `SparkSubmitOperator <https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/stable/_api/airflow/providers/apache/spark/operators/spark_submit/index.html>`_
    
    
Pull Request Policy
*******************
Whenever the files the for a particular use-case are updated, the name of **use-case folder** should be added in front of the pull request message. For instance, if one has updated requirement.txt in qsr's *usecase/folder* commit message can be: ``qsr-updated requirement.txt``. 

.. info::
    As a standard just pass ``usecase/folder-<your commit message here>``.
