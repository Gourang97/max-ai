API Reference
=============

This section provides comprehensive API documentation for all Max.AI DS Core modules.

Core Modules
------------

.. autosummary::
   :toctree: generated
   :recursive:

   maxaibase
   maxaidatahandling
   maxaifeaturization
   maxaimodel
   maxairesources
   maxaimetadata
   maxaimonitoring
   maxaillm

Base Classes and Interfaces
---------------------------

.. autosummary::
   :toctree: generated

   maxaibase.data_connector.data_connector_interface
   maxaibase.model.model_base
   maxaibase.evaluation.evaluator_base
   maxaibase.featurization.featurization_base
   maxaibase.ensemble.ensemble_base
   maxaibase.optimization.optimizer_base

Data Handling
-------------

.. autosummary::
   :toctree: generated

   maxaidatahandling.dataset.MaxDataset
   maxaidatahandling.datafactory.MAXDataFactory
   maxaidatahandling.data_evaluator.DataEvaluator
   maxaidatahandling.data_expectations.MaxExpectations

Feature Engineering
-------------------

.. autosummary::
   :toctree: generated

   maxaifeaturization.aggregation.aggregation.Aggregation
   maxaifeaturization.featuretools.featuretools.FeatureGenerator
   maxaifeaturization.transformation.transform.Transformation
   maxaifeaturization.transformation.window.WindowOperations
   maxaifeaturization.timeseries.univariate
   maxaifeaturization.selection.selector
   maxaifeaturization.decomposition.pca

Model Training and Optimization
-------------------------------

.. autosummary::
   :toctree: generated

   maxaimodel.optimization.optimizer.Optimizer
   maxaimodel.H2O
   maxaimodel.spark
   maxaimodel.python

Resources and Utilities
-----------------------

.. autosummary::
   :toctree: generated

   maxairesources.ensemble.ensemble.Ensemble
   maxairesources.eval.classifier_evaluator_spark.ClassifierEvaluator
   maxairesources.eval.regressor_evaluator_spark.RegressorEvaluator
   maxairesources.eval.model_evaluator.ModelEvaluator
   maxairesources.eval.model_explainer.ModelExplainer
   maxairesources.datachecks.dataframe_analysis_spark.SparkDataFrameAnalyser
   maxairesources.utilities
   maxairesources.pipeline
   maxairesources.model_approval
   maxairesources.optimizer
   maxairesources.logging

Metadata and Experiment Tracking
--------------------------------

.. autosummary::
   :toctree: generated

   maxaimetadata.maxflow
   maxaimetadata.utils.models
   maxaimetadata.utils.runs
   maxaimetadata.utils.email
   maxaimetadata.utils.prometheus

Monitoring and Drift Detection
------------------------------

.. autosummary::
   :toctree: generated

   maxaimonitoring.data_drift.data_drift_checker.DataDriftChecker
   maxaimonitoring.model_drift.model_drift_checker.ModelDriftChecker

Large Language Models
---------------------

.. autosummary::
   :toctree: generated

   maxaillm.data.extractor.MaxExtractor
   maxaillm.data.extractor.MaxPDFExtractor
   maxaillm.data.extractor.MaxDOCExtractor
   maxaillm.data.extractor.MaxPPTExtractor
   maxaillm.data.extractor.MaxHTMLExtractor
   maxaillm.data.extractor.MaxMDExtractor
   maxaillm.dev.finetune.base
   maxaillm.dev.finetune.interface
   maxaillm.dev.finetune.providers.hugging_face
   maxaillm.dev.finetune.tune_method.peft

Quick Reference
---------------

**Data Loading**

.. code-block:: python

   from maxaidatahandling.dataset import MaxDataset
   
   dataset = MaxDataset(name="data", dataset_config=config)
   dataset.prepare_dataset()

**Feature Engineering**

.. code-block:: python

   from maxaifeaturization.aggregation.aggregation import Aggregation
   
   agg = Aggregation(df=dataframe, arguments=agg_config)
   result = agg.execute()

**Model Training**

.. code-block:: python

   from maxairesources.utilities.multi_train import MultiTrain
   
   trainer = MultiTrain(models_config)
   trainer.train_models(training_data)

**Model Evaluation**

.. code-block:: python

   from maxairesources.eval.classifier_evaluator_spark import ClassifierEvaluator
   
   evaluator = ClassifierEvaluator(predictions, "prediction", "label")
   metrics = evaluator.evaluate()

**Ensemble Methods**

.. code-block:: python

   from maxairesources.ensemble.ensemble import Ensemble
   
   ensemble = Ensemble(model_list)
   predictions = ensemble.VotingClassifier(test_data)

**Data Drift Detection**

.. code-block:: python

   from maxaimonitoring.data_drift.data_drift_checker import DataDriftChecker
   
   checker = DataDriftChecker(reference_data, current_data)
   drift_report = checker.detect_drift()

**Document Processing**

.. code-block:: python

   from maxaillm.data.extractor.MaxExtractor import MaxExtractor
   
   extractor = MaxExtractor()
   text, metadata = extractor.extract_text_metadata("document.pdf")

**Fine-Tuning**

.. code-block:: python

   from maxaillm.dev.finetune.providers.hugging_face import MaxHuggingFaceFineTuning
   
   fine_tuner = MaxHuggingFaceFineTuning(config)
   model = fine_tuner.train(train_dataset, eval_dataset)

Configuration Examples
----------------------

**Data Configuration**

.. code-block:: python

   data_config = {
       "port": 1,
       "dataType": "dataframe",
       "sourceDetails": {
           "source": "s3",
           "fileFormat": "csv",
           "filePath": "s3://bucket/data.csv"
       },
       "preprocess": {
           "rename_cols": {"old_name": "new_name"},
           "select_cols": ["col1", "col2", "col3"],
           "cache": True
       }
   }

**Model Configuration**

.. code-block:: python

   models_config = {
       "SparkGBTClassifier": {
           "target_col": "label",
           "feature_col": "features",
           "params": {
               "maxIter": 10,
               "maxDepth": 5
           }
       }
   }

**Pipeline Configuration**

.. code-block:: python

   pipeline_stages = {
       'VectorAssembler': {
           'inputCols': ['feature1', 'feature2'],
           'outputCol': 'features'
       },
       'StandardScaler': {
           'inputCol': 'features',
           'outputCol': 'scaled_features'
       }
   }

Error Handling
--------------

All Max.AI DS Core modules include comprehensive error handling:

.. code-block:: python

   try:
       dataset = MaxDataset(name="data", dataset_config=config)
       dataset.prepare_dataset()
   except FileNotFoundError:
       print("Data file not found")
   except ValueError as e:
       print(f"Configuration error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

For detailed API documentation of specific modules, please refer to the individual module documentation pages.
