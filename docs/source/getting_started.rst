Getting Started
===============

This guide will help you get started with Max.AI Data Science Core, from installation to running your first machine learning pipeline.

Prerequisites
-------------

Before installing Max.AI DS Core, ensure you have the following prerequisites:

**System Requirements**
    * Python 3.6 or higher
    * Java 8 or 11 (required for Apache Spark)
    * Minimum 8GB RAM (16GB recommended)
    * 10GB free disk space

**Required Services** (for full functionality)
    * Apache Spark 3.1.2 cluster
    * PostgreSQL database (for metadata storage)
    * Redis (for caching and vector storage)
    * S3-compatible storage (AWS S3, MinIO, etc.)

**Optional Services**
    * Kubernetes cluster (for production deployment)
    * MLflow tracking server
    * Feast feature store
    * Milvus vector database

Installation
------------

**Option 1: Wheel Installation (Recommended)**

.. code-block:: bash

    # Download the wheel package
    wget https://releases.maxai.com/maxai-1.0.0-py3-none-any.whl
    
    # Install the package
    pip install maxai-1.0.0-py3-none-any.whl

**Option 2: Development Installation**

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-org/max.ai.ds.core.git
    cd max.ai.ds.core
    
    # Install dependencies
    pip install -r requirement.txt
    
    # Install in development mode
    pip install -e .

**Option 3: Docker Installation**

.. code-block:: bash

    # Pull the Docker image
    docker pull maxai/ds-core:latest
    
    # Run the container
    docker run -it -p 8888:8888 maxai/ds-core:latest

Environment Setup
-----------------

**Environment Variables**

Set the following environment variables for proper functionality:

.. code-block:: bash

    # AWS Configuration (if using S3)
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export MAX_STORAGE_BUCKET=your_s3_bucket
    
    # Environment
    export ENV=dev  # or prod
    
    # Spark Configuration
    export SPARK_HOME=/path/to/spark
    export PYSPARK_PYTHON=python3
    
    # Database Configuration
    export POSTGRES_HOST=localhost
    export POSTGRES_PORT=5432
    export POSTGRES_DB=maxai
    export POSTGRES_USER=maxai_user
    export POSTGRES_PASSWORD=your_password
    
    # Redis Configuration
    export REDIS_HOST=localhost
    export REDIS_PORT=6379

**Spark Configuration**

Create a Spark configuration file or set environment variables:

.. code-block:: bash

    # For local development
    export SPARK_CONF_DIR=/path/to/spark/conf
    
    # For Kubernetes deployment
    export SPARK_KUBERNETES_NAMESPACE=maxai
    export SPARK_KUBERNETES_CONTAINER_IMAGE=maxai/spark:latest

Quick Start Tutorial
--------------------

Let's walk through a complete machine learning pipeline using Max.AI DS Core.

**Step 1: Data Loading and Preprocessing**

.. code-block:: python

    from maxaidatahandling.dataset import MaxDataset
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("MaxAI_QuickStart") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Configure data source
    data_config = {
        "port": 1,
        "dataType": "dataframe",
        "sourceDetails": {
            "source": "s3",
            "fileFormat": "csv",
            "filePath": "s3://your-bucket/data/customer_data.csv"
        },
        "preprocess": {
            "rename_cols": {"customer_id": "id", "purchase_amount": "amount"},
            "select_cols": ["id", "age", "income", "amount", "category"],
            "data_analysis": {
                "sample_ratio": 0.3,
                "col_types": {
                    "numerical_cols": ["age", "income", "amount"],
                    "categorical_cols": ["category"],
                    "unique_identifier_cols": ["id"]
                }
            },
            "cache": True
        }
    }
    
    # Load and preprocess data
    dataset = MaxDataset(name="customer_data", dataset_config=data_config)
    dataset.prepare_dataset()
    df = dataset.df
    
    print(f"Loaded {df.count()} records with {len(df.columns)} columns")

**Step 2: Data Quality Analysis**

.. code-block:: python

    from maxairesources.datachecks.dataframe_analysis_spark import SparkDataFrameAnalyser
    
    # Define column types for analysis
    col_types = {
        "numerical_cols": ["age", "income", "amount"],
        "categorical_cols": ["category"],
        "unique_identifier_cols": ["id"]
    }
    
    # Analyze data quality
    analyzer = SparkDataFrameAnalyser(df=df, column_types=col_types)
    report = analyzer.generate_data_health_report()
    
    # Save analysis report
    analyzer.save_analysis_report(report, output_path="./data_quality_report.html")
    print("Data quality report saved to data_quality_report.html")

**Step 3: Feature Engineering**

.. code-block:: python

    from maxairesources.pipeline.spark_pipeline import SparkPipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    
    # Define feature engineering pipeline
    stages = {
        'VectorAssembler': {
            'inputCols': ['age', 'income', 'amount'], 
            'outputCol': 'features_raw'
        },
        'StandardScaler': {
            'inputCol': 'features_raw',
            'outputCol': 'features',
            'withStd': True,
            'withMean': True
        }
    }
    
    # Build and apply pipeline
    pipeline = SparkPipeline(stages=stages)
    pipeline.fit_pipeline(df)
    transformed_df = pipeline.transform_pipeline(df)
    
    print("Feature engineering completed")

**Step 4: Train-Test Split**

.. code-block:: python

    # Split data into training and testing sets
    train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training set: {train_df.count()} records")
    print(f"Test set: {test_df.count()} records")

**Step 5: Multi-Model Training**

.. code-block:: python

    from maxairesources.utilities.multi_train import MultiTrain
    
    # Define multiple models to train
    models = {
        "SparkGBTClassifier": {
            "target_col": "category",
            "feature_col": "features",
            "params": {
                "maxIter": 10,
                "maxDepth": 5,
                "stepSize": 0.1
            }
        },
        "SparkRFClassifier": {
            "target_col": "category",
            "feature_col": "features",
            "params": {
                "maxDepth": 5,
                "numTrees": 20,
                "subsamplingRate": 0.8
            }
        },
        "SparkLogisticRegression": {
            "target_col": "category",
            "feature_col": "features",
            "params": {
                "maxIter": 100,
                "regParam": 0.01
            }
        }
    }
    
    # Train models in parallel
    multi_trainer = MultiTrain(models)
    multi_trainer.train_models(train_df)
    
    print("Multi-model training completed")
    print(f"Trained models: {list(multi_trainer.trained_models.keys())}")

**Step 6: Model Evaluation**

.. code-block:: python

    from maxairesources.eval.classifier_evaluator_spark import ClassifierEvaluator
    
    # Evaluate each model
    evaluation_results = {}
    
    for model_name, model in multi_trainer.trained_models.items():
        # Make predictions
        predictions = model.transform(test_df)
        
        # Evaluate model
        evaluator = ClassifierEvaluator(
            predicted_actual_pdf=predictions.toPandas(),
            predicted_col="prediction",
            label_col="category",
            classification_mode="multiclass"
        )
        
        metrics = evaluator.evaluate()
        evaluation_results[model_name] = metrics
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

**Step 7: Model Ensemble**

.. code-block:: python

    from maxairesources.ensemble.ensemble import Ensemble
    
    # Create ensemble from trained models
    model_list = list(multi_trainer.trained_models.values())
    ensemble = Ensemble(model_list)
    
    # Hard voting ensemble
    hard_predictions = ensemble.VotingClassifier(test_df, method="hard")
    
    # Soft voting ensemble with custom weights
    soft_predictions = ensemble.VotingClassifier(
        test_df, 
        method="soft", 
        weight=[0.4, 0.4, 0.2]  # Weights for GBT, RF, LogReg
    )
    
    # Evaluate ensemble
    ensemble_evaluator = ClassifierEvaluator(
        predicted_actual_pdf=soft_predictions.toPandas(),
        predicted_col="prediction",
        label_col="category",
        classification_mode="multiclass"
    )
    
    ensemble_metrics = ensemble_evaluator.evaluate()
    print(f"\nEnsemble Results:")
    print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {ensemble_metrics['f1_score']:.4f}")

**Step 8: Model Approval and Export**

.. code-block:: python

    from maxairesources.model_approval.model_approver_spark import ModelApprover
    
    # Select best performing model
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda k: evaluation_results[k]['f1_score'])
    best_model = multi_trainer.trained_models[best_model_name]
    
    print(f"Best model: {best_model_name}")
    
    # Model approval workflow
    approver = ModelApprover(
        model=best_model,
        evaluator_class=ClassifierEvaluator,
        predicted_actual_pdf=predictions.toPandas(),
        metric_thresholds={
            "f1_score": 0.7,
            "accuracy": 0.75
        },
        predicted_col="prediction",
        label_col="category",
        classification_mode="multiclass"
    )
    
    # Check if model meets approval criteria
    is_approved, approval_details = approver.is_above_threshold()
    
    if is_approved:
        print("✅ Model approved for deployment!")
        
        # Export model to ONNX format
        model_path = f"s3://your-bucket/models/{best_model_name}"
        best_model.save(model_path, spark, mode='onnx')
        print(f"Model exported to: {model_path}")
    else:
        print("❌ Model did not meet approval criteria")
        print(f"Approval details: {approval_details}")

**Step 9: Experiment Tracking**

.. code-block:: python

    from maxaimetadata.metadata import WorkFlow, Run, Execution
    
    # Create workflow for experiment tracking
    workflow = WorkFlow(
        name='Customer_Segmentation_Model',
        description='Multi-model customer segmentation pipeline',
        reuse_workflow_if_exists=True
    )
    
    # Create run for this experiment
    run = Run(
        workflow=workflow, 
        description='Quick start tutorial run v1.0'
    )
    run.update_status('completed')
    
    # Log execution details
    execution = Execution(
        name='Multi_Model_Training',
        workflow=workflow,
        run=run,
        description=f'Trained {len(models)} models with best F1: {max(evaluation_results.values(), key=lambda x: x["f1_score"])["f1_score"]:.4f}'
    )
    
    print("Experiment logged to MLflow")

**Step 10: Cleanup**

.. code-block:: python

    # Stop Spark session
    spark.stop()
    print("Pipeline completed successfully!")

Advanced Examples
-----------------

**Hyperparameter Optimization**

.. code-block:: python

    from maxaimodel.optimization.optimizer import Optimizer
    
    # Define parameter grid for optimization
    param_grid = {
        'maxIter': [10, 20, 50],
        'maxDepth': [3, 5, 7],
        'stepSize': [0.1, 0.01, 0.001]
    }
    
    # Optimize hyperparameters
    optimizer = Optimizer(
        model_class=SparkGBTClassifier,
        param_grid=param_grid,
        optimization_metric='f1',
        cv_folds=3
    )
    
    best_model = optimizer.optimize(train_df, test_df)
    print(f"Best parameters: {optimizer.best_params}")

**Data Drift Monitoring**

.. code-block:: python

    from maxaimonitoring.data_drift.data_drift_checker import DataDriftChecker
    
    # Simulate new data for drift detection
    new_data = test_df.sample(0.5, seed=123)
    
    # Check for data drift
    drift_checker = DataDriftChecker(
        reference_data=train_df,
        current_data=new_data,
        drift_threshold=0.05
    )
    
    drift_report = drift_checker.detect_drift()
    
    if drift_report['drift_detected']:
        print("⚠️  Data drift detected - consider model retraining")
        print(f"Drift score: {drift_report['drift_score']:.4f}")
    else:
        print("✅ No significant data drift detected")

**Feature Store Integration**

.. code-block:: python

    from feast import FeatureStore
    import pandas as pd
    
    # Initialize feature store (requires Feast setup)
    store = FeatureStore(repo_path="./feast_config")
    
    # Define entity dataframe
    entity_df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "event_timestamp": pd.to_datetime("2023-01-01")
    })
    
    # Get historical features
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer:age",
            "customer:income",
            "customer:category"
        ]
    ).to_spark_df()
    
    print("Features retrieved from feature store")

Best Practices
--------------

**Development Workflow**
    1. Start with data exploration and quality analysis
    2. Implement feature engineering incrementally
    3. Use cross-validation for model selection
    4. Track all experiments with MLflow
    5. Implement proper error handling and logging
    6. Test with small datasets before scaling up

**Performance Optimization**
    * Use appropriate Spark configurations for your cluster size
    * Cache frequently accessed DataFrames
    * Optimize data partitioning for your use case
    * Monitor resource usage and adjust accordingly
    * Use broadcast variables for small lookup tables

**Production Deployment**
    * Implement proper model versioning
    * Set up monitoring and alerting
    * Use A/B testing for model rollouts
    * Implement rollback procedures
    * Monitor data drift and model performance

**Security Considerations**
    * Use secure credential management (Vault, AWS Secrets Manager)
    * Implement proper access controls
    * Encrypt sensitive data at rest and in transit
    * Regular security audits and updates
    * Follow principle of least privilege

Troubleshooting
---------------

**Common Issues**

**Spark Session Issues**
    * Ensure Java is properly installed and JAVA_HOME is set
    * Check Spark configuration and cluster connectivity
    * Verify memory and executor settings

**Data Loading Issues**
    * Verify S3 credentials and bucket permissions
    * Check file paths and formats
    * Ensure proper network connectivity

**Memory Issues**
    * Increase Spark executor memory
    * Optimize data partitioning
    * Use data sampling for development
    * Enable dynamic allocation

**Model Training Issues**
    * Check data quality and preprocessing
    * Verify feature engineering pipeline
    * Monitor convergence and adjust hyperparameters
    * Use appropriate evaluation metrics

Next Steps
----------

Now that you've completed the quick start tutorial, explore these advanced topics:

* **Advanced Feature Engineering**: Learn about time-series features and automated feature selection
* **Model Optimization**: Dive deeper into hyperparameter tuning and AutoML
* **Production Deployment**: Set up Kubernetes deployment and monitoring
* **LLM Integration**: Explore document processing and fine-tuning capabilities
* **Custom Extensions**: Build custom models and evaluators

For more detailed information, refer to the specific module documentation in this guide.

Happy machine learning with Max.AI DS Core! 🚀
