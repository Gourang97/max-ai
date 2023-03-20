maxaimodel
==========


H2O
########

.. note::
    
    Implementaion comes with MaxFlow Autologging Capabilities. Models considers all the columns except ``target`` columns as Features. 
    

Classification
**************


H2ODLClassifier
^^^^^^^^^^^^^^^
This class provide Max.AI wrapper for H2O PySparkling Deeplearning Classifier. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to None.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to None. If ``param_grid`` is not passed, optimizations will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model
        
Raises:
    - ``ModelBuildException``
    
>>> from maxaimodel.H2O.classification.h2o_dl import H2ODLClassifier
>>> model = H2ODLClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2ODRFClassifier
^^^^^^^^^^^^^^^^

This class provide Max.ai Wrapper for H2O Pysparkling H2ODRF Classifier. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to None. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model


Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.classification.h2o_drf import H2ODRFClassifier
>>> model = H2ODRFClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OGAMClassifier
^^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OGAM Classifier. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `<here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to None.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type `{"param": []}`. If `param_grid` is mentioned, it will over-write `params`. Defaults to None. If ``param_grid`` is not passed, optimizations will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.classification.h2o_gam import H2OGAMClassifier
>>> model = H2OGAMClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OGBMClassifier
^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OGBM Classifier. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type `{"param": []}`. If `param_grid` is mentioned, it will over-write `params`. Defaults to None. If ``param_grid`` is not passed, optimizations will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.classification.h2o_gbm import H2OGBMClassifier
>>> model = H2OGBMClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)

H2OGLMClassifier
^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling Deeplearning Classifier. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to None.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type `{"param": []}`. If `param_grid` is mentioned, it will over-write `params`. Defaults to None. If ``param_grid`` is not passed, optimizations will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.classification.h2o_glm import H2OGLMClassifier
>>> model = H2OGLMClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)

H2ORFClassifier
^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2ORuleFit Classifier. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to None.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimizations will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.classification.h2o_rf import H2ORFClassifier
>>> model = H2ORFClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OXGBClassifier
^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OXGBoost Classifier. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type `{"param": []}`. If `param_grid` is mentioned, it will over-write `params`. Defaults to None. If ``param_grid`` is not passed, optimizations will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> # Code Example Block
>>> from maxaimodel.H2O.classification.h2o_xgb import H2OXGBClassifier
>>> model = H2OXGBClassifier(
...     target_col='Survived',
...     params={'param': value}
...     param_grid={
...         "param": ['value1','value2'],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


Clustering
**********

KMeans
^^^^^^^^
This class provide implementation to run Spark GaussianMixture Clustering on the data.

Args:
    - ``k (int)`` - no of clusters required. Default to ``0``. If ``0`` model will try to find the optimal k value for the data.
    - ``k_max (int)`` - max k value to consider for running optimization. Required only if ``k`` is ``0``
    - ``params (dict, optional)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/3.0/latest-stable/doc/parameters/parameters_H2OKMeans.html>`_

>>> from maxaimodel.H2O.clustering.h2o_kmeans import H2OKmeansClustering
>>> model = H2OKmeansClustering(k=10)
>>> model.fit(data)
>>> predicition = model.predict(data)


Regression
**********

H2ODLRegressor
^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling Deeplearning Regressor. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to None.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.regression.h2o_dl import H2ODLRegressor
>>> model = H2ODLRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2ODRFRegressor
^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2ODRF Regressor. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> # Code Example Block
>>> from maxaimodel.H2O.regression.h2o_drf import H2ODRFRegressor
>>> model = H2ODRFRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OGAMRegressor
^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OGAM Regressor. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.regression.h2o_gam import H2OGAMRegressor
>>> model = H2OGAMRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OGBMRegressor
^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OGBM Regressor. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.regression.h2o_gbm import H2OGBMRegressor
>>> model = H2OGBMRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)

H2OGLMRegressor
^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OGLM Regressor. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to None.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.regression.h2o_glm import H2OGLMRegressor
>>> model = H2OGLMRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2ORULEFITRegressor
^^^^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2ORuleFit Regressor. Provides functionalities to train model on data and make prediction.


Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.regression.h2o_rulefit import H2ORULEFITRegressor
>>> model = H2ORULEFITRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OXGBRegressor
^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OXGBoost Regressor. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.regression.h2o_xgb import H2OXGBRegressor
>>> model = H2OXGBRegressor(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


Unsupervised
************

H2OIsolationForest
^^^^^^^^^^^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OIsolationForest for Anomaly Detection. Provides fucntionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` - dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``. If ``param_grid`` is not passed, optimization will be disabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.unsupervised.h2o_isolation_forest import H2OIsolationForest
>>> model = H2OIsolationForest(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


H2OPCA
^^^^^^^^
This class provide Max.ai Wrapper for H2O Pysparkling H2OPCA for Dimensionality Reduction. Provides functionalities to train model on data and make prediction.

Args:
    - ``target_col (str)`` - model which needs to be optimized
    - ``params (list)`` - dictionary of parameters described `here <https://docs.h2o.ai/sparkling-water/2.3/latest-stable/doc/parameters/parameters.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)`` -dictionary of parameters of type `{"param": []}`. If `param_grid` is mentioned, it will over-write `params`. Defaults to None. If param_grid is not passed, optimization will be desabled.

Returns (or Yields):
    - ``model`` - Trained Model

Raises:
    - ``ModelBuildException``

>>> from maxaimodel.H2O.unsupervised.h2o_pca import H2OPCA
>>> model = H2OPCA(
...     target_col="SalePrice",
...     params={"param": "value"},
...     param_grid={
...          "param": ["value1", "value2"],
...     }
... )
>>> model.fit(train)
>>> model.predict(test)


Spark
########

Classification
**************

SparkDecisionTreeClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^
runs Decision Tree classifier on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
    
>>> from maxaimodel.spark.classification import spark_dt
>>> dt = spark_dt.SparkDecisionTreeClassifier(
... target_col='Survived',
... feature_col='features',
... param_grid={
...    "impurity": ['gini','entropy'],
...    "maxDepth": [3,5]
...    }
...  )
>>> dt.fit(train)
>>> pred = dt.predict(test)


SparkFMClassifier
^^^^^^^^^^^^^^^^^
Factorization Machines learning algorithm for classification.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.FMClassifier.html>`_. Defaults to None.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to None.
    
>>> from maxaimodel.spark.classification import spark_fm
>>> fmc = spark_fmc.SparkFMClassifier(
...    target_col='Survived',
...    feature_col='features'
... )
>>> fmc.fit(train)
>>> pred = fmc.predict(test)
>>> fmc.save(path="./models/fmc")


SparkGBTClassifier
^^^^^^^^^^^^^^^^^^
runs gradient boosted classifier on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
    
>>> from maxaimodel.spark.classification import spark_gbt
>>> gbt = spark_dt.SparkGBTClassifier(
...     target_col='Survived',
...     feature_col='features',
...     param_grid={}
... )
>>> gbt.fit(train)
>>> pred = gbt.predict(test)


SparkLogisticRegression
^^^^^^^^^^^^^^^^^^^^^^^
runs logistic regression on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.classification import spark_lr
>>> lr = spark_lr.SparkLogisticRegression(
...     target_col='Survived',
...     feature_col='features'
... )
>>> lr.fit(train_df)
>>> lr_pred = lr.predict(test_df)
>>> lr.save(path="./models/lr")


SparkMultilayerPerceptronClassifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Classifier trainer based on the Multilayer Perceptron.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.MultilayerPerceptronClassifier.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
    
>>> from maxaimodel.spark.classification import spark_mlp
>>> mlp = spark_mlp.MultilayerPerceptronClassifier(
...     target_col='Survived',
...     feature_col='features'
... )
>>> mlp.fit(train_df)
>>> mlp_pred = mlp.predict(test_df)
>>> mlp.save(path="./models/mlp")


SparkNaiveBayes
^^^^^^^^^^^^^^^
runs naive bayes on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.NaiveBayes.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.classification import spark_nb
>>> nb = spark_nb.SparkNaiveBayes(
...     target_col='Survived',
...     feature_col='features'
... )
>>> nb.fit(train_df)
>>> nb_pred = nb.predict(test_df)
>>> nb.save(path="./models/nb")


SparkOneVsRest
^^^^^^^^^^^^^^
Reduction of Multiclass Classification to Binary Classification. Performs reduction using one against all strategy.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``classifier (maxaibase.model.spark.spark_classifier.SparkClassifierBaseModel)``: base model on which OneVsRest is to be run
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.OneVsRest.html>`_. Defaults to None.
    - ``param_grid (dict, optional)``: dictionary of parameters of type `{"param": []}`. If `param_grid` is mentioned, it will over-write `params`. Defaults to None.
    
>>> from maxaimodel.spark.classification import spark_ovr, spark_lr
>>> lr = spark_lr.SparkLogisticRegression(target_col="Species", feature_col="label")    # declare the classifier that would act as base classifier for OneVsRest
>>> ovr = spark_ovr.SparkOneVsRest(
...    target_col="label",
...    feature_col="features",
...    param_grid={},
...    classifier=lr
... )
>>> ovr.fit(train_df)
>>> pred_df = ovr.predict(test_df)


SparkRFClassifier
^^^^^^^^^^^^^^^^^
runs random forest on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.classification import spark_rf
>>> rf = spark_rf.SparkNaiveBayes(
...     target_col='Survived',
...     feature_col='features'
... )
>>> rf.fit(train_df)
>>> rf_pred = rf.predict(test_df)
>>> rf.save(path="./models/rf")


SparkLinearSVC
^^^^^^^^^^^^^^
runs support vectors on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LinearSVC.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
    
>>> from maxaimodel.spark.classification import spark_svc
>>> svc = spark_svc.SparkNaiveBayes(
...     target_col='Survived',
...     feature_col='features'
... )
>>> svc.fit(train_df)
>>> svc_pred = svc.predict(test_df)
>>> svc.save(path="./models/svc")


Clustering
**********

SparkBisectKmeansClustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This class provide implementation to run Spark Bisecting Kmeans Clustering on the data.

Args:
    - ``k (int)`` - no of clusters required. Default to 0. If 0 model will try to find the optimal k value for the data.
    - ``k_min (int)`` - min k value to consider for running optimization. Required only if k is 0
    - ``k_max (int)`` - max k value to consider for running optimization. Required only if k is 0
    - ``feature_col (string)`` - Column name which has the vectorized features. Default to features
    - ``params (dict, optional)`` - dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.BisectingKMeans.html>`_

>>> from maxaimodel.spark.clustering.spark_bisecting_kmeans import SparkBisectKmeansClustering
>>> model = SparkBisectKmeansClustering(k=10)
>>> model.fit(data)
>>> predicition = model.predict(data)

SparkGMMClustering
^^^^^^^^^^^^^^^^^^
This class provide implementation to run Spark GaussianMixture Clustering on the data.

Args:
    - ``k (int)`` - no of clusters required. Default to 0. If 0 model will try to find the optimal k value for the data.
    - ``k_min (int)`` - min k value to consider for running optimization. Required only if k is 0
    - ``k_max (int)`` - max k value to consider for running optimization. Required only if k is 0
    - ``feature_col (string)`` - Column name which has the vectorized features. Default to features
    - ``params (dict, optional)`` - dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.GaussianMixture.html>`_

Raises:
    - ````

>>> from maxaimodel.spark.clustering.spark_gmm import SparkGMMClustering
>>> model = SparkGMMClustering(k=10)
>>> model.fit(data)
>>> predicition = model.predict(data)


SparkKMeansClustering
^^^^^^^^^^^^^^^^^^^^^
This class provide implementation to run Spark GaussianMixture Clustering on the data.

Args:
    - ``k (int)`` - no of clusters required. Default to 0. If 0 model will try to find the optimal k value for the data.
    - ``k_min (int)`` - min k value to consider for running optimization. Required only if k is 0
    - ``k_max (int)`` - max k value to consider for running optimization. Required only if k is 0
    - ``feature_col (string)`` - Column name which has the vectorized features. Default to features
    - ``params (dict, optional)`` - dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html>`_

Raises:
    - ````

>>> from maxaimodel.spark.clustering.spark_kmeans import SparkKMeansClustering
>>> model = SparkKMeansClustering(k=10)
>>> model.fit(data)
>>> predicition = model.predict(data)


SparkLDAClustering
^^^^^^^^^^^^^^^^^^
This class provide implementation to run Spark GaussianMixture Clustering on the data.

Args:
    - ``k (int)`` - no of clusters required. Default to 0. If 0 model will try to find the optimal k value for the data.
    - ``feature_col (string)`` - Column name which has the vectorized features. Default to features
    - ``params (dict, optional)`` - dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.LDA.html>`_

Raises:
    - ````

>>> from maxaimodel.spark.clustering.spark_lda import SparkLDAClustering
>>> model = SparkLDAClustering(k=10)
>>> model.fit(data)
>>> predicition = model.predict(data)

SparkPICClustering
^^^^^^^^^^^^^^^^^^
This class provide implementation to run Spark PowerIterationClustering Clustering on the data.

Args:
    - ``k (int)`` - no of clusters required. Default to 0. If 0 model will try to find the optimal k value for the data.
    - ``k_min (int)`` - min k value to consider for running optimization. Required only if k is 0
    - ``k_max (int)`` - max k value to consider for running optimization. Required only if k is 0
    - ``src_col (str)`` - source column name. Default to 'src'
    - ``dst_col (str)`` - destination column name. Default to 'dst'
    - ``params (dict, optional)`` - dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.PowerIterationClustering.html>`_

Raises:
    - ````

>>> from maxaimodel.spark.clustering.spark_pic import SparkPICClustering
>>> model = SparkPICClustering(k=10)
>>> model.fit(data)
>>> predicition = model.predict(data)


Regression
**********

SparkAFTSurvivalRegressor
^^^^^^^^^^^^^^^^^^^^^^^^^
runs FM regression on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.FMRegressionModel.html?highlight=regression#pyspark.ml.regression.FMRegressionModel>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.regression import spark_aft_survival_regression
>>> aftsurvivalreg = spark_aft_survival_regression.SparkAFTSurvivalRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> aftsurvivalreg.fit(train_aft)
>>> pred_aftreg = aftsurvivalreg.predict(test_aft)
>>> aftsurvivalreg.save(path="./models/aftsurvivalreg")


SparkDTRegressor
^^^^^^^^^^^^^^^^
runs Decision Tree regression on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.

>>> from maxaimodel.spark.regression import spark_dt_regression
>>> dt = spark_dt_regression.SparkDTRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> dt.fit(train)
>>> pred = dt.predict(test)


SparkFMRegressor
^^^^^^^^^^^^^^^^
runs Factorization Machines learning algorithm for regression on data and gives predictions

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.FMRegressionModel.html?highlight=regression#pyspark.ml.regression.FMRegressionModel>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.regression import spark_fm_regression
>>> fmreg = spark_fm_regression.SparkFMRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> fmreg.fit(train)
>>> pred_fmreg = fmreg.predict(test)


SparkGBTRegressor
^^^^^^^^^^^^^^^^^
runs Gradient Boosted regression on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GBTRegressionModel.html?highlight=regression#pyspark.ml.regression.GBTRegressionModel>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.regression import spark_gbt_regression
>>> gbt = spark_gbt_regression.SparkGBTRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> gbt.fit(train)
>>> pred_gbt = gbt.predict(test)


SparkGLRegressor
^^^^^^^^^^^^^^^^
runs Generalized Linear regression on data and gives predictions.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GeneralizedLinearRegression.html?highlight=regression#pyspark.ml.regression.GeneralizedLinearRegression>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.

>>> from maxaimodel.spark.regression import spark_gl_regression
>>> glinreg = spark_gl_regression.SparkGLRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> glinreg.fit(train)
>>> pred_glinreg = gbt.predict(test)
>>> glinreg.save(path="./models/glinreg")


SparkIsotonicRegressor
^^^^^^^^^^^^^^^^^^^^^^
runs Isotonic regression on data and gives predictions

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.IsotonicRegression.html?highlight=regression#pyspark.ml.regression.IsotonicRegression>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.regression import spark_isotonic_regression
>>> isoreg = spark_isotonic_regression.SparkIsotonicRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> isoreg.fit(train)
>>> pred_isoreg = isoreg.predict(test)
>>> isoreg.save(path="./models/isoreg")


SparkLinearRegressor
^^^^^^^^^^^^^^^^^^^^
runs linear regression on data and gives prediction.

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html?highlight=regression#pyspark.ml.regression.LinearRegression>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.
    
>>> from maxaimodel.spark.regression import spark_linear_regression
>>> linreg = spark_linear_regression.SparkLinearRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> linreg.fit(train)
>>> pred_linreg = gbt.predict(test)


SparkRFRegressor
^^^^^^^^^^^^^^^^
runs Random Forest regression on data and gives predictions

Args:
    - ``target_col (str)``: column to be predicted
    - ``feature_col (str)``: vector of all the features to be used for prediction
    - ``params (dict, optional)``: dictionary of parameters described `here <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressionModel.html?highlight=regression#pyspark.ml.regression.RandomForestRegressionModel>`_. Defaults to ``None``.
    - ``param_grid (dict, optional)``: dictionary of parameters of type ``{"param": []}``. If ``param_grid`` is mentioned, it will over-write ``params``. Defaults to ``None``.

>>> from maxaimodel.spark.regression import spark_rf_regression
>>> rf = spark_rf_regression.SparkRFRegressor(
...     target_col='medv',
...     feature_col='features',
...     param_grid={}
... )
>>> rf.fit(train)
>>> pred = rf.predict(test)


Time-Series
***********

SparkTSForecaster
^^^^^^^^^^^^^^^^^
A PySpark Wrapper to run Python models in the workers.

Args:
    - ``grp_by_col (str)``: column name to distribute training for
    - ``target_col (string)``: time-series column
    - ``base_model (string)``: base model to be used. For instance, ``ARIMA``, ``GARCH``, ``Prophet`` and ``NProphet`` (for Neural Prophet).
    - ``model_params (dict)``: inputs to Python Base Model. For reference, see base_model definition
    

>>> from maxaimodel.spark.timeseries import spark_ts
>>> ts = spark_ts.SparkTSForecaster(
...     grp_by_col='Dept',
...     target_col='Weekly_Sales',
...     base_model='NProphet',
...     model_params={
...         "time_col": "Date",
...         "target_col": "Weekly_Sales",
...         "param_grid": {
...             "yearly_seasonality": [True],
...             "seasonality_mode": ["additive", "multiplicative"],
...             "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1]
...         },
...         "freq": "W",
...         "metric": "mean_squared_error"
...     }
... )
>>> ts.fit(sales_df)
>>> resultant_df = ts.predict(periods=24, freq='W', data=sales_df.select('Date', 'Dept'))


Model Optimization
##################

.. note::

    Currently supports only spark based models

MaxHyperOpt
***********
This class provides the Max.AI Wrapper for HyperOpt optmization engine. Supports the below algoithms for optmization:
- TreeParsenOptimizer
- Adaptive TreeParsenOptimizer
- Random Search
- Mixed Search
- Annealing

For more details on the engine please refer `HyperOpt <http://hyperopt.github.io/hyperopt/>`_

Args:
    - ``model (BaseModel)`` - model which needs to be optimized
    - ``train (SparkDataFrame)`` - training data frame
    - ``test (SparkDataFrame)`` - test data frame
    - ``params (list)`` - list of params to construct search space
    - ``evals (int)`` - no of iteration to run for optimizing the model
    - ``metric (str)`` - metric which needs to be optimized
    - ``algo (str)`` - meta algorithm to use for optimization
    - ``direction (str)`` - whether to maximise/minimise the given metric
    - ``parallelism (str)`` - no of trials to be run in parallel

Returns (or Yields):
    - ``best`` - Dictionary of best parameters
    - ``error`` - best value for the metric

Raises:
    - ``AlgoNotSupported``

>>> from maxairesources.optimization.engines.hyperopt import MaxHyperOpt
>>> from maxairesources.optimization.optimizer import (MaxOptimizer, DiscreteParam,
... IntegerParam, ContinousParam, logParams)
>>> params = [DiscreteParam('maxDepth', [1,2,3]), IntegerParam('numTrees', 1, 3, 1),
...     ContinousParam('minInfoGain', 0, 1, 0.1), logParams('minWeightFractionPerNode', 0, 1, 0.1)]
>>> opt = MaxHyperOpt(model,
...            train
...            test
...            params,
...            evals=10,
...            algo='tpe',
...            metric='accuracy',
...            direction='maximize',
...            engine='optuna',
...            parallelism=3
... )
>>> opt.optimize()


MaxOptuna
***********
This class provides the Max.AI Wrapper for Optuna optmization engine. Supports the below algoithms for optmization:
    1. TreeParsenOptimizer
    2. Grid Search
    3. Random Search
    4. Genetic Algorithm
    5. CMA-ES

For more details on the engine please refer `Optuna <https://optuna.org/>`_.

Args:
    - ``model (BaseModel)`` - model which needs to be optimized
    - ``train (SparkDataFrame)`` - training data frame
    - ``test (SparkDataFrame)`` - test data frame
    - ``params (list)`` - list of params to construct search space
    - ``evals (int)`` - no of iteration to run for optimizing the model
    - ``metric (str)`` - metric which needs to be optimized
    - ``algo (str)`` - meta algorithm to use for optimization
    - ``direction (str)`` - whether to maximise/minimise the given metric
    - ``parallelism (str)`` - no of trials to be run in parallel

Returns (or Yields):
    - ``best`` - Dictionary of best parameters
    - ``error`` - best value for the metric

Raises:
    - ``AlgoNotSupported``

>>> from maxairesources.optimization.engines.optuna import MaxOptuna
>>> from maxairesources.optimization.optimizer import (MaxOptimizer, DiscreteParam,
... IntegerParam, ContinousParam, logParams)
>>> params = [DiscreteParam('maxDepth', [1,2,3]), IntegerParam('numTrees', 1, 3, 1),
...     ContinousParam('minInfoGain', 0, 1, 0.1), logParams('minWeightFractionPerNode', 0, 1, 0.1)]
>>> opt = MaxOptuna(model,
...            train
...            test
...            params,
...            evals=10,
...            algo='tpe',
...            metric='accuracy',
...            direction='maximize',
...            engine='optuna',
...            parallelism=3)
>>> opt.optimize()
