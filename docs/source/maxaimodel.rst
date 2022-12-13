maxaimodel
==========

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
    - ``base_model (string)``: base model to be used. For instance, ``ARIMA``, ``GARCH``, ``Prophet`` and ``NeuralProphet``
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