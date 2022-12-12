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
... ovr.fit(train_df)
... prd_ovr = ovr.predict(test_df)


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