maxaimetadata
=============

MaxFlow
*******
Max.AI Wrapper class for MLflow Experiment. Provides aditional abstraction for grouping multiple experiments under a project, tracking code changes and commits and seemlessly manage registered models. 

Args:
    - ``uri (str, optional)``: mlflow tracking server uri. Defaults to './mlruns'
    - ``usr (str, optional)``: user name for authenticating mlflow tracking server
    - ``pwd (str, optional)``: password for authenticating mlflow tracking server
    
>>> from maxaimetadata.maxflow import MaxFlow
>>> mf = MaxFlow(uri="", usr="", pwd="")        # create a maxflow instance
>>> mf.set_experiment(
...    experiment="experiment_name", 
...    project=None,
...    repo_path=None
... )                                           # set experiment
>>> run = mf.start_run(name="run_name")         # start a run
>>> run.log_metric(key, value)                  # log metric to run
>>> child_run = run.start_child_run()           # start a child run
>>> child_run.log_param(key, value)             # log params to child run




Methods
^^^^^^^

get_active_flow
^^^^^^^^^^^^^^^
method to get the active MaxFlow instance in the context.

Args:
    - ``None``
    
Returns:
    - ``MaxRun``
    
>>> mf = MaxFlow.get_active_flow()    # get current activeflow in the context


get_active_run
^^^^^^^^^^^^^^^
method to get the current active run in the context.

Args:
    - ``None``
    
Returns:
    - ``MaxRun``
    
>>> run = mf.get_active_run()    # get the current active run in the context


set_experiment
^^^^^^^^^^^^^^
method to set experiment. Reuse existing experiment or creates new one

Args:
    - ``experiment (str)``: name of the experiment.
    - ``project (str, optional)``: name of the project to map this experiment to. Defaults to None.
    - ``repo_path (str, optional)``: local path to the repo which contains the code for the experiment.

Returns:
    - ``None``
    
>>> mf.set_experiment(
...    experiment="experiment_name", 
...    project=None,
...    repo_path=None
... ) 


start_run
^^^^^^^^^
method to start a run of the experiment.

Args:
    - ``name (str)``: name of the run.
    - ``type (str, optional)``: type of the run (eg. classification, regression etc). Defaults to None.
    - ``description (str, optional)``: description of the run. Defaults to None.
    
Returns:
    - ``MaxRun``

resume_run
^^^^^^^^^^
loads already existing (or previously created) ``MaxFlow`` run. To run this, MaxFlow experiment should be set first and the run should exist under that experiment.

Args:
    - ``run_id (str)``: Run ID of the already existing run

Returns:
    - ``MaxRun``
    
>>> from maxaimetadata.maxflow import MaxFlow
>>> mf = MaxFlow(uri="", usr="", pwd="")
>>> mf.set_experiment(experiment="existing_experiment_name", project="", repo_path=".")
>>> run = mf.resume_run(run_id='existing_run12345')
    
auto_log
^^^^^^^^
method to enable mlflow auto logging

Args:
    - ``log_models (Boolean)``: Flag indicating whether to log model or not. Defaults to False.

Returns:
    - ``None``
    
    
register_model
^^^^^^^^^^^^^^
method to register model to the model registry.

Args:
    - ``run (MaxRun, optional)``: run for which logged models needs to be registered. Defaults to None
    - ``recursive (Boolean, optional)``: If set to True, recursively registeres all the models logged to child runs. Defaults to False.
    - ``run_id (str, optional)``: mlflow run_id as an alternative to MaxRun object. Defaults to None.

Returns:
    - ``None``
    
get_registered_model
^^^^^^^^^^^^^^^^^^^^
method to get all the registered models to the current experiment

Args:
    - ``stage (str, optional)``: stage of the registered model. Defaults to None.

Returns:
    - ``None``
    
stop
^^^^
method to stop the MaxFlow instance. Ends all the active runs. This method also have provides a functionality to send e-mail along with the artifacts logged in the current active run (along with all corresponding parent and child runs).

Args:
    - ``artifact_list (list, optional)`` - a list of the artifacts that are to be attached with the email. Filename should be with extension (e.g. ``param_grid.json``)
    - ``to_email (Union[str, list], optional)`` - email-id(s) to whom mail is to be sent.
    - ``email_subject (str, optional)`` - subject line of email

Returns:
    - ``None``

To send without any e-mail notification:

>>> mf.stop()

To send with an e-mail notification, along with artifacts

>>> mf.stop(
...    artifact_list=["model_evaluation.html", "param_grid.json"],
...    to_email=["user1@company.com", "user2@company.com"],
...    email_subject="Subject Line"
... )

.. note::

    The aggregate size of all the artifacts should not exceed 10 MB. If it does, than first *N* attachments with size less than the limit will be sent. For instance, if we have three files, ``file1.json``, ``file2.html`` and ``file3.txt``, weighing in at 1 MB, 8 MB and 3 MB respectively, then first two files will attached with the email, but third one will be ommitted as first two will have size of 9 MB and attaching third file will exceed the pre-defined limit. Hence, it is best to mention the **important** files in the ``artifact_list`` on smaller indices.


MaxRun
******
Max.AI wrapper class for MLflow run. It provides an interface for creating and managing child runs. ``MaxRun`` class can be initialized to use MaxFlow functionalities or create child run method.

Args:
    - ``uri (str)``: MLflow tracking server uri
    - ``exp (mlflow.entities.Experiment)``: MLflow experiment name
    - ``run (mlflow.entities.Run)``: MLflow run object
    
>>> from maxaimetadata.maxflow import MaxFlow
>>> mf = MaxFlow(uri="", usr="", pwd="")        # create a maxflow instance
>>> mf.set_experiment(
...    experiment="experiment_name", 
...    project=None,
...    repo_path=None
... )                                           # set experiment
>>> run = mf.start_run(name="run_name")         # start a run


start_child_run
^^^^^^^^^^^^^^^
Method to start a child run of the currect active run instance.

Args:
    - ``name (str, optional)``: name of the child run. Defaults to None

Returns:
    - ``maxflow.MaxRun``
    

end_run
^^^^^^^
Method to end the current run. it will set the run state as finished in MLflow.

Args:
    - ``None``

Returns:
    - ``None``
    
set_active
^^^^^^^^^^
method to set the current run as active. All the autologging feature will pick the active run for logging.

Args:
    - ``None``

Returns:
    - ``None``
    
>>> run.set_active()    # set a run as active run to the context

log_dict
^^^^^^^^
method to log a dictionary as an MLflow artifact.

Args:
    - ``data (dict)``: dictionary
    - ``file_name (str)``: file name of the artifact which will be logged to MLflow artifact store

Returns:
    - ``None``
    
    
log_artifact
^^^^^^^^^^^^
method to log a local file as an MLflow artifact

Args:
    
    - ``local_path (str)``: Path to the file to log.
    - ``artifact_path (str, optional)``: run relative path to log the artifact in MLflow artifact store. Defaults to ``None``

Returns:
    - ``None``
    
log_artifacts
^^^^^^^^^^^^^
method to log a local directory as an mlflow artifact

Args:
    - ``local_dir (str)``: Path to the directory to log.
    - ``artifact_path (str, optional)``: run relative path to log the artifact in mlflow artifact store. Defaults to ``None``

Returns:
    - ``None``
    
log_figure
^^^^^^^^^^
method to log an image as an MLflow artifact

Args:
    - ``figure (matplotlib.figure.Figure)``: image to log
    - ``artifact_file (str)``: run relative path to log the artifact in mlflow artifact store.

Returns:
    ``None``
    
log_metric
^^^^^^^^^^
method to log a metric to MLflow

Args:
    - ``key (str)``: name of the metric to log.
    - ``value (float)``: value of the metric

Returns:
    - ``None``
    
log_metrics
^^^^^^^^^^^
method to log a dictionary of metrics to MLflow

Args:
    - ``metrics (dict)``: Dictionary of metrics to log.

Returns:
    - ``None``
    
log_param
^^^^^^^^^^
method to log a param to MLflow

Args:
    - ``key (str)``: name of the param to log.
    - ``value (float)``: value of the param

Returns:
    - ``None``
    
log_params
^^^^^^^^^^
method to log a dictionary of params to MLflow

Args:
    - ``params (dict)``: Dictionary of params to log
    
Returns:
    - ``None``
    
    
set_tag
^^^^^^^
method to set a tag to the run

Args:
    - ``key (str)``: name of the tag to log.
    - ``value (float)``: value of the tag

Returns:
    - ``None``
    
set_tags
^^^^^^^^
method to log a dictionary of tags to run

Args:
    - ``tags (dict)``: Dictionary of tags to log.

Returns:
    - ``None``
    
log_data
^^^^^^^^^
method to log details of dataset used for this run. Details will be logged as tags to the runs

Args:
    - ``feature_view (str)``: name of the feature view used for this run.
    - ``kwargs (dict)``: key word arguments capturing dataset details

Returns:
    - ``None``
    
log_prompts
^^^^^^^^^^^
logs the prompts and respective output to MLFlow

Args:
    - ``context (Union[list, str])``: input string or list of strings or dictionary
    - ``output (Union[list, str])``: output string or list of strings
    - ``prompts (Union[list, str])``: prompt string or list of prompt strings or prompt dictionary

Returns:
    - ``None``

>>> context = "some context"
>>> prompt = "input prompt"
>>> output = "output by the LLM"
>>> mlflow.llm.log_predictions(context, output, prompt)


log_huggingface_hosted_model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
downloads the pretrained Huggingface pipeline and logs it to MLFlow

Args:
    - ``architecture (str)``: name of the architecture as defined in Huggingface Model Hub
    - ``task (str)``: model task type (for instance, "text-generation")
    - ``model_type (str, optional)``: Defaults to ``HF-Pipeline``

Returns:
    - ``None``

>>> mf = MaxFlow('mlflow.url:5000', 'user', 'password')
>>> mf.set_experiment('MaxDemo')
>>> run = mf.start_run("MaxFlow-E2E", 'maxai_e2e', 'MaxAI E2E Run')
>>> architecture="edbeeching/gpt-neo-125M-imdb"
>>> task="text-generation"
>>> run.log_huggingface_hosted_model(architecture, task)

log_model
^^^^^^^^^
method to log Max.ai Models as mlflow artifacts.

Args:
    - ``model (Union[maxaibase.model.model_base.BaseModel, transformers.Pipeline])``: Model object to log.
    Must be one of ``maxaibase.model.model_base.BaseModel`` or ``transformers.Pipeline``.
    - ``model_kwargs (dict[str, str])``: Keyword arguments. If ``model`` passed is an instance of ``transformers.Pipeline``,
    the following arguments must be passed:
        - ``architecture (str)``: name of the architecture to be logged
        - ``task (str)``: task of the pipeline to be logged. For instance, ``text-generation``

Returns:
    - ``None``

>>> mf = MaxFlow('http://mlflow.url:5000', 'user', 'password')
>>> mf.set_experiment('MaxDemo')
>>> run = mf.start_run("MaxFlow-E2E", 'maxai_e2e', 'MaxAI E2E Run')
>>> run.log_model(model_pipeline, architecture="fine-tuned-opt-125m", task="text-generation")