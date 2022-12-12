Getting Started
===============
If you are new to Max.AI, please go through the installation guide and quickstart notebooks to get accustomed to the package.

Installation Guide
******************
As of now, Max.AI is not available as a pip installable module. So, to get started, one can clone the Max.AI repo and run the following commands to generate the `.egg` file.

``
>>> python setup.py sdist
>>> python setup.py bdist_egg
``
Once the `.egg` file created, install it using pip:

``
>>> pip install ./dist/maxai_artifacts-<some_extension>.egg
``
    

Quickstart Notebooks
********************
As Max.AI can be used for variety of use-cases, the following list of notebooks provides a quick glance over how to use these modules:

..  t3-field-list-table::
    :header-rows: 1

    -   :Header1:   Header1
        :Header2:   Header2

    -   :Header1:   1
        :Header2:   one

    -   :Header1:   2
        :Header2:   two
   

+----------+---------------------+
| Module | Accompanying Notebook |
+==========+=====================+
| 1        | one      |
+----------+----------+
| 2        | two      |
+----------+----------+




| Module |  |
|--------|-----------------------|
| MaxFlow (MLflow wrapper) | [Notebook](https://dev.azure.com/personalize-ai/max.ai/_git/max.ai.ds.core?path=/documents/mlflow_demo_notebook.ipynb&_a=preview)|
| maxaifeaturization | [Notebook](https://dev.azure.com/personalize-ai/max.ai/_git/max.ai.ds.core?path=/documents/maxaifeaturization-demo-notebook.ipynb&_a=preview)|