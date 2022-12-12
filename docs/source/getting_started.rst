Getting Started
===============
If you are new to Max.AI, please go through the installation guide and quickstart notebooks to get accustomed to the package.

Installation Guide
******************
As of now, Max.AI is not available as a pip installable module. So, to get started, one can clone the Max.AI repo and run the following commands to generate the ``.egg`` file.


>>> python setup.py sdist
>>> python setup.py bdist_egg


Once the ``.egg`` file created, install it using pip:

>>> pip install ./dist/maxai_artifacts-<some_extension>.egg
