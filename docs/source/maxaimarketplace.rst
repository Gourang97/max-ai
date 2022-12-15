maxaimarketplace
================
This document captures the flow to deploy DAGs and use-cases in the Max Marketplace repository. We encourage the use of conventions and structures laid down in this document for the following reasons:

- Standardization - when similar conventions are followed, it accelerates the debugging process.
- Efficiency - everytime a PR is accepted, the use-case drivers and handlers files should be packed and built into a Docker image. Following these steps, helps to automate that process and trigger build for only use-cases which have been updated (covered in detail in the later sections), rather than trigger build for all the use-cases.

Directory Structure
*******************
Every client deployment (dev and prod) should have three directories, one for *configs*, one for *DAGs* and one for *use-case*. These directories should be further divided into sub-directories based on *domain*, which further are divided into *usecase*. The *usecase* is further divided into *objectives*, which captures execution files and configs. For illustration:

|
| maxaimarketplace
|    ├──maxaiconfigs
│    ├──domain_01
|    │    │    └──usecase_01
|    │    |         ├──spark_config.json
|    |    |         └──obj0
|    |    |              └──py_config.json
|    │    └──domain_02
|    │         └──usecase_02
|    |              ├──spark_config.json
|    |              └──obj0
|    |                   └──py_config.json
|    └──maxaidags
|
|


| simpleble-master
| ├── docs
| │   ├── build
| │   ├── make.bat
| │   ├── Makefile
| │   └── source
| ├── LICENSE
| ├── README.md
| ├── requirements.txt
| └── simpleble
|     └── simpleble.py
| 
|