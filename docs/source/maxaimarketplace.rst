maxaimarketplace
================
This document captures the flow to deploy DAGs and use-cases in the Max Marketplace repository. We encourage the use of conventions and structures laid down in this document for the following reasons:
- Standardization - when similar conventions are followed, it accelerates the debugging process.
- Efficiency - everytime a PR is accepted, the use-case drivers and handlers files should be packed and built into a Docker image. Following these steps, helps to automate that process and trigger build for only use-cases which have been updated (covered in detail in the later sections), rather than trigger build for all the use-cases.


    domain_01
        |--usecase_01
            |--spark_config
            |--obj0
            |   |--py_config
            |--obj1
                |--py_config
    domain_02
        |--usecase_01
        |   |--spark_config
        |   |--obj0
        |      |--py_config
        |--usecase_02
            |--spark_config
            |--obj0
            |   |--py_config
            |--obj1
                |--py_config