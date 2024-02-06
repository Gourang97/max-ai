Architecture
============

Max.AI LLM architecture is designed for efficient handling and processing of data, as well as modern LLM app development tasks. At its core the Max.AI components run on Kubernetes and containers making it cloud agnostic and are very scalable.


Components
^^^^^^^^^^


.. image:: ../static/images/Data-Integration.png
   :width: 600px
   :align: center
   :alt: Max.AI LLM Architecture
   
|

- DNS, Loadbalancer and SSL: 
    - Enables access to the Max.AI apps for user access

- API Gateway: 
    - Takes care of routing requests to the respective microservice apps
    - Max.AI LLM apps can also be exposed as API as a service and API gateway ensures right routing, access control, rate limiting and throttling
    
- UI
    - User interface for the Max.AI apps which enable users to upload documents/content and then chat, summarize or run generic QnA
    
- Backend API
    - This is the core of the LLM app that houses the vectorization logic, RAG flow and handles summarization/QnA asks from the UI
    
- Code Notebooks
    - Code notebooks (jupyter or vscode) are developers point to get access to a mini IDE environment on the Max.AI platform and use to write use-case/app specific code flows. Max.AI LLM libraries or models can be used as base for building these use-cases.

- Redis cache
    - While the app makes a lot of LLM calls for every summarization/QnA ask, Redis ensures the calls are cached and we reduce LLM token calls for same repeated requests

- Vector DB and Storage
    - Data store of documents and their vectors

- PostgreSQL Database
    - Houses app meta data and RBAC configs


Max.AI apps are engineered to be swapped with any backend LLM models - Enterprise or OpenSource thus supporting models like Azure OpenAI, AWS Bedrock, Anthropic Claude, Llama etc. Access to these interfaces is abstracted ensuring compatibility with future updates to model endpoints/api calls.
   
   
