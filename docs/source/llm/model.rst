Models
=========

This is an abstraction layer powered via one common entry point api gateway that enables access to all kind of LLMs APIs including the Enterprise ones. LLM’s can be integrated with client applications via REST PAI's or cloud specific SDKs.

LLM
****

MaxAnthropicLLM
^^^^^^^^^^^^^^^
MaxAnthropicLLM is a class that inherits from BaseLLM and represents an Anthropic language model. It includes specific configurations for the Anthropic model.

Args:
    - ``model_name (str)``: Name of the Anthropic model to be loaded.
    - ``temperature (float, optional)``: The temperature for the language generation. Default is 0.0.
    - ``max_tokens_to_sample (int, optional)``: The maximum number of tokens for generation. Default is 2048.
    - ``streaming (bool, optional)``: Whether to enable streaming for the model. Default is True.
    - ``top_p (float, optional)``: The nucleus sampling parameter. Default is None.

Attributes:
    - ``temperature``: The temperature for the language generation.
    - ``max_tokens_to_sample``: The maximum number of tokens for generation.
    - ``streaming``: Whether to enable streaming for the model.
    - ``top_p``: The nucleus sampling parameter.
    - ``cost_param``: The cost parameters for the model.
    - ``api_key``: The API key for the Anthropic model.

Raises:
    - ``EnvironmentError``: If the ANTHROPIC_API_KEY environment variable is not set.

Methods:
    - ``load_model()``: Loads the Anthropic model specified in the initialization.
    
.. code-block:: python
        
    from maxaillm.model.llm import MaxAnthropicLLM
    llm = MaxAnthropicLLM(model_name="Claude-2", temperature=0.7, max_tokens_to_sample=1024, streaming=False, top_p=0.9)
    model = llm.load_model()


MaxOpenAILLM
^^^^^^^^^^^^^
MaxOpenAILLM is a class that inherits from BaseLLM and represents an OpenAI language model. It includes specific configurations for the OpenAI model.

Args:
    - ``model_name (str)``: Name of the OpenAI model to be loaded.
    - ``temperature (float, optional)``: The temperature for the language generation. Default is 0.0.
    - ``max_tokens (int, optional)``: The maximum number of tokens for generation. Default is None.
    - ``stop (str, optional)``: Stop tokens for generation. Default is None.
    - ``streaming (bool, optional)``: Whether to enable streaming for the model. Default is True.
    - ``seed (int, optional)``: The seed for the random number generator. Default is 123.
    - ``top_p (float, optional)``: The nucleus sampling parameter. Default is None.

Attributes:
    - ``temperature``: The temperature for the language generation.
    - ``max_tokens``: The maximum number of tokens for generation.
    - ``stop``: Stop tokens for generation.
    - ``streaming``: Whether to enable streaming for the model.
    - ``seed``: The seed for the random number generator.
    - ``top_p``: The nucleus sampling parameter.
    - ``cost_param``: The cost parameters for the model.
    - ``api_key``: The API key for the OpenAI model.

Raises:
    - ``EnvironmentError``: If the OPENAI_API_KEY environment variable is not set.

Methods:
    - ``load_model()``: Loads the OpenAI model specified in the initialization.
    
.. code-block:: python

    from maxaillm.model.llm import MaxOpenAILLM
    llm = MaxOpenAILLM(model_name="text-davinci-003", temperature=0.7, max_tokens=150, stop=["\n"], streaming=False)
    model = llm.load_model()
    
    
MaxAzureOpenAILLM
^^^^^^^^^^^^^^^^^^
MaxAzureOpenAILLM is a class that inherits from BaseLLM and represents an Azure-hosted OpenAI language model. It includes specific configurations for Azure-hosted OpenAI models.

Args:
    - ``max_retries (int, optional)``: The maximum number of retries for loading the model. Default is 2.
    - ``streaming (bool, optional)``: Whether to enable streaming for the model. Default is True.
    - ``temperature (float, optional)``: The temperature for the language generation. Default is 0.0.
    - ``max_tokens (int, optional)``: The maximum number of tokens for generation. Default is None.
    - ``seed (int, optional)``: The seed for the random number generator. Default is 123.
    - ``top_p (float, optional)``: The nucleus sampling parameter. Default is None.

Attributes:
    - ``deployment_name``: The name of the deployment for the Azure-hosted OpenAI model.
    - ``model_name``: The name of the Azure-hosted OpenAI model.
    - ``deployment_endpoint``: The endpoint for the Azure-hosted OpenAI model.
    - ``deployment_version``: The version of the Azure-hosted OpenAI model.
    - ``api_key``: The API key for the Azure-hosted OpenAI model.
    - ``max_tokens``: The maximum number of tokens for generation.
    - ``temperature``: The temperature for the language generation.
    - ``streaming``: Whether to enable streaming for the model.
    - ``max_retries``: The maximum number of retries for loading the model.
    - ``seed``: The seed for the random number generator.
    - ``top_p``: The nucleus sampling parameter.
    - ``cost_param``: The cost parameters for the model.

Raises:
    - ``EnvironmentError``: If one or more required environment variables are not set for AzureChatOpenAI.

Methods:
   - ``load_model()``: Loads the Azure-hosted OpenAI model specified in the initialization.

.. code-block:: python

    from maxaillm.model.llm import MaxAzureOpenAILLM
    llm = MaxAzureOpenAILLM(model_name="gpt-3.5-turbo", max_retries=3, streaming=False, temperature=0.7, max_tokens=150)
    model = llm.load_model()
   
   
MaxBedrockLLM
^^^^^^^^^^^^^^
MaxBedrockLLM is a class that inherits from BaseLLM and represents a Bedrock-based language model. It includes specific configurations for the Bedrock-based LLM model.

Args:
    - ``model_name (str)``: Name of the Bedrock model to be loaded. The name should be provided as Provider_name.model_name.
    - ``temperature (float, optional)``: The temperature for the language generation. Default is 0.0.
    - ``max_tokens_to_sample (int, optional)``: The maximum number of tokens for generation. Default is 2048.
    - ``streaming (bool, optional)``: Whether to enable streaming for the model. Default is True.
    - ``top_p (float, optional)``: The nucleus sampling parameter. Default is None.

Attributes:
    - ``temperature``: The temperature for the language generation.
    - ``max_tokens_to_sample``: The maximum number of tokens for generation.
    - ``streaming``: Whether to enable streaming for the model.
    - ``top_p``: The nucleus sampling parameter.
    - ``cost_param``: The cost parameters for the model.

Raises:
    - ``EnvironmentError``: If Bedrock environment configurations are not set.

Methods:
    - ``load_model()``: Loads the Bedrock-based model specified in the initialization.

.. code-block:: python
        
        from maxaillm.model.llm import MaxBedrockLLM
        llm = MaxBedrockLLM(model_name="anthropic.claude-v2", temperature=0.7, max_tokens_to_sample=1024, streaming=False)
        model = llm.load_model()


MaxGoogleLLM 
^^^^^^^^^^^^^^
Represents a Google language model. This class extends the BaseLLM class and includes specific configurations for the Google model.

Args:
    - ``model_name (str)``: Name of the Google model to be loaded.
    - ``temperature (float, Optional)``: The temperature for the language generation. Defaults to 0.0.
    - ``max_tokens_to_sample (int, Optional)``: The maximum number of tokens for generation. Defaults to 2048.
    - ``streaming (bool, Optional)``: Whether to enable streaming for the model. Defaults to True.
    - ``top_p (float, Optional)``: Controls the nucleus sampling. Defines the probability mass to consider for the next token's generation. Defaults to None.
    - ``convert_system_message_to_human (bool, Optional)``: Whether system-generated messages should be converted to a more human-readable format. Defaults to True.

Raises:
    - ``ImportError``: If the necessary dependencies for interacting with Google's Generative AI are not installed.
    - ``EnvironmentError``: If the GOOGLE_API_KEY environment variable is not set.

Methods:
    - ``load_model``: Loads the Google model specified in the initialization.

        - Returns:
            - ``ChatGoogleGenerativeAI``: The loaded Google model instance.

.. code-block:: python

    from maxaillm.model.llm import MaxGoogleLLM
    llm = MaxGoogleLLM(model_name="gemini-1.5-pro", temperature=0.7, max_tokens_to_sample=1024, streaming=False)
    model = llm.load_model()
    
    
MaxVertexAILLM
^^^^^^^^^^^^^^^
Represents a VertexAI language model. This class extends the BaseLLM class and includes specific configurations for the VertexAI model.

Args:
    - ``model_name (str)``: The name of the model to be loaded.
    - ``temperature (float, Optional)``: Controls randomness in generation. Defaults to 0.0.
    - ``max_tokens (int, Optional)``: The maximum number of tokens to generate. Defaults to None.
    - ``stop (str, Optional)``: The stop sequence for generation. Defaults to None.
    - ``streaming (bool, Optional)``: Whether to stream the output. Defaults to True.
    - ``seed (int, Optional)``: Seed for random number generator for reproducibility. Defaults to 123.
    - ``top_p (float, Optional)``: Controls diversity via nucleus sampling. Defaults to None.

Raises:
    - ``ImportError``: If the required `langchain_google_vertexai` package is not installed.

Methods:
    - ``load_model``: Loads the VertexAI model with the specified parameters.

        - Returns:
            - ``VertexAI``: The loaded VertexAI model instance.

.. code-block:: python

    from maxaillm.model.llm import MaxVertexAILLM
    llm = MaxVertexAILLM(model_name="gemini-1.0-pro-002", temperature=0, max_tokens_to_sample=1024, streaming=False)
    model = llm.load_model()


MaxAzuremlLLM
^^^^^^^^^^^^^^
Represents an Azure-ML-hosted language model. This class extends the BaseLLM class and includes specific configurations for Azure-hosted OpenAI models.

Args:
    - ``max_retries (int, Optional)``: The maximum number of retries for a request in case of failures. Defaults to 2.
    - ``streaming (bool, Optional)``: Whether to enable streaming for the model. Defaults to True.
    - ``temperature (float, Optional)``: The temperature for the language generation. Defaults to 0.0.
    - ``max_tokens (int, Optional)``: The maximum number of tokens for generation. Defaults to 512.
    - ``seed (int, Optional)``: Seed for reproducibility. Defaults to 123.
    - ``top_p (float, Optional)``: Controls the nucleus sampling. Defines the probability mass to consider for the next token's generation. Defaults to None.

Raises:
    - ``ImportError``: If Azure ML dependencies are not installed.
    - ``KeyError``: If required environment variables are not set.

Methods:
    - ``load_model``: Loads the Azure-ML-hosted LLM model specified in the initialization.

        - Returns:
            - ``AzureMLChatOnlineEndpoint``: The loaded Azure-ML-hosted LLM model instance.

.. code-block:: python

    from maxaillm.model.llm import MaxAzuremlLLM
    llm = MaxAzuremlLLM(max_retries=3, streaming=False, temperature=0.5, max_tokens=256, seed=42, top_p=0.9)
    model = llm.load_model()


MaxGroqLLM
^^^^^^^^^^^
Represents a Groq language model. This class extends the BaseLLM class and includes specific configurations for the Groq model.

Args:
    - ``model_name (str)``: Name of the Groq model to be loaded.
    - ``temperature (float, Optional)``: The temperature for the language generation. Defaults to 0.0.
    - ``max_tokens_to_sample (int, Optional)``: The maximum number of tokens for generation. Defaults to 2048.
    - ``streaming (bool, Optional)``: Whether to enable streaming for the model. Defaults to True.
    - ``top_p (float, Optional)``: Controls the nucleus sampling. Defines the probability mass to consider for the next token's generation. Defaults to None.

Raises:
    - ``ImportError``: If the necessary dependencies for interacting with the Groq platform are not installed.
    - ``EnvironmentError``: If the GROQ_API_KEY environment variable is not set.

Methods:
    - ``load_model``: Loads the Groq model specified in the initialization.

        - Returns:
            - ``ChatGroq``: The loaded Groq model instance.

.. code-block:: python

    from maxaillm.model.llm import MaxGroqLLM
    lm = MaxGroqLLM(model_name="mixtral-8x7b-32768", temperature=0.7, max_tokens_to_sample=1024, streaming=False)
    model = llm.load_model()


MaxVertexAILLM
^^^^^^^^^^^^^^^
Represents a VertexAI language model. This class extends the BaseLLM class and includes specific configurations for the VertexAI model.

Args:
    - ``model_name (str)``: The name of the model to be loaded.
    - ``temperature (float, Optional)``: Controls randomness in generation. Defaults to 0.0.
    - ``max_tokens (int, Optional)``: The maximum number of tokens to generate. Defaults to None.
    - ``stop (str, Optional)``: The stop sequence for generation. Defaults to None.
    - ``streaming (bool, Optional)``: Whether to stream the output. Defaults to True.
    - ``seed (int, Optional)``: Seed for random number generator for reproducibility. Defaults to 123.
    - ``top_p (float, Optional)``: Controls diversity via nucleus sampling. Defaults to None.

Raises:
    - ``ImportError``: If the required `langchain_google_vertexai` package is not installed.

Methods:
    - ``load_model``: Loads the VertexAI model with the specified parameters.

        - Returns:
            - ``VertexAI``: The loaded VertexAI model instance.

.. code-block:: python

    from maxaillm.model.llm import MaxVertexAILLM
    llm = MaxVertexAILLM(model_name="gemini-1.0-pro-002", temperature=0, max_tokens_to_sample=1024, streaming=False)
    model = llm.load_model()


MaxHuggingFaceLLM
^^^^^^^^^^^^^^^^^
Represents a HuggingFace language model. This class extends the BaseLLM class and includes specific configurations for the HuggingFace model.

Args:
    - ``model_name (str)``: The name of the model to be loaded.
    - ``task (str, Optional)``: The task to be performed by the model. Defaults to "text-generation".
    - ``temperature (float, Optional)``: Controls randomness in generation. Defaults to 0.0.
    - ``max_tokens (int, Optional)``: The maximum number of tokens to generate. Defaults to None.
    - ``stop (str, Optional)``: The stop sequence for generation. Defaults to None.
    - ``streaming (bool, Optional)``: Whether to stream the output. Defaults to True.
    - ``seed (int, Optional)``: Seed for random number generator for reproducibility. Defaults to 123.
    - ``top_p (float, Optional)``: Controls diversity via nucleus sampling. Defaults to None.

Raises:
    - ``ImportError``: If the required `langchain-huggingface` package is not installed.

Methods:
    - ``load_model``: Loads the HuggingFace model with the specified parameters.

        - Returns:
            - ``HuggingFace``: The loaded HuggingFace model instance.

.. code-block:: python

    from maxaillm.model.llm import MaxHuggingFaceLLM
    llm = MaxHuggingFaceLLM(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        temperature=0,
        max_tokens_to_sample=1024,
        streaming=True
    )
    model = llm.load_model()