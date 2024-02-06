Model
=====

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


    llm = MaxAnthropicLLM("claude-2").load_model()

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

