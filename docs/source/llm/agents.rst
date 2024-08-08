Max Agents
==========

Agents are a system with complex reasoning capabilities, memory, and the means to execute tasks. Max.AI LLM provides a list of agents that can be used out-of-the-box to develop and power applications.

MaxAgentQA
**********
A MaxAgentQA class for processing and analyzing text data using large language models (LLM).

Args:
    - ``llm_provider (Optional[str])``: The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, OpenAI, etc.
    - ``model_name (Optional[str])``: The name of the LLM model for the given provider.
    - ``model_kwargs (Optional[Dict])``: Optional keyword arguments for the LLM model, defaults to None.
    - ``chunk_size (int)``: The size of text chunks for processing, defaults to 800.
    - ``chunk_overlap (int)``: The overlap size between consecutive text chunks, defaults to 400.
    - ``chunk_method (str)``: The chunking strategy to be used in chunking the raw text, defaults to "recursive".
    - ``stream (bool)``: Flag to indicate if the data should be processed as a stream, defaults to False.
    - ``collection (Optional[str])``: The name of the collection to be used in the vector database, defaults to None.
    - ``prompt_config (Optional[Dict[str, str]])``: Configuration settings for prompts, defaults to None.
    - ``metadata_dict (Dict[str, bool])``: A dictionary to specify which metadata features to include, defaults to None.
    - ``embedding_model (str)``: The name of the embedding model to be used, defaults to "text-embedding-3-small".
    - ``embedding_provider (str)``: The provider of the embedding model, defaults to "openai".
    - ``embedding_kwargs (Optional[Dict])``: Optional keyword arguments for the embedding model, defaults to None.
    - ``retriever_type (str)``: The type of retriever to be used, defaults to "base".
    - ``reranker_type (str)``: The ranking method to be used by the retriever, defaults to "LostInMiddle".
    - ``generate_method (str)``: The method used for generation tasks, defaults to "stuff".
    - ``verbose (bool)``: Flag to indicate if verbose mode is enabled, defaults to True.
    - ``cache (bool)``: Flag to indicate if caching is enabled, defaults to False.
    - ``vector_store (str)``: The type of vector store to be used, defaults to "pgvector".
    - ``guardrails (bool)``: Flag to indicate if guardrails for content safety are enabled, defaults to False.
    - ``guardrails_kwargs (dict)``: Optional keyword arguments for configuring guardrails, defaults to an empty dict.
    - ``source_threshold (Optional[float])``: Similarity threshold for the embeddings compression filter, defaults to 0.10.

Attributes:
    - ``agent_type``: The type of the agent, defaults to "MaxAgentQA".
    - ``llm_provider``: The name of the large language provider.
    - ``model_name``: The name of the LLM model.
    - ``model_kwargs``: Keyword arguments for the LLM model.
    - ``chunk_size``: The size of text chunks for processing.
    - ``chunk_overlap``: The overlap size between consecutive text chunks.
    - ``chunk_method``: The chunking strategy to be used.
    - ``stream``: Flag indicating if the data should be processed as a stream.
    - ``collection``: The name of the collection to be used in the vector database.
    - ``embedding_model``: The name of the embedding model to be used.
    - ``embedding_provider``: The provider of the embedding model.
    - ``embedding_kwargs``: Keyword arguments for the embedding model.
    - ``metadata_dict``: Dictionary specifying which metadata features to include.
    - ``retriever_type``: The type of retriever to be used.
    - ``reranker_type``: The ranking method to be used by the retriever.
    - ``generate_method``: The method used for generation tasks.
    - ``verbose``: Flag indicating if verbose mode is enabled.
    - ``cache``: Flag indicating if caching is enabled.
    - ``vector_store``: The type of vector store to be used.
    - ``guardrails``: Flag indicating if guardrails for content safety are enabled.
    - ``source_threshold``: Similarity threshold for the embeddings compression filter.

Raises:
    - ``ValueError``: If `prompt_config` is not provided.
    - ``ValueError``: If `llm_provider` is not provided.
    - ``ValueError``: If `embedding_model` is not provided.

Returns:
    - ``MaxAgentQA``: An instance of MaxAgentQA.

Methods:
    - ``initialize_llm``: Initialize the LLM based on the provider.

        - ``provider (str)``: Name of the LLM provider.
        - ``model_name (str)``: Name of the model to be used, optional.
        - ``model_kwargs (dict)``: Additional keyword arguments for the model.

    - ``set_collection``: Set the current collection and initialize the vector database.

        - ``collection (str)``: Name of the collection.

    - ``get_collection``: Fetch the collection.

        - ``collection (str)``: Name of the collection.

    - ``_create_embeddings_instance``: Initialize the embedding based on the embedding model.

    - ``_init_vector_db``: Initialize the vector database based on the configured vector store.

    - ``_init_cache``: Initializes and returns a MaxCache instance.

    - ``_retrieve_text``: Retrieves and reranks text documents based on a given query using MaxRetriever.

        - ``query (str)``: The query string.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: Number of top results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``score_threshold (float)``: Threshold for scoring the results.

    - ``_retrieve_text_async``: Asynchronously retrieves and reranks text based on the given query and search parameters using MaxRetriever.

        - ``query (str)``: The query string.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: Number of top results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``score_threshold (float)``: Threshold for scoring the results.

    - ``_initialize_generator``: Initializes and returns an instance of MaxGenerator with the specified configurations.

        - ``prompt_config (Optional[Dict[str, str]])``: Configuration settings for prompts.
        - ``streamable (bool)``: Flag indicating if the generator should operate in a streamable mode.
        - ``verbose (bool)``: Flag indicating if verbose mode is enabled.
        - ``enable_chat (bool)``: Flag indicating if chat history should be enabled.

    - ``_log_prompt_response``: Logs the interaction between the query and the response to the MaxFlow run.

        - ``retrieved_text (List[TextDocument])``: The context texts retrieved.
        - ``response (str)``: The response generated by the model.
        - ``query (str)``: The initial query or prompt.

    - ``_log_query_costs``: Logs the costs associated with a query to a given dictionary.

        - ``response (str)``: The response generated by the model for the input prompt.
        - ``input_tokens_count (int)``: The number of tokens in the input prompt.
        - ``prompt_cost (float)``: The cost associated with processing the input prompt.
        - ``cost_log_dict (dict)``: The dictionary to which the cost data will be logged.

    - ``_log_chat_session``: Asynchronously logs a chat session message.

        - ``chat_session (list[str])``: Chat session information.
        - ``message_id (str)``: Identifier for the message.
        - ``query (str)``: The processed query.
        - ``resp (str)``: The response generated by the model.
        - ``org_query (str)``: The original query text as input by the user.
        - ``token_usage (dict)``: Information about the cost data.

    - ``_calculate_prompt_cost``: Calculates the cost of a prompt.

        - ``generator (MaxGenerator)``: A MaxGenerator object.
        - ``retrieved_text (list)``: A list of retrieved documents.
        - ``query (str)``: The query text.
        - ``prompt_config (Optional[Dict[str, str]])``: Configuration for the prompt.
        - ``chat (list[str])``: The chat history or conversation context.

    - ``process_file``: Asynchronously processes a given file to extract, clean, split text into documents, and add them to the vector database.

        - ``file (str)``: The file path to be processed.
        - ``doc_metadata (dict)``: Metadata associated with the document.

    - ``add``: Adds documents from the specified files to the collection managed by this instance.

        - ``files (Union[List[str], str])``: A list of file paths or a single file path to be added to the collection.
        - ``default_metadata (List[Dict])``: Optional list of metadata dictionaries corresponding to each file.

    - ``query``: Execute a query against the set collection and return the generated response.

        - ``query (str)``: The query string.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: Number of results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``score_threshold (float)``: Threshold for scoring the results.
        - ``prompt_config (Optional[Dict[str, str]])``: Configuration for the prompt.

    - ``_generate_response``: Generates a response to a given query, logs both the prompt and the response, and calculates the cost associated with generating the response.

        - ``generator (MaxGenerator)``: A MaxGenerator object.
        - ``query (str)``: The query string.
        - ``retrieved_text (list)``: A list of retrieved documents.
        - ``input_tokens_count (int)``: The number of tokens in the input prompt.
        - ``prompt_cost (float)``: The cost associated with generating the prompt.
        - ``cost_log_dict (dict)``: A dictionary to log the cost details of the query processing.

    - ``_process_chat_session``: Asynchronously processes a chat session to condense the query based on the chat history and update the cache.

        - ``query (str)``: The user's input query.
        - ``chat_session (list[str])``: Chat session information.

    - ``aquery``: Asynchronously execute a query and generate a response in a streaming fashion.

        - ``query (str)``: The query string.
        - ``k (int)``: Number of results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``search_type (str)``: Type of search to perform.
        - ``score_threshold (float)``: Threshold for scoring the results.
        - ``prompt_config (Optional[Dict[str, str]])``: Configuration for the prompt.
        - ``chat_session (list)``: Chat session information, if available.
        - ``message_id (str)``: Identifier for the message, if applicable.

    - ``_pre_generation_check_llm``: LLM to make pregeneration calls, hardcoded to GPT-4.

        - ``None``: No arguments needed.
            
    >>> from maxaillm.app.agent import MaxAgentQA
    >>> agent = MaxAgentQA(
    ...     llm_provider="anthropic",
    ...     model_name ="claude-2", 
    ...     chunk_size=1000,
    ...     stream=True, 
    ...     collection="myCollection", 
    ...     prompt_config=myPromptConfig
    ... )


MaxMultiModalAgentQA
*********************
MaxMultiModalAgentQA class for processing and analyzing text and image data using large language models (LLM).

Args:
    - ``llm_provider (Optional[str])``: The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, OpenAI, etc.
    - ``model_name (Optional[str])``: The name of the LLM model for the given provider.
    - ``model_kwargs (Optional[Dict])``: Optional keyword arguments for the LLM model, defaults to None.
    - ``doc_store_kwargs (Optional[Dict])``: Optional keyword arguments for the doc store, defaults to None.
    - ``chunk_size (int)``: The size of text chunks for processing, defaults to 800.
    - ``chunk_overlap (int)``: The overlap size between consecutive text chunks, defaults to 400.
    - ``chunk_method (str)``: The chunking strategy to be used in chunking the raw text, defaults to "recursive".
    - ``stream (bool)``: Flag to indicate if the data should be processed as a stream, defaults to False.
    - ``collection (Optional[str])``: The name of the collection to be used in the vector database, defaults to None.
    - ``prompt_config (Optional[Dict[str, str]])``: Configuration settings for prompts, defaults to None.
    - ``metadata_dict (Dict[str, bool])``: A dictionary to specify which metadata features to include, defaults to a predefined set.
    - ``embedding_model (str)``: The name of the embedding model to be used, defaults to "text-embedding-3-small".
    - ``embedding_provider (str)``: The provider of the embedding model, defaults to "openai".
    - ``embedding_kwargs (Optional[Dict])``: Optional keyword arguments for the embedding model, defaults to None.
    - ``retriever_type (str)``: The type of retriever to be used, defaults to "base".
    - ``reranker_type (str)``: The ranking method to be used by the retriever, defaults to "LostInMiddle".
    - ``generate_method (str)``: The method used for generation tasks, defaults to "stuff".
    - ``verbose (bool)``: Flag to indicate if verbose mode is enabled, defaults to True.
    - ``cache (bool)``: Flag to indicate if caching is enabled, defaults to False.
    - ``vector_store (str)``: The type of vector store to be used, defaults to "pgvector".
    - ``guardrails (bool)``: Flag to indicate if guardrails for content safety are enabled, defaults to False.
    - ``guardrails_kwargs (dict)``: Optional keyword arguments for configuring guardrails, defaults to an empty dict.
    - ``source_threshold (Optional[float])``: Similarity threshold for the embeddings compression filter, defaults to 0.10.

Attributes:
    - ``agent_type``: The type of the agent, defaults to "MaxMultiModalAgentQA".
    - ``doc_store_kwargs``: Keyword arguments for the document store.
    - ``doc_store``: The document store instance.
    - ``doc_store_id_key``: The key used for identifying documents in the store.
    - ``llm_provider``: The name of the large language provider.
    - ``model_name``: The name of the LLM model.
    - ``model_kwargs``: Keyword arguments for the LLM model.
    - ``chunk_size``: The size of text chunks for processing.
    - ``chunk_overlap``: The overlap size between consecutive text chunks.
    - ``chunk_method``: The chunking strategy to be used.
    - ``stream``: Flag indicating if the data should be processed as a stream.
    - ``collection``: The name of the collection to be used in the vector database.
    - ``prompt_config``: Configuration settings for prompts.
    - ``metadata_dict``: Dictionary specifying which metadata features to include.
    - ``embedding_model``: The name of the embedding model to be used.
    - ``embedding_provider``: The provider of the embedding model.
    - ``embedding_kwargs``: Keyword arguments for the embedding model.
    - ``retriever_type``: The type of retriever to be used.
    - ``reranker_type``: The ranking method to be used by the retriever.
    - ``generate_method``: The method used for generation tasks.
    - ``verbose``: Flag indicating if verbose mode is enabled.
    - ``cache``: Flag indicating if caching is enabled.
    - ``vector_store``: The type of vector store to be used.
    - ``guardrails``: Flag indicating if guardrails for content safety are enabled.
    - ``source_threshold``: Similarity threshold for the embeddings compression filter.

Raises:
    - ``ValueError``: If `prompt_config` is not provided.
    - ``ValueError``: If `llm_provider` is not provided.

Returns:
    - ``MaxMultiModalAgentQA``: An instance of MaxMultiModalAgentQA.

Methods:
    - ``set_collection``: Set the current collection and initialize the vector database.

        - ``collection (str)``: Name of the collection.

    - ``process_file``: Asynchronously processes a given file to extract, clean, split text into documents, and add them to the vector database.

        - ``file (str)``: The file path of the document to be processed.
        - ``doc_metadata (dict)``: Metadata associated with the document to be processed.

    - ``query``: Execute a query against the set collection and return the generated response leveraging images and extracted tables.

        - ``query (str)``: The query string.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: Number of results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``score_threshold (float)``: Threshold for scoring the results.
        - ``prompt_config (Optional[Dict[str, str]])``: Configuration for the prompt.

    - ``_generate_response``: Generates a response for a given query.

        - ``generator (MaxMultiModelGenerator)``: The generator object used for generating responses.
        - ``query (str)``: The user's query.
        - ``retrieved_text (str)``: The text context retrieved.
        - ``retrieved_images (list)``: The image context retrieved.
        - ``input_tokens_count (int)``: The number of tokens in the input query and context.
        - ``prompt_cost (float)``: The cost associated with generating the prompt.
        - ``cost_log_dict (dict)``: A dictionary for logging costs.

    - ``_retrieve_text``: Retrieves text based on a given query.

        - ``query (str)``: The search query.
        - ``search_type (str)``: The type of search to perform.
        - ``k (int)``: The number of top results to retrieve.
        - ``filters (dict)``: Filters to apply to the search query.
        - ``score_threshold (float)``: The minimum score threshold for retrieved results.

    - ``aquery``: Asynchronously executes a query against a collection and generates a response in a streaming fashion.

        - ``query (str)``: The query string.
        - ``k (int)``: Number of results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``search_type (str)``: Type of search to perform.
        - ``score_threshold (float)``: Threshold for scoring the results.
        - ``prompt_config (Optional[Dict[str, str]])``: Configuration for the prompt.
        - ``chat_session (list)``: Chat session information, if available.
        - ``message_id (str)``: Identifier for the message, if applicable.

    - ``_retrieve_text_async``: Asynchronously retrieves text data based on a given query.

        - ``query (str)``: The search query.
        - ``search_type (str)``: The type of search to perform.
        - ``k (int)``: The number of top results to retrieve.
        - ``filters (dict)``: Filters to apply to the search query.
        - ``score_threshold (float)``: The minimum score threshold for retrieved results.

    - ``resize_image``: Resizes an image if its dimensions exceed a maximum size, maintaining the aspect ratio.

        - ``max_image (object)``: An object containing the image data.

    - ``is_valid_image``: Validates an image based on its size, format, dimensions, and animation properties.

        - ``image_data (object)``: An object containing the image data.

    - ``_extract_and_save_images``: Extracts images from a given file using a specified extractor, filters valid images, and resizes them.

        - ``file (str or file-like object)``: The file from which images are to be extracted.
        - ``extractor (MaxDocumentExtractor)``: A MaxDocumentExtractor object.
        - ``extracted_images_dir (str)``: The directory where the extracted images will be saved.

    - ``_extract_tables``: Extracts tables from a given document file using the specified extractor tool.

        - ``file (str)``: The path to the document file from which tables are to be extracted.
        - ``extractor (MaxDocumentExtractor)``: A MaxDocumentExtractor object.

    - ``_add_image_documents``: Adds image documents to the retriever's vector and document stores.

        - ``retriever (Retriever)``: A retriever instance.
        - ``max_image_docs (List[object])``: An object containing the image data.
        - ``metadata_ext (Dict[str, Any])``: A dictionary containing additional metadata to be included in each summary document.

    - ``_initialize_generator``: Initializes a MaxMultiModelGenerator model with the specified configurations.

        - ``prompt_config (Dict[str, Any])``: Configuration settings for prompts.
        - ``streamable (bool)``: Flag to indicate if the generation process should be streamable.
        - ``verbose (bool)``: Flag to enable verbose mode.
        - ``enable_chat (bool)``: Flag to enable chat history in the generation process.
        

    >>> from maxaillm.agents.MaxMultiModalQA import MaxMultiModalAgentQA
    >>> agent = MaxMultiModalAgentQA(
    ...     llm_provider="anthropic",
    ...     model_name ="claude-2",
    ...     chunk_size=1000,
    ...     stream=True,
    ...     collection="myCollection",
    ...     prompt_config=myPromptConfig
    ... )
