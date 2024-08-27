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

    - ``get_sources``: Retrieve the sources used to provide a response to the provided query.

        - ``query (str)``: The query string.
        - ``search_type (str)``: The type of search to perform.
        - ``k (int)``: The number of documents to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``score_threshold (float)``: The minimum score threshold for retrieved documents.
        - ``top_k (bool)``: Flag to indicate whether to return only the top-k documents.
        - ``chat_session (object)``: The chat session object, if applicable.
    
.. code-block:: python
    from maxaillm.app.agent import MaxAgentQA

    # Define prompt configuration
    myPromptConfig = {'moderations':'', 'task':'', 'identity':''}

    # Initialize MaxAgentQA
    agent = MaxAgentQA(
        llm_provider="openai",
        model_name ="gpt-4o",
        chunk_size=1000,
        stream=True,
        collection="myCollection",
        prompt_config=myPromptConfig
    )

    # Example of initializing the LLM
    agent.initialize_llm(provider="anthropic", model_name="claude-2", model_kwargs={"temperature": 0.7})

    # Example of setting a collection
    agent.set_collection(collection="myCollection")

    # Example of getting the collection name
    collection_name = agent.get_collection(collection="myCollection")

    # Example of processing a file
    agent.process_file(file="document.pdf", doc_metadata={"author": "John Doe"})

    # Example of adding documents to the collection
    agent.add(files=["doc1.pdf", "doc2.pdf"], default_metadata=[{"author": "John Doe"}, {"author": "Jane Doe"}])

    # Example of querying the collection
    response = agent.query(query="Explain Reinforcement Learning", search_type="mmr", k=5)

    # Example of asynchronously querying the collection
    async for token in agent.aquery(query="Explain Reinforcement Learning", k=5):
        print(token, end="")

    # Example of retrieving the sources used in a response
    sources = agent.get_sources(query="Explain Reinforcement Learning", search_type="mmr", k=5)


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

.. code-block:: python
    from maxaillm.agents.MaxMultiModalQA import MaxMultiModalAgentQA

    # Initialize MaxMultiModalAgentQA
    agent = MaxMultiModalAgentQA(
        llm_provider="anthropic",
        model_name="claude-2",
        chunk_size=1000,
        stream=True,
        collection="myCollection",
        prompt_config=myPromptConfig
    )

    # Example of setting a collection
    agent.set_collection("myCollection")

    # Example of processing a file
    success = agent.process_file("/path/to/document.pdf", {"title": "Sample Document", "author": "John Doe"})

    # Example of querying the collection
    response = agent.query("What is the tallest mountain?", search_type="mmr", k=5)

    # Example of asynchronously querying the collection
    async for token in agent.aquery(query="What is the capital of France?", k=5):
        print(token, end='')

    # Example of resizing an image
    resized_image_obj = agent.resize_image(image_obj)
    
    
MaxGraphAgentQA
****************
A MaxGraphAgentQA for processing and analyzing text data using large language models (LLM).

Args:
    - ``llm_provider (str, Optional)``: The name of the large language provider you want to use. Defaults to None.
    - ``model_name (str, Optional)``: The name of the LLM model for the given provider. Defaults to None.
    - ``model_kwargs (Dict, Optional)``: Optional keyword arguments for the LLM model. Defaults to None.
    - ``doc_store_kwargs (Dict, Optional)``: Optional keyword arguments for the doc store. Defaults to None.
    - ``chunk_size (int, Optional)``: The size of text chunks for processing. Defaults to 800.
    - ``chunk_overlap (int, Optional)``: The overlap size between consecutive text chunks. Defaults to 400.
    - ``chunk_method (str, Optional)``: The chunking strategy to be used in chunking the raw text. Defaults to "recursive".
    - ``stream (bool, Optional)``: Flag to indicate if the data should be processed as a stream. Defaults to False.
    - ``collection (str, Optional)``: The name of the collection to be used in the vector database. Defaults to None.
    - ``prompt_config (Dict[str, str], Optional)``: Configuration settings for prompts. Defaults to None.
    - ``metadata_dict (Dict[str, bool], Optional)``: A dictionary to specify which metadata features to include. Defaults to a predefined set.
    - ``embedding_model (str, Optional)``: The name of the embedding model to be used. Defaults to "text-embedding-3-small".
    - ``embedding_provider (str, Optional)``: The provider of the embedding model. Defaults to "openai".
    - ``embedding_kwargs (Dict, Optional)``: Optional keyword arguments for the embedding model. Defaults to None.
    - ``retriever_type (str, Optional)``: The type of retriever to be used. Defaults to "base".
    - ``reranker_type (str, Optional)``: The ranking method to be used by the retriever. Defaults to "LostInMiddle".
    - ``generate_method (str, Optional)``: The method used for generation tasks. Defaults to "stuff".
    - ``verbose (bool, Optional)``: Flag to indicate if verbose mode is enabled. Defaults to True.
    - ``cache (bool, Optional)``: Flag to indicate if caching is enabled. Defaults to False.
    - ``vector_store (str, Optional)``: The type of vector store to be used. Defaults to "pgvector".
    - ``guardrails (bool, Optional)``: Flag to indicate if guardrails are enabled. Defaults to False.
    - ``guardrails_config (GuardrailConfiguration, Optional)``: Configuration for guardrails. Defaults to None.
    - ``source_threshold (float, Optional)``: Similarity threshold for the embeddings compression filter. Defaults to 0.10.

Attributes:
    - ``agent_type``: The type of the agent.

Raises:
    - ``Exception``: If an error occurs during initialization.

Returns:
    - ``MaxGraphAgentQA``: An instance of MaxGraphAgentQA.

Methods:
    - ``set_collection``: Sets the current collection and initializes the vector database.

        - Args:
            - ``collection (str)``: Name of the collection.

    - ``process_file``: Processes a given file to extract, clean, split text into documents, and add them to the vector database.

        - Args:
            - ``file``: The file path to be processed.
            - ``doc_metadata``: Metadata associated with the document.

        - Returns:
            - ``bool``: True if the file was processed successfully.

        - Raises:
            - ``Exception``: If an error occurs during file processing.

    - ``query``: Executes a query against the set collection and returns the generated response.

        - Args:
            - ``query (str, Optional)``: The query string. Defaults to "".
            - ``search_type (str, Optional)``: Type of search to perform. Defaults to "mmr".
            - ``k (int, Optional)``: Number of results to retrieve. Defaults to 10.
            - ``filters (dict, Optional)``: Filters to apply for the query. Defaults to {}.
            - ``score_threshold (float, Optional)``: Threshold for scoring the results. Defaults to 0.05.
            - ``prompt_config (Optional)``: Configuration for the prompt. Defaults to None.

        - Returns:
            - ``str``: The response generated by the generator.

        - Raises:
            - ``Exception``: If the collection is not set or other errors occur during processing.

    - ``_generate_response``: Generates a response using the generator.

        - Args:
            - ``generator``: The generator to use.
            - ``query``: The query string.
            - ``retrieved_text``: The retrieved text.
            - ``retrieved_images``: The retrieved images.
            - ``input_tokens_count``: The count of input tokens.
            - ``prompt_cost``: The cost of the prompt.
            - ``cost_log_dict``: The cost log dictionary.

        - Returns:
            - ``str``: The generated response.

        - Raises:
            - ``Exception``: If an error occurs during response generation.

    - ``_retrieve_text``: Retrieves text based on the query.

        - Args:
            - ``query``: The query string.
            - ``search_type``: The type of search to perform.
            - ``k``: The number of results to retrieve.
            - ``filters``: The filters to apply.
            - ``score_threshold``: The score threshold.

        - Returns:
            - ``list``: The list of retrieved text.

        - Raises:
            - ``Exception``: If an error occurs during text retrieval and reranking.

    - ``_log_chat_session``: Logs the chat session.

        - Args:
            - ``chat_session``: The chat session information.
            - ``message_id``: The message identifier.
            - ``chunks``: The chunks of text.
            - ``resp``: The response.
            - ``org_query``: The original query.
            - ``graph``: The graph information.

        - Raises:
            - ``Exception``: If an error occurs during chat session logging.

    - ``aquery``: Asynchronously executes a query and generates a response in a streaming fashion.

        - Args:
            - ``query (str, Optional)``: The query string. Defaults to "".
            - ``k (int, Optional)``: Number of results to retrieve. Defaults to 10.
            - ``filters (dict, Optional)``: Filters to apply for the query. Defaults to {}.
            - ``search_type (str, Optional)``: Type of search to perform. Defaults to "mmr".
            - ``score_threshold (float, Optional)``: Threshold for scoring the results. Defaults to 0.05.
            - ``prompt_config (Optional)``: Configuration for the prompt. Defaults to None.
            - ``chat_session (list, Optional)``: Chat session information. Defaults to [].
            - ``message_id (Optional)``: Identifier for the message. Defaults to None.
            - ``citation (bool, Optional)``: Flag to indicate if citation is enabled. Defaults to False.

        - Returns:
            - ``AsyncGenerator``: Yields tokens of the generated response.

        - Raises:
            - ``Exception``: If the collection is not set or if other errors occur during processing.


SourceProvider
**************
SourceProvider class for retrieving and managing sources using large language models (LLM).

Args:
    - ``llm (BaseLLM)``: An instance of a BaseLLM class.
    - ``embedding_model (MaxLangchainEmbeddings)``: An instance of MaxLangchainEmbeddings class for generating embeddings from text.
    - ``vectordb (MaxLangchainVectorStore)``: An instance of MaxLangchainVectorStore class for storing and retrieving vector embeddings.
    - ``doc_store (Optional[Any])``: A document store for storing raw documents. Defaults to None.
    - ``retriever_type (str)``: The type of retriever to be used for fetching documents. Defaults to "base".
    - ``reranker_type (str)``: The type of reranker to be used for re-ranking the retrieved documents. Defaults to "LostInMiddle".
    - ``collection (Optional[str])``: The name of the collection to be used in the vector database. Defaults to None.
    - ``stream (Optional[bool])``: Flag to indicate if the data should be processed as a stream. Defaults to True.
    - ``source_threshold (Optional[float])``: Similarity threshold for the embeddings compression filter. Defaults to 0.10.

Attributes:
    - ``llm``: The large language model instance.
    - ``embedding_model``: The embedding model instance.
    - ``vectordb``: The vector database instance.
    - ``doc_store``: The document store instance.
    - ``retriever_type``: The type of retriever.
    - ``reranker_type``: The type of reranker.
    - ``stream``: Flag indicating if the data should be processed as a stream.
    - ``source_threshold``: Similarity threshold for the embeddings compression filter.


Methods:
    - ``_retrieve_text_async``: Asynchronously retrieves and reranks text based on the given query and search parameters using MaxRetriever.

        - ``vectordb (MaxLangchainVectorStore)``: The instance of the vector database.
        - ``query (str)``: The query string.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: The number of top results to retrieve.
        - ``filters (dict)``: Filters to apply for the query.
        - ``score_threshold (float)``: Threshold for scoring the results.

    - ``get_sources``: Retrieve and return sources related to a given query.

        - ``query (str)``: The query string for which sources are to be retrieved.
        - ``filter_type (str)``: The type of filter used for source retrieval ('embeddings' or 'llm'). Defaults to 'embeddings'.

    - ``get_chunks``: Retrieve and return text chunks related to a given query using different retrieval methods.

        - ``query (str)``: The query string for which text chunks are to be retrieved.
        - ``filter_type (str)``: The retrieval method ('vanilla', 'llm', 'embeddings'). Defaults to 'vanilla'.
        - ``search_type (str)``: The type of search to be performed (e.g., 'mmr'). Defaults to 'mmr'.
        - ``k (int)``: The number of results to retrieve. Defaults to 10.
        - ``filters (dict)``: Any filters to apply to the search query. Defaults to empty dict.
        - ``score_threshold (float)``: The threshold score for including a document in the results. Defaults to 0.05.

.. code-block:: python
    from maxaillm.agents.SourceProvider import SourceProvider

    # Initialize SourceProvider within an agent class
    sources = SourceProvider(
        llm=self.llm,
        embedding_model=self.embedding_model, 
        source_threshold=self.source_threshold, 
        vectordb=self.vectordb
    )

    # Example of retrieving sources related to a query
    sources = await source_provider.get_sources(
        query="What are the key points in the Quantum Computing document?",
        filter_type='embeddings'
    )

    # Example of retrieving text chunks related to a query
    chunks = await source_provider.get_chunks(
        query="machine learning",
        filter_type="vanilla",
        search_type="mmr",
        k=5,
        filters={},
        score_threshold=0.05
    )