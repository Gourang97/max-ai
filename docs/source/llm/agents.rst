Max Agents
==========

MaxAgentQA
**********
A QA Agent for processing and analyzing text data using large language models (LLM).

Args:
    - ``llm_provider (str, optional)``: The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, openAI etc.
    - ``model_name (str, optional)``: The name of the llm model for the given provider.
    - ``model_kwargs (dict, optional)``: Optional keyword arguments for the llm model, defaults to None.
    - ``chunk_size (int, optional)``: The size of text chunks for processing, defaults to 2000.
    - ``chunk_overlap (int, optional)``: The overlap size between consecutive text chunks, defaults to 200.
    - ``stream (bool, optional)``: Flag to indicate if the data should be processed as a stream, defaults to False.
    - ``collection (str, optional)``: The name of the collection to be used in the vector database, defaults to None.
    - ``prompt_config (dict, optional)``: Configuration settings for prompts, defaults to None.
    - ``metadata_dict (dict, optional)``: A dictionary to specify which metadata features to include, defaults to a predefined set.
    - ``embedding_model (str, optional)``: The name of the embedding model to be used, defaults to "sentence-transformers/all-mpnet-base-v2".
    - ``retriever_rank (str, optional)``: The ranking method to be used by the retriever, defaults to "LostInMiddle".
    - ``generate_method (str, optional)``: The method used for generation tasks, defaults to "stuff".
    - ``verbose (bool, optional)``: Flag to indicate if verbose mode is enabled, defaults to "True".
    - ``vector_store (str, optional)``: The type of vector store to be used, defaults to "pgvector".
    
Attributes:
    - ``agent_type``: The type of the agent.
    - ``llm_provider``: The provider of the LLM.
    - ``model_name``: The name of the model.
    - ``model_kwargs``: Additional keyword arguments for the model.
    - ``chunk_size``: The size of the chunks to be processed.
    - ``chunk_overlap``: The overlap between chunks.
    - ``stream``: Whether to use streaming mode.
    - ``collection``: The collection to use.
    - ``embedding_model``: The embedding model to use.
    - ``metadata_dict``: Metadata configuration.
    - ``retriever_rank``: The rank of the retriever.
    - ``generate_method``: The method to use for generation.
    - ``verbose``: Whether to print verbose output.
    - ``vector_store``: The vector store to use.
    - ``cost_param``: The cost parameters.
    
Methods:
    - ``initialize_llm(provider, model_name, model_kwargs)``: Initializes the LLM.

        - ``provider (str)``: Provider of the LLM.
        - ``model_name (str)``: Name of the model.
        - ``model_kwargs (dict)``: Additional arguments for the model.
        
    - ``set_collection(collection)``: Sets the collection name and initializes the vector database.

        - ``collection (str)``: The name of the collection to be used in the vector database.

    - ``get_collection()``: Returns the current collection name.

    - ``init_vector_db()``: Initializes the vector database.
    
    - ``process_file(file, doc_metadata)``: Processes a file and adds it to the vector database.

        - ``file (str)``: File to process.
        - ``doc_metadata (dict)``: Metadata for the document.
        
    - ``add(files, default_metadata)``: Adds files to the collection.

        - ``files (list[str])``: List of files to add.
        - ``default_metadata (list[dict])``: Default metadata for the files.
    
    - ``query(query, search_type, k, filters, score_threshold, prompt_config)``: Queries the collection and generates a response.

        - ``query (str)``: Query to use.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: Number of results to return.
        - ``filters (dict)``: Filters to apply.
        - ``score_threshold (float)``: Threshold for the score.
        - ``prompt_config (dict)``: Configuration for the prompt.
        
    - ``aquery(query, k, filters, search_type, score_threshold, prompt_config, chat_session, message_id)``: Asynchronously queries the collection and generates a response.

        - ``query (str)``: Query to use.
        - ``k (int)``: Number of results to return.
        - ``filters (dict)``: Filters to apply.
        - ``search_type (str)``: Type of search to perform.
        - ``score_threshold (float)``: Threshold for the score.
        - ``prompt_config (dict)``: Configuration for the prompt.
        - ``chat_session (str)``: Chat session to use.
        - ``message_id (str)``: ID of the message.
    
    - ``get_sources(query, search_type, k, filters, score_threshold, top_k, chat_session)``: Retrieves sources based on the query.

        - ``query (str)``: Query to use.
        - ``search_type (str)``: Type of search to perform.
        - ``k (int)``: Number of results to return.
        - ``filters (dict)``: Filters to apply.
        - ``score_threshold (float)``: Threshold for the score.
        - ``top_k (bool)``: Whether to return top k results.
        - ``chat_session (str, optional)``: Chat session to use.
        

Raises:
    - ``ValueError``: If `prompt_config` is not provided.
    - ``ValueError``: If `llm_provider` is not provided.

Returns:
    - An instance of MaxAgentQA.

>>> from maxaillm.app.agent import MaxAgentQA
>>> agent = MaxAgentQA(llm_provider="anthropic",model_name ="claude-2", chunk_size=1000, stream=True, collection="myCollection", prompt_config=myPromptConfig)
