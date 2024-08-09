Application
===========

The development layer abstracts the intricacies of generators, tokenization, and memory techniques. Instead of writing numerous lines of code for setting up and serving, users can initiate their application with just a few method calls.

Cache
******

MaxCache
^^^^^^^^^
MaxCache is a class that provides caching functionality for language model outputs. For enabling MaxCache, set ``REDIS_CONNECTION_STRING`` as an environment variable.

Attributes:
    - ``redis_url (str)``: The connection string for the Redis server.
    - ``cache``: The cache object, which can be a RedisSemanticCache or a RedisCache.

Args:
    - ``semantic (bool, optional)``: Whether to use semantic caching. Defaults to False.
    - ``ttl (int, optional)``: The time-to-live for cache entries in seconds. Defaults to None, which means entries do not expire.
    - ``cache_type (str, optional)``: The type of cache to use. Currently, only 'redis' is supported. Defaults to 'redis'.
    - ``embedding_model (optional)``: The embedding model to use for semantic caching. Required if semantic is True.

Raises:
    - ``ValueError``: If an unsupported cache type is provided or if semantic is True but no embedding model is provided.

Methods:
    - ``lookup``: Looks up a cache entry based on a prompt and a language model string.

        - ``prompt (str)``: The prompt to look up.
        - ``llm_string (str)``: The language model string to look up.

    - ``update``: Updates the cache with a new entry.

        - ``prompt (str)``: The prompt for the new entry.
        - ``llm_string (str)``: The language model string for the new entry.
        - ``output (str)``: The output to cache.

Returns:
    - ``lookup``: The cached output for the given prompt and language model string, or None if no entry was found.
    
>>> from maxaillm.app.cache.cache import MaxCache
>>> cache = MaxCache(
...     semantic=True,
...     embedding_model=emb.model
... )
>>> collection = "collection_name"
>>> query = "question asked by user"
>>> resp = "response by LLM"
>>> cache.update(query, collection, resp)    # save the results for a particular collection


Generator
************

MaxGenerator
^^^^^^^^^^^^
A generator is used for generating responses from LLM.
It is capable of generating both single responses and a stream of responses based on queries, context, and conversational history.

Args:
    - ``llm (LLM)``: The large language model instance.
    - ``method (str)``: The method to be used for generating responses or processing text.
    - ``prompt_config (dict)``: Configuration settings for prompts.
    - ``streamable (bool, Optional)``: Flag to indicate if the data should be processed as a stream. Defaults to False.
    - ``context_window (int, Optional)``: The size of the context window in terms of the number of tokens. Defaults to 10000.
    - ``verbose (bool, Optional)``: Flag to indicate if verbose mode is enabled. Defaults to True.
    - ``chat_history (bool, Optional)``: Flag to indicate if the conversation history should be maintained. Defaults to False.

Attributes:
    - ``streamable``: A flag indicating whether the generator is capable of streaming responses.
    - ``llm``: An instance of a language model (LLM) used for generating responses.
    - ``method``: The method used for generating responses.
    - ``context_window``: The size of the context window for generating responses.
    - ``prompt_config``: Configuration settings for prompts.
    - ``verbose``: Flag to indicate if verbose mode is enabled.
    - ``chat_history``: Flag to indicate if the conversation history should be maintained.

Methods:
    - ``_create_template``: Creates a chat prompt template based on the configuration and chat history.

        - ``None``: No arguments.

    - ``_initialize_chain``: Initializes the question-answering chain with a specific prompt template and method.

        - ``None``: No arguments.

    - ``generate``: Generates a response based on queries, context, and conversational history.

        - ``query (str)``: The user's query.
        - ``context (List[str])``: List of contexts associated with each query.
        - ``conversation (List[dict], Optional)``: Conversational context. Defaults to an empty list.

    - ``generate_async``: Asynchronously generates a response based on queries, context, and conversational history.

        - ``query (str)``: The user's query.
        - ``context (List[str])``: List of contexts associated with each query.
        - ``conversation (List[dict], Optional)``: Conversational context. Defaults to an empty list.

    - ``generate_stream``: Generates a stream of responses based on queries, context, and conversational history.

        - ``query (str)``: The user's query.
        - ``context (List[str])``: List of contexts associated with each query.
        - ``conversation (List[dict], Optional)``: Conversational context. Defaults to an empty list.

    - ``prepare_messages``: Prepares messages formatted for a chatbot system using GPT-4 model.

        - ``context (List[str])``: Context information for the queries.
        - ``conversation (List[dict], Optional)``: Previous conversation messages with role and content. Defaults to an empty list.
        
.. code-block:: python

    from maxaillm.app.generator.MaxGenerator import MaxGenerator
    
    
    # define prompt configuration
    p_conf = {'moderations':'', 'task':'', 'identity':''}
    
    # initialize MaxGenerator
    mg = MaxGenerator(llm=llm, method='stuff', prompt_config=p_conf, engine="langchain")
    
    # generate batch response
    mg.generate(query='Explain Reinforcement Learning', context=out)
    
    # to generate 
    mg.generate_stream(query='Explain Reinforcement Learning', context=out)
        
        
Memory
******

MaxMemory
^^^^^^^^^
MaxMemory is a class that provides functionality for managing chat message history in a PostgreSQL database.

Args:
    - ``session (type)``: The ID of the chat session.

Attributes:
    - ``connection_string (str)``: The connection string for the PostgreSQL database.
    - ``session_id (str)``: The ID of the current chat session.
    - ``history (MaxChatMessageHistory)``: The chat message history.
    
Raises:
    - ``Exception``: If the necessary environment variables for the database connection are not set.

Methods:
    - ``add_message(message)``: Adds a message to the chat history.

        - ``message (dict)``: The message to add.

    - ``clear()``: Clears the chat history.

    - ``get_message_history(n)``: Returns the last n messages from the chat history.

        - ``n (int)``: The number of messages to return.

    - ``get_chat_sessions(sessions)``: Returns the chat history for the given sessions.

        - ``sessions (list, optional)``: The IDs of the sessions to return the chat history for.