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
This class manages multiple generators for response generation and allows switching between different generator implementations.

Initializes the GeneratorManager and sets up available generators.

Args:
    - ``llm (LLM)``: The language model to be used.
    - ``method (str)``: The method to be used for generation.
    - ``prompt_config (dict)``: The configuration for the prompt.
    - ``engine (str, optional)``: The engine to be used for generation. Defaults to "langchain".
    - ``streamable (bool, optional)``: Indicates if the generator supports streaming responses. Defaults to False.
    - ``verbose (bool, optional)``: Indicates if verbose output should be enabled. Defaults to True.

Attributes:
    - ``generators (dict[str, Generator])``: A dictionary mapping generator names to their instances.
    - ``selected_generator (Generator)``: The currently selected generator for response generation.

Methods:
    - ``generate(query, context, conversation)``: Generates a response based on the query and context.

        - ``query (str)``: The query to generate a response for.
        - ``context (List[str])``: The context for the query.
        - ``conversation (List[dict], optional)``: The conversation history.

    - ``generate_stream(query, context, conversation)``: Asynchronously generates a response based on the query and context.

        - ``query (str)``: The query to generate a response for.
        - ``context (List[str])``: The context for the query.
        - ``conversation (List[dict], optional)``: The conversation history.

    - ``set_generator(generator)``: Sets the current generator.

        - ``generator (str)``: The name of the generator to set.

    - ``get_generators()``: Returns the available generators.

    - ``calculate_tokens(prompt_config, context, query, chat)``: Calculates the number of tokens in the formatted response.

        - ``prompt_config (dict)``: The configuration for the prompt.
        - ``context (List[str])``: The context for the query.
        - ``query (str)``: The query to generate a response for.
        - ``chat (str)``: The chat history.
        
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