App
====

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