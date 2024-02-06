Data Processing
==============

Chunking
********

MarkdownHeaderSplitter
^^^^^^^^^^^^^^^^^^^^^^^^
MarkdownHeaderSplitter is a class that provides functionality for splitting text based on Markdown headers.

Args:
    - ``splits (List[Tuple[str, str]], optional)``: A list of tuples representing Markdown header patterns to split on. Each tuple contains two strings: the Markdown header prefix (e.g., "#") and the corresponding header name (e.g., "Header 1"). If not provided, default header patterns will be used.

Attributes:
    - ``headers_to_split_on (List[Tuple[str, str]])``: The Markdown header patterns to split on.

Methods:
    - ``split_text(text)``: Splits the given text based on Markdown headers.

        - ``text (str)``: The text to split.

    - ``split_document(document)``: Splits the text of the given document based on Markdown headers.

        - ``document (Document)``: The document to split.

Raises:
    - ``Exception``: If an error occurs while splitting the text.
    

HTMLHeaderSplitter
^^^^^^^^^^^^^^^^^^
HTMLHeaderSplitter is a class that provides functionality for splitting text based on HTML headers.

Args:
    - ``splits (List[Tuple[str, str]], optional)``: A list of tuples representing HTML header tags to split on. Each tuple contains two strings: the HTML header tag (e.g., "h1") and the corresponding header name (e.g., "Header 1"). If not provided, default header tags will be used.

Attributes:
    - ``headers_to_split_on (List[Tuple[str, str]])``: The HTML header tags to split on.

Methods:
    - ``split_text(text)``: Splits the given text based on HTML headers.

        - ``text (str)``: The text to split.

    - ``split_text_from_url(url)``: Fetches a document from the given URL and splits its text based on HTML headers.

        - ``url (str)``: The URL of the document to fetch and split.

    - ``split_document(document)``: Splits the text of the given document based on HTML headers.

        - ``document (Document)``: The document to split.

Raises:
    - ``Exception``: If an error occurs while splitting the text.
    
    
TextSplitter
^^^^^^^^^^^^
TextSplitter is a class that provides functionality for splitting text based on various methods.

Args:
    - ``chunk_size (int, optional)``: The maximum size of each chunk.
    - ``chunk_overlap (int, optional)``: The size of the overlap between chunks.
    - ``length_function (Callable[[str], int], optional)``: The function to use to calculate the length of a chunk.
    - ``separator (str, optional)``: The separator to use between chunks.
    - ``is_separator_regex (bool, optional)``: Whether the separator is a regular expression.
    - ``token_method_params (dict, optional)``: The parameters to use for the token method.

Attributes:
    - ``chunk_size (int)``: The maximum size of each chunk.
    - ``chunk_overlap (int)``: The size of the overlap between chunks.
    - ``length_function (Callable[[str], int])``: The function to use to calculate the length of a chunk.
    - ``separator (str)``: The separator to use between chunks.
    - ``is_separator_regex (bool)``: Whether the separator is a regular expression.
    - ``token_method_params (dict)``: The parameters to use for the token method.

Methods:
    - ``split_text(text, method)``: Splits the given text based on the specified method.

        - ``text (str)``: The text to split.
        - ``method (str)``: The method to use to split the text.

    - ``add_default_metadata_to_document(document, default_metadata)``: Adds default metadata to the given document.

        - ``document (Document)``: The document to add metadata to.
        - ``default_metadata (dict, optional)``: The default metadata to add.

    - ``create_documents(texts, file_metadata, metadata, default_metadata, method)``: Creates documents from the given texts.

        - ``texts (List[str])``: The texts to create documents from.
        - ``file_metadata (Dict[str, str], optional)``: The file metadata to add to the documents.
        - ``metadata (Dict[str, bool], optional)``: The metadata to add to the documents.
        - ``default_metadata (Dict[str, str], optional)``: The default metadata to add to the documents.
        - ``method (str)``: The method to use to split the texts.

    - ``dedup_chunks(chunks)``: Removes duplicate chunks from the given list.

        - ``chunks (List[str])``: The list of chunks to deduplicate.

    - ``split_document(document)``: Splits the text of the given document.

        - ``document (Document)``: The document to split.

    - ``serialize_datetime(obj)``: Serializes the given datetime object.

        - ``obj (datetime)``: The datetime object to serialize.

    - ``extract_metadata_generic(text, metadata_extraction_function, metadata_key)``: Extracts metadata from the given text.

        - ``text (str)``: The text to extract metadata from.
        - ``metadata_extraction_function (Callable[[str], Any])``: The function to use to extract metadata.
        - ``metadata_key (str)``: The key to use for the extracted metadata.

    - ``add_metadata_to_document(document, metadata_extraction_function, metadata_key)``: Adds metadata to the given document.

        - ``document (Document)``: The document to add metadata to.
        - ``metadata_extraction_function (Callable[[str], Any])``: The function to use to extract metadata.
        - ``metadata_key (str)``: The key to use for the extracted metadata.

    - ``add_metadata_to_documents_parallel(documents, metadata_extraction_function, metadata_key, max_workers)``: Adds metadata to the given documents in parallel.

        - ``documents (List[Document])``: The documents to add metadata to.
        - ``metadata_extraction_function (Callable[[str], Any])``: The function to use to extract metadata.
        - ``metadata_key (str)``: The key to use for the extracted metadata.
        - ``max_workers (int, optional)``: The maximum number of workers to use.

Raises:
    - ``Exception``: If an error occurs while splitting the text or adding metadata.
    
.. code-block:: python
    
    from maxaillm.data.chunking.TextSplitter import TextSplitter
    from maxaillm.data.extractor.MaxExtractor import MaxExtractor
    
    # extract the text from the document and clean the text
    me_obj = MaxExtractor()
    text, metadata = me.extract_text_metadata("path/to/file")
    clean_text = me.clean_text(
        text,
        dehyphenate=True, 
        ascii_only=True, 
        remove_isolated_symbols=True, 
        compress_whitespace=True
    )
    
    # define splitter
    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(
        [text],
        file_metadata=metadata,
        metadata={
            "default": True,
            "summary": False,
            "entities": False,
            "frequent_keywords": True,
            "links": True,
        },
        default_metadata={"file_name": "file_name"},
    )


Embeddings
**********

MaxHuggingFaceEmbeddings
^^^^^^^^^^^^^^^^^^^^^^^^
MaxHuggingFaceEmbeddings is a class that inherits from MaxLangchainEmbeddings and initializes a HuggingFaceEmbeddings model.

Args:
    - ``**kwargs``: Arbitrary keyword arguments for the HuggingFaceEmbeddings model.

Attributes:
    - ``model (MaxEmbeddingsBase)``: The translated MaxEmbeddingsBase model.
    
Methods:
    - ``embed_documents(texts)``: Embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``embed_query(text)``: Embeds the given query text.

        - ``text (str)``: The query text to embed.

    - ``aembed_documents(texts)``: Asynchronously embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``aembed_query(text)``: Asynchronously embeds the given query text.

        - ``text (str)``: The query text to embed.
        
.. code-block:: python

    from maxaillm.data.embeddings.MaxHuggingFaceEmbeddings import MaxHuggingFaceEmbeddings
    
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = MaxHuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


MaxLangchainEmbeddings
^^^^^^^^^^^^^^^^^^^^^^
MaxLangchainEmbeddings is a class that inherits from MaxEmbeddingsBase and provides methods for embedding texts.

Args:
    - ``model``: The model to use for embedding.

Attributes:
    - ``model (MaxEmbeddingsBase)``: The model used for embedding.

Methods:
    - ``embed_many(texts)``: Embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``embed(text)``: Embeds the given query text.

        - ``text (str)``: The query text to embed.

    - ``embed_many_async(texts)``: Asynchronously embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``embed_async(text)``: Asynchronously embeds the given query text.

        - ``text (str)``: The query text to embed.
        
        
Extractor
*********

MaxExtractor
^^^^^^^^^^^^
MaxExtractor is a class that inherits from MaxExtractorBase and MaxLLMBase and provides methods for extracting text, pages, details, tables, and metadata from documents.

Args:
    - ``parser_class_map_override (Optional[Dict[str, MaxExtractorBase]])``: A dictionary with a mapping of extensions to parser classes to merge with or override the defaults.

Attributes:
    - ``parser_class_map``: A dictionary with a mapping of extensions to parser classes.
    - ``supported_extensions``: A set of supported extensions.

Methods:
    - ``get_parser(extension)``: Returns the appropriate parser for a file type.

        - ``extension (str)``: The file extension.

    - ``get_extension_from_path(path)``: Gets the file extension from a path.

        - ``path (Union[str, Path])``: The file path.

    - ``get_extension(document, extension)``: Gets the file extension from a path or a named file-like object.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

    - ``extract_text(document, extension, ocr, **kwargs)``: Extracts text from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``ocr (bool)``: Whether to use OCR.

    - ``extract_pages(document, extension, ocr, **kwargs)``: Extracts pages from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``ocr (bool)``: Whether to use OCR.

    - ``extract_details(document, extension, ocr)``: Extracts details from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``ocr (bool)``: Whether to use OCR.

    - ``extract_tables(document, extension)``: Extracts tables from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

    - ``extract_metadata(document, extension)``: Extracts metadata from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

    - ``split_document(document, extension, split_size)``: Splits a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``split_size (int)``: The split size.

    - ``to_pdf(document, extension)``: Converts a document to PDF.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

Raises:
    - ``Exception``: If the extension must be provided and the document does not refer to a path or a named file-like object.
    - ``ValueError``: If the Azure Storage connection string is not found in environment variables or if the Azure blob path format is invalid.
    
.. code-block:: python

    from maxaillm.data.extractor.MaxExtractor import MaxExtractor
    
    
    me_obj = MaxExtractor()
    text, metadata = me.extract_text_metadata("path/to/file")
    
    # clean the text
    clean_text = me.clean_text(
        text,
        dehyphenate=True, 
        ascii_only=True, 
        remove_isolated_symbols=True, 
        compress_whitespace=True
    )
    

Retriever
*********

HyDE
^^^^^^
HyDE is a class that inherits from Retriever and provides methods for retrieving documents using a Hypothetical-Deductive Engine.

Args:
    - ``search_type``: The type of search to perform.
    - ``vectordb (MaxLangchainVectorStore)``: The vector database used for document retrieval.
    - ``llm (LLM)``: The language model used for generating hypothetical answers.
    - ``search_args``: The arguments for the search.

Attributes:
    - ``llm (LLM)``: The language model used for generating hypothetical answers.
    - ``vectordb (MaxLangchainVectorStore)``: The vector database used for document retrieval.
    - ``search_args``: The arguments for the search.
    - ``search_type``: The type of search to perform.
    - ``_template (str)``: A template string used for generating hypothetical answers.

Methods:
    - ``_get_relevant_documents(query)``: Generates a hypothetical answer for the given query.

        - ``query (str)``: The input query string.

    - ``retrieve(query)``: Retrieves documents based on the hypothetical answer generated for the query.

        - ``query (str)``: The input query string.
        
MultiQuery  
^^^^^^^^^^^^
MultiQuery is a class that inherits from Retriever and provides methods for retrieving documents using multiple queries.

Args:
    - ``search_type``: The type of search to perform.
    - ``vectordb (MaxLangchainVectorStore)``: The vector database used for document retrieval.
    - ``llm (LLM)``: The language model used for query expansion.
    - ``search_args``: The arguments for the search.

Attributes:
    - ``retriever (MultiQueryRetriever)``: The retriever used for document retrieval.

Methods:
    - ``retrieve(query)``: Retrieves documents using multiple queries generated from the input query.

        - ``query (str)``: The input query string.
        
HybridSearch
^^^^^^^^^^^^
HybridSearch is a class that inherits from Retriever and provides methods for retrieving documents using a hybrid approach.

Args:
    - ``search_type``: The type of search to perform.
    - ``vectordb (MaxLangchainVectorStore)``: The vector database used for document retrieval.
    - ``llm (LLM)``: The language model used for query expansion.
    - ``collection_desc (str)``: Description of the document collection.
    - ``metadata_schema (dict)``: Schema for the metadata associated with the documents.

Attributes:
    - ``retriever (SelfQueryRetriever)``: The retriever used for document retrieval.

Methods:
    - ``retrieve(query)``: Retrieves documents based on a hybrid approach combining vector search and language model query expansion.

        - ``query (str)``: The input query string.
        
LostInMiddle
^^^^^^^^^^^^
LostInMiddle is a class that inherits from ReRanker and provides methods for reordering documents based on longer context.

Attributes:
    - ``reranker (LongContextReorder)``: The reranker used for document reordering.

Methods:
    - ``rerank(query, docs)``: Reranks documents based on the given query.

        - ``query (str)``: The input query string.
        - ``docs (List[Document])``: The list of input documents.
        
Cohere
^^^^^^
Cohere is a class that inherits from ReRanker and provides methods for reordering documents based on the query using cohere.

Attributes:
    - ``reranker (CohereRerank)``: The reranker used for document reordering.

Methods:
    - ``rerank(query, docs)``: Reranks documents based on the given query.

        - ``query (str)``: The input query string.
        - ``docs (List[Document])``: The list of input documents.
        
MaxRetriever
^^^^^^^^^^^^
MaxRetriever is a class that inherits from MaxLLMMixin and provides a single interface for using different retrieval and reranking methods.

Args:
    - ``vectordb (MaxLangchainVectorStore)``: The vector database used for document retrieval.
    - ``llm (LLM, optional)``: The language model.
    - ``search_type (str, optional)``: The type of search to perform. Default is "mmr".
    - ``retriever_type (str, optional)``: The type of retriever to use. Default is an empty string.
    - ``reranker_type (str, optional)``: The type of reranker to use. Default is an empty string.
    - ``k (int, optional)``: The number of documents to retrieve. Default is 10.
    - ``score_threshold (float, optional)``: The score threshold for document retrieval. Default is 0.5.
    - ``filters (dict, optional)``: The filters to apply during document retrieval. Default is an empty dictionary.

Attributes:
    - ``vectordb``: The vector database used for document retrieval.
    - ``llm``: The language model.
    - ``retriever_type``: The type of retriever to use.
    - ``reranker_type``: The type of reranker to use.
    - ``search_type``: The type of search to perform.
    - ``search_args``: The arguments for the search.
    - ``retriever``: The retriever used for document retrieval.
    - ``reranker``: The reranker used for document reranking.

Methods:
    - ``_init_retriever(retriever_type)``: Initializes the specific retriever based on the retriever_type.

        - ``retriever_type (str)``: The type of retriever to initialize.

    - ``_init_reranker(reranker_type)``: Initializes the specific reranker based on the reranker_type.

        - ``reranker_type (str)``: The type of reranker to initialize.

    - ``retrieve_and_rerank(query)``: Retrieves documents using the configured retriever and then reranks them using the configured reranker.

        - ``query (str)``: The input query string.
        
.. code-block:: python
    
    from maxaillm.data.embeddings.MaxHuggingFaceEmbeddings import MaxHuggingFaceEmbeddings
    from maxaillm.data.retriever.Retriever import MaxRetriever
    from maxaillm.data.vectorstore.MaxPGVector import MaxPGVector
    
    
    # initialize embedding model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = MaxHuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # add docs to VectorDB
    conn_str = MaxPGVector.get_conn_string()
    vectordb = MaxPGVector(connection_string=conn_str, collection_name="collection_name", embedding_function=embeddings.to_langchain())
    vectordb.add(docs)
    
    # retrieve the text from VectorDB
    retrieve = MaxRetriever(vectordb=vectordb, llm=llm, reranker_type="LostInMiddle", k=2)
    output = retrieve.retrieve_and_rerank("some question?")
    
    
Vector store
*************

MaxLangchainVectorStore
^^^^^^^^^^^^^^^^^^^^^^^^
MaxLangchainVectorStore is a class that inherits from MaxVectorStoreBase and represents a vector store for language chains. It includes specific configurations for the language chain vector store.

Args:
    - ``vectorstore (VectorStore)``: The vector store to be used.

Attributes:
    - ``vectorstore``: The vector store used.
    - ``supported_search_types``: The types of search supported by this vector store.

Methods:
    - ``add(data, metadata, **kwargs)``: Adds a vector to the vector store.

        - ``data (Union[str, List[str], LangchainDocument, List[LangchainDocument]])``: The data to be added.
        - ``metadata (Union[dict, List[dict], None], optional)``: The metadata for the data. Default is None.

    - ``add_async(data, metadata, **kwargs)``: Asynchronously adds a vector to the vector store.

    - ``delete(ids, **kwargs)``: Deletes vectors corresponding to ids from the vector store.

        - ``ids (Union[List[str], str])``: The ids of the vectors to be deleted.

    - ``delete_async(ids, **kwargs)``: Asynchronously deletes vectors corresponding to ids from the vector store.

    - ``search(query, k, search_type, score, metadata_filter, return_metadata, **kwargs)``: Performs a query on the vector store.

        - ``query (Union[str, List[float]])``: The query to be performed.
        - ``k (int, optional)``: The number of results to return. Default is 3.
        - ``search_type (str, optional)``: The type of search to be performed. Default is "similarity".
        - ``score (bool, optional)``: Whether to return the score. Default is False.
        - ``metadata_filter (Union[dict, None], optional)``: The metadata filter for the search. Default is None.
        - ``return_metadata (bool, optional)``: Whether to return the metadata. Default is True.

    - ``search_async(query, k, search_type, score, metadata_filter, return_metadata, **kwargs)``: Asynchronously performs a query on the vector store.

    - ``to_langchain()``: Converts the vector store to a class implementing the langchain vector db interface.
    
    
MaxMilvus
^^^^^^^^^^
MaxMilvus is a class that inherits from MaxLangchainVectorStore and represents a Milvus vector store. It includes specific configurations for the Milvus vector store.

Args:
    - ``embedding_function (Union[Embeddings, MaxEmbeddingsBase])``: The embedding function to be used.
    - ``collection_name (str)``: The name of the collection.
    - ``connection_args (Optional[Dict[str, Any]])``: The arguments for the connection. Default is None.
    - ``consistency_level (str, optional)``: The consistency level. Default is "Session".
    - ``index_params (Optional[dict], optional)``: The parameters for the index. Default is None.
    - ``search_params (Optional[dict], optional)``: The parameters for the search. Default is None.
    - ``drop_old (Optional[bool], optional)``: Whether to drop the old data. Default is False.
    - ``primary_field (str, optional)``: The primary field. Default is "pk".
    - ``text_field (str, optional)``: The text field. Default is "text".
    - ``vector_field (str, optional)``: The vector field. Default is "vector".

Attributes:
    - ``vectorstore``: The Milvus vector store used.

Methods:
    - ``delete(ids, **kwargs)``: Deletes vectors corresponding to ids from the vector store.

        - ``ids (Union[List[str], str])``: The ids of the vectors to be deleted.

    - ``delete_async(ids, **kwargs)``: Asynchronously deletes vectors corresponding to ids from the vector store.

    - ``get_search_types()``: Returns the types of search supported by this vector store.
    

MaxPGVector
^^^^^^^^^^^
MaxPGVector is a class that inherits from MaxLangchainVectorStore and represents a PostgreSQL vector store. It includes specific configurations for the PostgreSQL vector store.

Args:
    - ``connection_string (str)``: The connection string for the PostgreSQL database.
    - ``embedding_function (Union[Embeddings, MaxEmbeddingsBase])``: The embedding function to be used.
    - ``collection_name (str, optional)``: The name of the collection. Default is DEFAULT_COLLECTION_NAME.
    - ``collection_metadata (Optional[Dict])``: The metadata for the collection. Default is None.
    - ``distance_strategy (DistanceStrategy, optional)``: The distance strategy to be used. Default is DEFAULT_DISTANCE_STRATEGY.
    - ``pre_delete_collection (bool, optional)``: Whether to delete the collection before creating a new one. Default is False.
    - ``logger (Optional[logging.Logger])``: The logger to be used. Default is None.
    - ``relevance_score_fn (Optional[Callable[[float], float]])``: The function to calculate the relevance score. Default is None.
    - ``connection (Optional[sqlalchemy.engine.Connection])``: The SQLAlchemy connection to be used. Default is None.
    - ``engine_args (Optional[Dict[str, Any]])``: The arguments for the SQLAlchemy engine. Default is None.

Attributes:
    - ``vectorstore``: The PostgreSQL vector store used.

Methods:
    - ``get_conn_string()``: Returns the connection string for the PostgreSQL database.

    - ``search_metadata(metadata_filter, k, return_metadata)``: Queries the collection.

        - ``metadata_filter (Optional[Dict[str, str]])``: The filter for the metadata. Default is None.
        - ``k (int)``: The number of results to return. Default is -1.
        - ``return_metadata (bool)``: Whether to return the metadata. Default is True.

    - ``delete_async(ids, **kwargs)``: Asynchronously deletes vectors corresponding to ids from the vector store.

        - ``ids (Union[str, List[str]])``: The ids of the vectors to be deleted.

    - ``result_to_vector_result(pg_result)``: Converts the result from the PostgreSQL query to a vector result.

    - ``search(query, k, search_type, score, metadata_filter, return_metadata, **kwargs)``: Searches the vector store.

    - ``search_async(query, k, search_type, score, metadata_filter, return_metadata, **kwargs)``: Asynchronously searches the vector store.

    - ``drop()``: Deletes the vector collection and documents from the PostgreSQL vector store.

Raises:
    - ``EnvironmentError``: If the environment variables for the PostgreSQL database are not set.
    - ``ValueError``: If neither query nor metadata_filter is set in the search and search_async methods.
    
    
MaxRedis
^^^^^^^^
MaxRedis is a class that inherits from MaxLangchainVectorStore and represents a Redis vector store. It includes specific configurations for the Redis vector store.

Args:
    - ``index_name (str)``: The name of the index.
    - ``embedding_function (Union[Embeddings, MaxEmbeddingsBase])``: The embedding function to be used.
    - ``redis_url (str, optional)``: The Redis URL. Default is None.
    - ``index_schema (Optional[Union[Dict[str, str], str, os.PathLike]], optional)``: The schema for the index. Default is None.
    - ``vector_schema (Optional[Dict[str, Union[str, int]]], optional)``: The schema for the vector. Default is None.
    - ``relevance_score_fn (Optional[Callable[[float], float]], optional)``: The function to calculate the relevance score. Default is None.
    - ``key_prefix (Optional[str], optional)``: The prefix for the key. Default is None.

Attributes:
    - ``redis_url``: The Redis URL used.
    - ``vectorstore``: The Redis vector store used.
    - ``schema``: The schema used.

Methods:
    - ``_build_schema(index_name)``: Builds the schema for the index.

        - ``index_name (str)``: The name of the index.

    - ``add(data, metadata, **kwargs)``: Adds data to the vector store.

        - ``data (Union[str, List[str], LangchainDocument, List[LangchainDocument]])``: The data to be added.
        - ``metadata (Union[dict, List[dict], None], optional)``: The metadata for the data. Default is None.

    - ``add_async(data, metadata, **kwargs)``: Asynchronously adds data to the vector store.

    - ``delete(ids)``: Deletes vectors corresponding to ids from the vector store.

        - ``ids (Union[List[str], str])``: The ids of the vectors to be deleted.

    - ``delete_async(ids, **kwargs)``: Asynchronously deletes vectors corresponding to ids from the vector store.

    - ``search(query, k, search_type, score, metadata_filter, return_metadata, **kwargs)``: Searches the vector store.

        - ``query (any, optional)``: The query for the search. Default is None.
        - ``k (int, optional)``: The number of results to return. Default is 3.
        - ``search_type (str, optional)``: The type of search. Default is "similarity".
        - ``score (bool, optional)``: Whether to return the score. Default is False.
        - ``metadata_filter (Union[dict, None], optional)``: The filter for the metadata. Default is None.
        - ``return_metadata (bool, optional)``: Whether to return the metadata. Default is True.

    - ``search_async(query, k, search_type, score, metadata_filter, return_metadata, **kwargs)``: Asynchronously searches the vector store.

    - ``redis_result_to_vector_result(redis_result)``: Converts a Redis result to a vector result.

        - ``redis_result (any)``: The Redis result to be converted.

    - ``get_redis_filters(metadata)``: Gets the Redis filters for the metadata.

        - ``metadata (dict)``: The metadata for which to get the filters.

    - ``search_metadata(metadata_filter, k, return_metadata)``: Searches the metadata.

        - ``metadata_filter (dict)``: The filter for the metadata.
        - ``k (int, optional)``: The number of results to return. Default is -1.
        - ``return_metadata (bool, optional)``: Whether to return the metadata. Default is True.

    - ``drop()``: Drops the vector store.

Raises:
    - ``ValueError``: If one of query or metadata_filter is not set for the search and search_async methods.
    - ``ValueError``: If all metadata filter keys are not present in the vector store index schema for the get_redis_filters method.