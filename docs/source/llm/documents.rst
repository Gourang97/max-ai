Document Processing
===================

Max.AI LLM provides comprehensive document processing capabilities for extracting, processing, and analyzing various document formats. The document processing engine supports multiple file types and offers advanced text extraction, cleaning, and structuring features.

Overview
--------

The Max.AI Document Processing Engine offers:

* **Multi-Format Support**: PDF, DOCX, PPTX, HTML, Markdown, and more
* **Advanced Text Extraction**: OCR support, table extraction, and metadata preservation
* **Intelligent Chunking**: Context-aware document splitting and segmentation
* **Content Cleaning**: Text normalization, noise removal, and formatting standardization
* **Structured Output**: Organized extraction with metadata and hierarchical structure

Supported Document Formats
---------------------------

**PDF Documents (MaxPDFExtractor)**
    Advanced PDF processing with OCR support and table extraction.

**Microsoft Office Documents**
    * **Word Documents (MaxDOCExtractor)**: DOCX file processing with formatting preservation
    * **PowerPoint Presentations (MaxPPTExtractor)**: Slide content and structure extraction

**Web Documents**
    * **HTML Documents (MaxHTMLExtractor)**: Web page content extraction with structure preservation
    * **Markdown Documents (MaxMDExtractor)**: Markdown parsing and content extraction

**Apache Tika Integration (MaxTikaBase)**
    Support for additional formats through Apache Tika integration.

Document Extractor
------------------

**MaxExtractor**
    Universal document extractor providing a unified interface for all document types.

Core Features:
    * Automatic format detection
    * Unified extraction API
    * Metadata preservation
    * Error handling and recovery
    * Remote file support (S3, Azure Blob, URLs)

Basic Usage:

.. code-block:: python

    from maxaillm.data.extractor.MaxExtractor import MaxExtractor
    
    # Initialize extractor
    extractor = MaxExtractor()
    
    # Extract text from document
    text, metadata = extractor.extract_text_metadata("path/to/document.pdf")
    
    # Clean extracted text
    clean_text = extractor.clean_text(
        text,
        dehyphenate=True,
        ascii_only=True,
        remove_isolated_symbols=True,
        compress_whitespace=True
    )
    
    print(f"Extracted {len(text)} characters")
    print(f"Metadata: {metadata}")

Advanced Extraction Methods:

.. code-block:: python

    # Extract pages separately
    pages = extractor.extract_pages("document.pdf")
    for i, page in enumerate(pages):
        print(f"Page {i+1}: {page[:100]}...")
    
    # Extract tables
    tables = extractor.extract_tables("document.pdf")
    for table in tables:
        print(f"Table with {len(table)} rows")
    
    # Extract detailed information
    details = extractor.extract_details("document.pdf")
    print(f"Document details: {details}")
    
    # Convert to PDF
    pdf_bytes = extractor.to_pdf("document.docx")

PDF Processing
--------------

**MaxPDFExtractor**
    Specialized PDF processing with advanced features.

Features:
    * Text extraction with layout preservation
    * OCR support for scanned documents
    * Table detection and extraction
    * Image extraction
    * Metadata extraction
    * Page-by-page processing

.. code-block:: python

    from maxaillm.data.extractor.MaxPDFExtractor import MaxPDFExtractor
    
    # Initialize PDF extractor
    pdf_extractor = MaxPDFExtractor()
    
    # Extract with OCR support
    text = pdf_extractor.extract_text("scanned_document.pdf", ocr=True)
    
    # Extract tables
    tables = pdf_extractor.extract_tables("document_with_tables.pdf")
    
    # Extract metadata
    metadata = pdf_extractor.extract_metadata("document.pdf")
    print(f"Author: {metadata.get('author')}")
    print(f"Creation Date: {metadata.get('creation_date')}")

Office Document Processing
--------------------------

**MaxDOCExtractor**
    Microsoft Word document processing.

.. code-block:: python

    from maxaillm.data.extractor.MaxDOCExtractor import MaxDOCExtractor
    
    doc_extractor = MaxDOCExtractor()
    
    # Extract text with formatting
    text = doc_extractor.extract_text("document.docx")
    
    # Extract with structure preservation
    structured_content = doc_extractor.extract_structured_content("document.docx")

**MaxPPTExtractor**
    PowerPoint presentation processing.

.. code-block:: python

    from maxaillm.data.extractor.MaxPPTExtractor import MaxPPTExtractor
    
    ppt_extractor = MaxPPTExtractor()
    
    # Extract slide content
    slides = ppt_extractor.extract_slides("presentation.pptx")
    for i, slide in enumerate(slides):
        print(f"Slide {i+1}: {slide['title']}")
        print(f"Content: {slide['content']}")

Web Document Processing
-----------------------

**MaxHTMLExtractor**
    HTML document processing with structure preservation.

.. code-block:: python

    from maxaillm.data.extractor.MaxHTMLExtractor import MaxHTMLExtractor
    
    html_extractor = MaxHTMLExtractor()
    
    # Extract clean text from HTML
    text = html_extractor.extract_text("webpage.html")
    
    # Extract with structure
    structured_content = html_extractor.extract_structured_content("webpage.html")
    
    # Extract specific elements
    links = html_extractor.extract_links("webpage.html")
    images = html_extractor.extract_images("webpage.html")

**MaxMDExtractor**
    Markdown document processing.

.. code-block:: python

    from maxaillm.data.extractor.MaxMDExtractor import MaxMDExtractor
    
    md_extractor = MaxMDExtractor()
    
    # Extract text from markdown
    text = md_extractor.extract_text("document.md")
    
    # Extract with markdown structure
    structured_content = md_extractor.extract_structured_content("document.md")

Remote File Processing
----------------------

**RemoteFileReader**
    Support for processing files from remote sources.

Supported Sources:
    * Amazon S3
    * Azure Blob Storage
    * HTTP/HTTPS URLs
    * FTP servers

.. code-block:: python

    from maxaillm.data.extractor.utils.RemoteFileReader import RemoteFileReader
    
    # Process S3 file
    s3_url = "s3://bucket-name/path/to/document.pdf"
    text, metadata = extractor.extract_text_metadata(s3_url)
    
    # Process Azure Blob file
    azure_url = "https://account.blob.core.windows.net/container/document.pdf"
    text, metadata = extractor.extract_text_metadata(azure_url)
    
    # Process HTTP URL
    web_url = "https://example.com/document.pdf"
    text, metadata = extractor.extract_text_metadata(web_url)

Text Cleaning and Preprocessing
-------------------------------

**Advanced Text Cleaning**
    Comprehensive text cleaning and normalization.

.. code-block:: python

    # Clean extracted text
    clean_text = extractor.clean_text(
        raw_text,
        dehyphenate=True,           # Remove hyphenation
        ascii_only=True,            # Keep only ASCII characters
        remove_isolated_symbols=True,  # Remove standalone symbols
        compress_whitespace=True,   # Normalize whitespace
        remove_headers_footers=True,  # Remove headers/footers
        remove_page_numbers=True,   # Remove page numbers
        normalize_quotes=True,      # Normalize quotation marks
        fix_encoding=True          # Fix encoding issues
    )

**Custom Cleaning Rules**

.. code-block:: python

    def custom_cleaning_function(text):
        # Custom cleaning logic
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)  # Replace dates
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)  # Replace phone numbers
        return text
    
    # Apply custom cleaning
    cleaned_text = custom_cleaning_function(raw_text)

Document Chunking and Segmentation
-----------------------------------

**Intelligent Document Splitting**
    Context-aware document segmentation for better processing.

.. code-block:: python

    from maxaillm.data.chunking.TextSplitter import TextSplitter
    
    # Initialize text splitter
    splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separator="\n\n"
    )
    
    # Create documents with metadata
    documents = splitter.create_documents(
        texts=[clean_text],
        file_metadata=metadata,
        metadata={
            "default": True,
            "summary": False,
            "entities": True,
            "frequent_keywords": True,
            "links": True
        },
        default_metadata={"source": "document.pdf"}
    )

**Header-Based Splitting**

.. code-block:: python

    from maxaillm.data.chunking import MarkdownHeaderSplitter, HTMLHeaderSplitter
    
    # Markdown header splitting
    md_splitter = MarkdownHeaderSplitter(
        splits=[("#", "Chapter"), ("##", "Section"), ("###", "Subsection")]
    )
    md_chunks = md_splitter.split_text(markdown_text)
    
    # HTML header splitting
    html_splitter = HTMLHeaderSplitter(
        splits=[("h1", "Chapter"), ("h2", "Section"), ("h3", "Subsection")]
    )
    html_chunks = html_splitter.split_text(html_text)

Metadata Extraction and Enhancement
-----------------------------------

**Automatic Metadata Extraction**
    Extract and enhance document metadata automatically.

.. code-block:: python

    # Extract comprehensive metadata
    metadata = extractor.extract_metadata("document.pdf")
    
    # Common metadata fields
    print(f"Title: {metadata.get('title')}")
    print(f"Author: {metadata.get('author')}")
    print(f"Subject: {metadata.get('subject')}")
    print(f"Keywords: {metadata.get('keywords')}")
    print(f"Creation Date: {metadata.get('creation_date')}")
    print(f"Modification Date: {metadata.get('modification_date')}")
    print(f"Page Count: {metadata.get('page_count')}")
    print(f"File Size: {metadata.get('file_size')}")

**Custom Metadata Enhancement**

.. code-block:: python

    def extract_custom_metadata(text):
        # Custom metadata extraction logic
        metadata = {}
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        metadata['emails'] = emails
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b', text)
        metadata['phone_numbers'] = phones
        
        # Extract dates
        dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text)
        metadata['dates'] = dates
        
        return metadata
    
    # Add custom metadata to documents
    splitter.add_metadata_to_documents_parallel(
        documents=documents,
        metadata_extraction_function=extract_custom_metadata,
        metadata_key="custom_metadata",
        max_workers=4
    )

Batch Processing
----------------

**Bulk Document Processing**
    Process multiple documents efficiently.

.. code-block:: python

    import os
    from concurrent.futures import ThreadPoolExecutor
    
    def process_document(file_path):
        try:
            text, metadata = extractor.extract_text_metadata(file_path)
            clean_text = extractor.clean_text(text)
            
            return {
                'file': file_path,
                'text': clean_text,
                'metadata': metadata,
                'status': 'success'
            }
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'status': 'failed'
            }
    
    # Process multiple documents
    document_folder = "path/to/documents"
    file_paths = [os.path.join(document_folder, f) 
                  for f in os.listdir(document_folder)]
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_document, file_paths))
    
    # Process results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successfully processed: {len(successful)} documents")
    print(f"Failed to process: {len(failed)} documents")

Error Handling and Recovery
---------------------------

**Robust Error Handling**
    Handle various document processing errors gracefully.

.. code-block:: python

    def safe_extract_text(file_path):
        try:
            # Attempt extraction
            text, metadata = extractor.extract_text_metadata(file_path)
            return text, metadata, None
            
        except FileNotFoundError:
            return None, None, "File not found"
            
        except PermissionError:
            return None, None, "Permission denied"
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                text, metadata = extractor.extract_text_metadata(
                    file_path, encoding='latin-1'
                )
                return text, metadata, "Encoding issue resolved"
            except:
                return None, None, "Encoding error"
                
        except Exception as e:
            return None, None, f"Unexpected error: {str(e)}"

**Fallback Strategies**

.. code-block:: python

    def extract_with_fallback(file_path):
        # Try primary extraction method
        try:
            return extractor.extract_text(file_path)
        except:
            pass
        
        # Try OCR if primary method fails
        try:
            return extractor.extract_text(file_path, ocr=True)
        except:
            pass
        
        # Try alternative extractor
        try:
            alternative_extractor = MaxTikaExtractor()
            return alternative_extractor.extract_text(file_path)
        except:
            pass
        
        # Return empty string if all methods fail
        return ""

Performance Optimization
------------------------

**Memory Management**
    Optimize memory usage for large document processing.

.. code-block:: python

    # Process large documents in chunks
    def process_large_document(file_path, chunk_size=1000000):
        with open(file_path, 'r', encoding='utf-8') as file:
            chunks = []
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                
                # Process chunk
                clean_chunk = extractor.clean_text(chunk)
                chunks.append(clean_chunk)
        
        return ''.join(chunks)

**Caching**
    Cache extraction results for improved performance.

.. code-block:: python

    import hashlib
    import pickle
    import os
    
    class CachedExtractor:
        def __init__(self, cache_dir="./extraction_cache"):
            self.extractor = MaxExtractor()
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        
        def _get_cache_key(self, file_path):
            # Create cache key from file path and modification time
            stat = os.stat(file_path)
            key_string = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(key_string.encode()).hexdigest()
        
        def extract_text_cached(self, file_path):
            cache_key = self._get_cache_key(file_path)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Check cache
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Extract and cache
            text, metadata = self.extractor.extract_text_metadata(file_path)
            result = (text, metadata)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result

Best Practices
--------------

**Document Processing Pipeline**
    1. **Validate Input**: Check file existence and format
    2. **Extract Content**: Use appropriate extractor for file type
    3. **Clean Text**: Apply consistent cleaning rules
    4. **Chunk Content**: Split into manageable pieces
    5. **Extract Metadata**: Enhance with relevant information
    6. **Quality Check**: Validate extraction results
    7. **Store Results**: Save processed content and metadata

**Performance Tips**
    * Use parallel processing for multiple documents
    * Implement caching for frequently accessed files
    * Monitor memory usage with large documents
    * Use appropriate chunk sizes for your use case
    * Consider OCR only when necessary (it's slower)

**Quality Assurance**
    * Validate extraction results
    * Check for missing content
    * Verify metadata accuracy
    * Test with various document types
    * Monitor processing errors and failures

The Max.AI Document Processing Engine provides a comprehensive solution for extracting, cleaning, and structuring content from various document formats, enabling efficient document analysis and processing workflows.
