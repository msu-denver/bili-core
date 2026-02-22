"""
Module: file_utils

This module provides utility functions for file processing and logging. It includes
functions to load JSON data, preprocess directories and files, and handle various file
types such as CSV, Excel, Word, JSON, HTML, Markdown, PDF, XML, and text files.

Functions:
    - load_from_json(file_path, parameter=None):
      Loads and parses JSON data from a file, optionally returning a specific parameter's
      value.
    - preprocess_directory(data_dir):
      Preprocesses the contents of a directory by applying a preprocessing function to
      each file.
    - preprocess_file(file_path):
      Preprocesses a file by determining its extension and delegating the processing to a
      specific function.
    - process_csv_data(file_path, _):
      Processes data from a CSV file using the `CSVLoader` utility.
    - process_excel_data(file_path, _):
      Processes data from an Excel file using the `UnstructuredExcelLoader`.
    - process_word_data(file_path, _):
      Processes data from a Word document using the `UnstructuredWordDocumentLoader`.
    - process_json_data(file_path, _):
      Processes JSON data from a file using the `JSONLoader`.
    - process_html_data(file_path, _):
      Processes HTML data from a file using the `BSHTMLLoader`.
    - process_markdown_data(file_path, _):
      Processes Markdown data from a file using the `UnstructuredMarkdownLoader`.
    - process_pdf_data(file_path, _):
      Processes PDF data from a file using the `PyPDFLoader`.
    - process_xml_data(file_path, _):
      Processes XML data from a file using the `UnstructuredXMLLoader`.
    - process_text_data(file_path, _):
      Processes text data from a file using the `TextLoader`.
    - process_unstructured_data(file_path, _):
      Processes unstructured data from a file using the `UnstructuredFileLoader`.

Dependencies:
    - json: Provides functions to parse JSON data.
    - os: Provides a way of using operating system dependent functionality.
    - langchain_community.document_loaders: Imports various document loaders for different
      file types.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used for file processing tasks within the application. It
    provides functions to load and preprocess various file types, making it easier to handle
    different data formats.

Example:
    from bili.utils.file_utils import load_from_json

    # Load JSON data from a file
    data = load_from_json("path/to/file.json")
"""

import json
import os

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredXMLLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bili.utils.logging_utils import get_logger

# Initialize logger for this module
LOGGER = get_logger(__name__)

# Default chunk parameters for text splitting
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def load_from_json(file_path, parameter=None):
    """
    Load data from a JSON file and optionally retrieve a specific parameter value.

    This function reads a JSON file from the specified file path and loads its
    contents into a Python data structure (e.g., dictionary or list). Optionally,
    a specific parameter value can be retrieved if it exists in the JSON data. If
    the parameter is not found, the entire data content is returned.

    :param file_path:
        The path to the JSON file to be loaded. Must be a valid file path as a string.
    :param parameter:
        The key of the parameter to retrieve from the loaded JSON data. Defaults to
        None. If specified, the function attempts to extract the associated value
        from the data structure.
    :return:
        The value associated with the `parameter` key in the JSON data if it exists,
        otherwise, the full loaded data structure.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # If the JSON data has a 'parameter' key, return the parameter
    if parameter in data:
        return data[parameter]
    return data


def preprocess_directory(data_dir):
    """
    Preprocesses a directory of files, creating and constructing a list of processed
    documents by utilizing a specific preprocessing function. Files in the directory
    are filtered to exclude hidden files and subdirectories. Each valid file is processed,
    with debug logs available for each processed document.

    :param data_dir: The path to the directory containing files to preprocess.
    :type data_dir: str
    :return: A list of processed documents generated from the files in the directory.
    :rtype: list
    """
    LOGGER.info("Preprocessing data directory: %s", data_dir)

    processed_documents = []

    for file in os.listdir(data_dir):
        # Ignore hidden files and subdirectories
        if file.startswith(".") or os.path.isdir(os.path.join(data_dir, file)):
            continue

        file_path = os.path.join(data_dir, file)
        processed_documents.extend(preprocess_file(file_path))

        # Print all processed documents for debugging
        for doc in processed_documents:
            LOGGER.debug(doc)

    return processed_documents


def preprocess_file(file_path):
    """
    Processes a given file based on its file extension. The function determines
    the file type and applies an appropriate processing function. If the
    file's extension does not match any pre-defined types, a default
    processing function for unstructured data is applied.

    :param file_path: Path to the file to be processed. It can be a relative or
        absolute path string.
    :type file_path: str
    :return: Processed data or an empty string in case of an error during file
        processing.
    :rtype: str
    :raises IOError: Raised if there are issues in opening or reading the file.
    :raises ValueError: Raised if there are issues in the content or structure
        of the file.
    """
    LOGGER.debug(f"Processing file {file_path}...")

    # Extract the file extension from the file path
    file_extension = os.path.splitext(file_path)[1].lower()

    # Process the file based on its extension
    try:
        extension_to_function = {
            ".csv": process_csv_data,
            ".xls": process_excel_data,
            ".xlsx": process_excel_data,
            ".xlsm": process_excel_data,
            ".xlm": process_excel_data,
            ".doc": process_word_data,
            ".docx": process_word_data,
            ".json": process_json_data,
            ".html": process_html_data,
            ".md": process_markdown_data,
            ".pdf": process_pdf_data,
            ".xml": process_xml_data,
            ".txt": process_text_data,
        }
        return extension_to_function.get(file_extension, process_unstructured_data)(
            file_path, file_extension
        )
    except (IOError, ValueError) as e:
        LOGGER.error(f"Error processing file {file_path}: {e}")
        return ""


# Function to process CSV files
# https://python.langchain.com/docs/modules/data_connection/document_loaders/csv
def process_csv_data(file_path, _):
    """
    Processes CSV data using a CSVLoader instance.

    This function loads data from a CSV file provided as input and returns the
    processed data. It uses the CSVLoader class for handling the file reading and
    parsing operations.

    :param file_path: Path to the CSV file to be processed.
    :type file_path: str
    :param _: Additional placeholder argument, currently unused.
    :type _: Any
    :return: Processed data loaded from the CSV file.
    :rtype: Any
    """
    loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
        },
    )
    return loader.load()


# Function to process Excel files
# https://python.langchain.com/docs/integrations/document_loaders/excel
def process_excel_data(file_path, _):
    """
    Processes Excel data using the UnstructuredExcelLoader.

    This function takes the path to an Excel file and utilizes an
    instance of the UnstructuredExcelLoader to process the file's
    content. It can process the data in "elements" mode. The loaded
    data is subsequently returned for further usage.

    :param file_path: Path to the Excel file to process.
    :type file_path: str
    :param _: Unused parameter for compatibility. Allows flexibility
        for potential future extension.
    :return: Processed elements extracted from the Excel file.
    :rtype: list
    """
    # Might want to loop through all sheets in the Excel file and use the
    # metadata.text_as_html to process as HTML instead
    excel_document = UnstructuredExcelLoader(file_path, mode="elements")
    return excel_document.load()


# Function to process Word files
# https://python.langchain.com/docs/integrations/document_loaders/microsoft_word
def process_word_data(file_path, _):
    """
    Processes word document data by utilizing an unstructured Word document
    loader to extract elements and retrieve the content.

    :param file_path: The path to the Word document to process.
    :param _: A placeholder argument not used directly within the function.
    :return: A list of extracted elements from the Word document.
    """
    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    return loader.load()


# Function to process JSON files
# https://python.langchain.com/docs/modules/data_connection/document_loaders/json
def process_json_data(file_path, _):
    """
    Processes JSON data from a given file path.

    This function leverages the JSONLoader utility to read and process data from a given
    file path. It initializes the loader instance with specific parameters such as JSON
    schema and ensures the data is treated as JSON lines.

    :param file_path: The path to the JSON file to be processed.
    :type file_path: str
    :param _: Reserved for future arguments or features, currently unused.
    :type _: Any
    :return: Loaded data from the JSON file.
    :rtype: Any
    """
    loader = JSONLoader(file_path=file_path, jq_schema=".", json_lines=True)
    return loader.load()


# Function to process HTML files
# https://python.langchain.com/docs/modules/data_connection/document_loaders/html
def process_html_data(file_path, _):
    """
    Processes HTML data from a given file path using BSHTMLLoader.

    This function utilizes the BSHTMLLoader class to load and parse HTML
    content from the provided file path. It abstracts the loading process
    and returns the parsed HTML data.

    :param file_path: Path to the HTML file to be loaded and processed.
    :type file_path: str
    :param _: Additional parameter which is not used within this function.
    :return: Parsed HTML data loaded by BSHTMLLoader.
    :rtype: Any
    """
    loader = BSHTMLLoader(file_path)
    return loader.load()


# Function to process Markdown files
# https://python.langchain.com/docs/modules/data_connection/document_loaders/markdown
def process_markdown_data(file_path, _):
    """
    Processes markdown file data and returns the loaded content.

    This function utilizes the UnstructuredMarkdownLoader to load elements
    from a specified markdown file.

    :param file_path: File path to the markdown file to be processed.
    :type file_path: str
    :param _: Placeholder parameter, not used within the function.
    :return: The loaded content from the given markdown file, represented
        as a list of elements.
    :rtype: list
    """
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    return loader.load()


# Function to process PDF files
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
# We may want to consider using MathPix for complex math equations in PDF documents,
# as might happen with academic documents
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#using-mathpix
def process_pdf_data(file_path, _):
    """
    Processes the PDF data at the given file path by attempting to load and split
    its contents. Initially, it attempts to extract images during processing;
    if an error occurs, it retries the processing without extracting images.

    :param file_path: The path to the PDF file to be processed.
    :type file_path: str
    :param _: Placeholder parameter that is not utilized.
    :type _: Any
    :return: A list containing processed and split PDF content.
    :rtype: list
    """
    try:
        # First attempt with image extraction
        loader = PyPDFLoader(file_path, extract_images=True)
        return loader.load_and_split()
    except (IOError, ValueError) as e:
        LOGGER.error(
            f"Error processing images in {file_path}: {e}"
            f"\nRetrying without image extraction..."
        )
        # Retry without extracting images
        loader = PyPDFLoader(file_path, extract_images=False)
        return loader.load_and_split()


# Function to process XML files
# https://python.langchain.com/docs/integrations/document_loaders/xml
def process_xml_data(file_path, _):
    """
    Processes XML data from the specified file and loads its contents into a structured
    format using the UnstructuredXMLLoader.

    :param file_path: The path to the XML file that needs to be processed.
    :type file_path: str

    :param _: Placeholder parameter reserved for compatibility, unused within the
        current function implementation.
    :type _: Any

    :return: The result of the XML loading operation, typically representing the
        structured content parsed from the specified XML file.
    :rtype: Any
    """
    loader = UnstructuredXMLLoader(file_path)
    return loader.load()


# Function to process Text files
# https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.text.TextLoader.html
def process_text_data(file_path, _):
    """
    Processes text data from a specified file.

    This function utilizes the TextLoader class to load and process text
    data from the given file path. It ensures that the text data is properly
    retrieved and returned for further processing or usage.

    :param file_path: Path to the file from which text data should be loaded.
    :type file_path: str
    :param _: Unused parameter for potential future extensibility or compatibility.
    :type _: Any
    :return: The loaded text data from the specified file.
    :rtype: str
    """
    loader = TextLoader(file_path)
    return loader.load()


# Function to process unstructured files
# https://python.langchain.com/docs/integrations/document_loaders/unstructured_file
# In theory we could use just this one function for all file types, but we might want
# more control than that over certain types, so I left them split out and explicitly
# chose what processor to use. This function is also used as a fallback for unsupported
# file types.
def process_unstructured_data(file_path, _):
    """
    Processes unstructured data from a file by utilizing an UnstructuredFileLoader in
    single mode. The function takes a file path and an additional placeholder parameter,
    then loads and returns the data in an unstructured format.

    :param file_path: The path to the file containing unstructured data.
    :type file_path: str
    :param _: A placeholder parameter that is not actively used within
        the function.
    :return: The loaded unstructured data extracted from the file.
    :rtype: Any
    """
    loader = UnstructuredFileLoader(file_path, mode="single")
    return loader.load()


def process_text(
    text,
    metadata=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """
    Process in-memory text into chunked LangChain Document objects.

    Unlike the file-based ``process_*`` functions above, this accepts raw text
    strings directly — useful for scraped web content, API responses, or any
    text that doesn't originate from a file on disk.

    The text is split into overlapping chunks using
    ``RecursiveCharacterTextSplitter`` so that each chunk is a reasonable size
    for embedding.  Each resulting ``Document`` carries the provided metadata
    (if any) plus a ``chunk_index`` field.

    :param text: The raw text content to process.
    :type text: str
    :param metadata: Optional metadata dict attached to every Document.
        Common keys include ``source``, ``name``, ``url``.
    :type metadata: dict or None
    :param chunk_size: Maximum characters per chunk.  Defaults to 1000.
    :type chunk_size: int
    :param chunk_overlap: Overlap between consecutive chunks.  Defaults to 200.
    :type chunk_overlap: int
    :return: A list of Document objects, one per chunk.  Returns an empty list
        if the input text is empty or too short to chunk.
    :rtype: list[Document]
    """
    if not text or not text.strip():
        return []

    base_metadata = metadata or {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        doc_metadata = {**base_metadata, "chunk_index": i, "total_chunks": len(chunks)}
        documents.append(Document(page_content=chunk, metadata=doc_metadata))

    LOGGER.debug(
        "Split text into %d chunks (chunk_size=%d)", len(documents), chunk_size
    )
    return documents


def chunk_documents(
    documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """
    Split existing Document objects into smaller chunks.

    This can be applied after any of the ``process_*`` functions to further
    subdivide documents that are too large for effective embedding.  Each
    resulting chunk preserves the original document's metadata and adds a
    ``chunk_index`` field.

    :param documents: A list of LangChain Document objects to split.
    :type documents: list[Document]
    :param chunk_size: Maximum characters per chunk.  Defaults to 1000.
    :type chunk_size: int
    :param chunk_overlap: Overlap between consecutive chunks.  Defaults to 200.
    :type chunk_overlap: int
    :return: A flat list of chunked Document objects.
    :rtype: list[Document]
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked = splitter.split_documents(documents)

    LOGGER.debug(
        "Split %d documents into %d chunks (chunk_size=%d)",
        len(documents),
        len(chunked),
        chunk_size,
    )
    return chunked
