# Chunking and Embedding Module

This module provides functionality for chunking and embedding text documents (currently TXT, with PDF support planned) using the Google Vertex AI embedding model and storing the embeddings in a Chroma vector database.  This is a simplified demonstration of a key component in a Retrieval Augmented Generation (RAG) pipeline.

## Functionality

The `chunker.py` script performs the following steps:

1. **Initialization:** Initializes the Google Vertex AI embedding model specified by `EMBEDDING_MODEL_NAME` (default: `textembedding-gecko@003`).  Requires `GOOGLE_API_KEY` and `GOOGLE_PROJECT` environment variables to be set.
2. **Vector Store Initialization:** Initializes a Chroma vector database in the directory specified by `PERSIST_DIRECTORY` (default: `sample_database`).
3. **Document Processing:** Iterates through all files in the input directory.
4. **Chunking (Currently Basic):**  For TXT files, the entire file content is treated as a single chunk. More sophisticated chunking methods can be implemented.
5. **Embedding:**  Generates embeddings for each chunk using the specified Google Vertex AI embedding model.
6. **Storage:** Stores the embeddings and optionally metadata in the Chroma vector database.
7. **Error Handling:** Includes error handling and logging to manage issues during file processing.

## Usage

1. **Prerequisites:**
   - Install required libraries: `pip install google-cloud-aiplatform langchain chromadb tqdm`
   - Set environment variables:
     ```bash
     export GOOGLE_API_KEY="YOUR_API_KEY"
     export GOOGLE_PROJECT="YOUR_PROJECT_ID"
     ```
2. **Prepare Data:** Place your TXT files in the `chunks` directory (or modify the `source_path` variable in the script).
3. **Run the script:**  `python chunker.py`

## Customization

* **Embedding Model:** Change the `EMBEDDING_MODEL_NAME` constant to use a different Google Vertex AI embedding model.
* **Chunk Size:**  The current implementation treats each file as a single chunk.  Implement more advanced chunking strategies (e.g., recursive text splitting, sentence-based chunking) within the `embed_documents` function.
* **Metadata:** Add metadata to the `vector_store.add_texts()` call to store additional information about each chunk.
* **Vector Database:**  While Chroma is used in this example, you can easily adapt the code to use other vector databases by modifying the `initialize_vectorstore` function.
* **PDF Handling:**  Add support for PDF files by incorporating a PDF processing library (like `PyPDF2`) and adding the necessary logic to the `embed_documents` function.  Example:
    ```python
    import PyPDF2

    # ... inside embed_documents function ...
    elif filename.endswith(".pdf"):
        with open(filepath, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text = page.extract_text()
                # ... (Chunking and embedding logic for the extracted text)
    ```



## Example Data

The `example_data` directory (create if needed) can contain sample TXT and PDF files to demonstrate the functionality.

## Example Output (Optional)

The `example_output` directory (create if needed) can contain a small sample or visualization of the generated vector database.


## Dependencies

- `google-cloud-aiplatform`: For interacting with Google Vertex AI.
- `langchain`:  Framework for building language model applications.
- `chromadb`:  Vector database for storing and querying embeddings.
- `tqdm`:  For displaying progress bars.
- `PyPDF2` (if adding PDF support): For processing PDF files.


## Contributing

Contributions and improvements are welcome!  Feel free to open issues or submit pull requests.
