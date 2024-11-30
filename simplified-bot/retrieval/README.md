# Hybrid Search Retrieval Module

This module demonstrates a hybrid search approach combining semantic search with BM25 retrieval for enhanced retrieval performance. It leverages the Google Vertex AI embedding model, Chroma vector database, and LangChain retrieval functionalities. This simplified example showcases a core component of a Retrieval Augmented Generation (RAG) system.

## Functionality

The `retrieval.py` script performs the following steps:

1. **Initialization:** Initializes the Google Vertex AI embedding model specified by `EMBEDDING_MODEL_NAME` and a Chroma vector database.  Requires the same environment variables as `chunker.py`: `GOOGLE_API_KEY` and `GOOGLE_PROJECT`.
2. **Document Loading and Chunking:** Loads text documents from the specified directory and chunks them using `CharacterTextSplitter`.  The chunk size and overlap are configurable.
3. **BM25 Retrieval:** Creates a BM25 retriever using the *chunked* documents.  BM25 is a powerful keyword-based retrieval method.
4. **Semantic Retrieval:** Creates a semantic retriever using the Chroma vector database and the specified embedding model.
5. **Ensemble Retrieval:** Combines the BM25 and semantic retrievers into an ensemble retriever, weighting their contributions (default: 0.6 for semantic, 0.4 for BM25).
6. **Query Processing:**  Retrieves the top-k most relevant documents for a given query using the ensemble retriever.
7. **Result Display:** Prints the content of the retrieved documents.  You can easily modify this to return the documents or process them further.
8. **Error Handling:** Includes error handling to manage potential issues during retrieval.


## Usage

1. **Prerequisites:**
   - Ensure you have the required libraries installed (see Dependencies).
   - Set the required environment variables:
      ```bash
      export GOOGLE_API_KEY="YOUR_API_KEY"
      export GOOGLE_PROJECT="YOUR_PROJECT_ID"
      ```
2. **Prepare Data:**  Place your text files in the `example_data` directory (or modify the `data_directory` argument). Ensure the Chroma vector database (`example_data`) has been populated using the `chunker.py` script.
3. **Run the script:** `python retrieval.py`

## Customization

* **Embedding Model:** Change the `EMBEDDING_MODEL_NAME` constant in `chunker.py` to use a different Google Vertex AI embedding model.
* **Chunking:** Modify the `chunk_size` and `chunk_overlap` parameters in `retrieval.py` to adjust the chunking strategy.  Consider implementing more advanced chunking techniques if necessary.
* **Retrieval Parameters:** Adjust the `top_k` parameter to control the number of retrieved documents.  Experiment with the weights in the `EnsembleRetriever` to fine-tune the balance between semantic and BM25 retrieval.
* **Data Directory:**  Change the `data_directory` argument in the `hybrid_search` function to load documents from a different location.
* **Vector Database:**  While this example uses Chroma, you can adapt the code to use other vector databases.  Modify the `initialize_vectorstore` function accordingly.


## Example

The `if __name__ == "__main__":` block provides a basic example of how to use the `hybrid_search` function.  Modify the `query` variable to test different search queries.

## Dependencies

- `google-cloud-aiplatform`: For interacting with Google Vertex AI.
- `langchain`:  Framework for building language model applications.
- `chromadb`:  Vector database for storing and querying embeddings.
- `tqdm`: For displaying progress bars (used in `chunker.py`).


## Contributing

Contributions and improvements are welcome!  Feel free to open issues or submit pull requests.

## License

[MIT License]