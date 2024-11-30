# LLM Call Module (call.py)

This module demonstrates how to make API calls to the Google Gemini language model using LangChain.  It uses a chat prompt template to provide context and history to the LLM.

## Functionality

The `call.py` script performs the following:

1. **LLM Initialization:** Initializes the specified Gemini model (`MODEL_NAME`). Requires `GOOGLE_API_KEY` and `GOOGLE_PROJECT` environment variables.
2. **Prompt Construction:** Creates a chat prompt using a template, incorporating the current time, retrieved context, session history, and user query.
3. **LLM Call:** Makes an API call to the Gemini model with the constructed prompt.
4. **Response Handling:** Returns the LLM's response.  Error handling is included to manage potential API issues.

## Usage

1. **Prerequisites:** Install required libraries and set environment variables (see Dependencies).
2. **Prepare Inputs:** Provide the `retrieved_text`, `user_query`, and `session_history` (if any) to the `llm_call` function.
3. **Run the script:** `python call.py`

## Customization

* **Model:** Change the `MODEL_NAME` constant to use a different Gemini model.
* **Temperature:** Adjust the `TEMPERATURE` to control the randomness of the LLM's output.
* **Max Output Tokens:**  Modify `MAX_OUTPUT_TOKENS` to limit the length of the LLM's response.
* **Prompt Template:** Customize the `SYSTEM_PROMPT` to modify the instructions given to the LLM.

## Dependencies

- `google-cloud-aiplatform`: For interacting with Google Vertex AI.
- `langchain`: Framework for LLM interactions.

## License

[MIT License]