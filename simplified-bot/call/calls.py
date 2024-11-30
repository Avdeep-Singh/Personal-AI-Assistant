from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatVertexAI
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants and Configuration
MODEL_NAME = "gemini-1.5-pro"  # Or "gemini-1.5-flash"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1500
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_PROJECT = os.environ.get("GOOGLE_PROJECT")
LOCATION = "us-central1"  # Replace with your model's region

SYSTEM_PROMPT = """
You are an advanced, highly knowledgeable AI assistant.

Variables:

* Current Date and Time: {time}
* Context: {context}
* History: {session_history}
* User Question: {question}
"""


def initialize_llm():
    """Initializes and returns the ChatVertexAI language model."""
    if not GOOGLE_API_KEY or not GOOGLE_PROJECT:
        raise ValueError("GOOGLE_API_KEY and GOOGLE_PROJECT environment variables must be set.")

    return ChatVertexAI(
        google_api_key=GOOGLE_API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        project=GOOGLE_PROJECT,
        location=LOCATION,
    )


def llm_call(retrieved_text, user_query, session_history, llm):
    """Makes a call to the LLM with the provided context and history."""
    try:
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        chain = (prompt | llm)  # Simplified chain

        response = chain.invoke(
            {
                "time": time.asctime(),
                "context": retrieved_text,
                "session_history": session_history,
                "question": user_query,
            }
        )
        return response

    except Exception as e:
        logging.error(f"Error during LLM call: {e}")
        return "An error occurred during processing."  # Return a message in case of error



if __name__ == "__main__":
    try:
        llm = initialize_llm()

        retrieved_text = "Example retrieved context"  # Replace with actual retrieved text
        user_query = "Example user query"  # Replace with user input
        session_history = []  # Or load previous history

        response = llm_call(retrieved_text, user_query, session_history, llm)
        print(response)

    except Exception as e:
        logging.error(f"An error occurred: {e}")