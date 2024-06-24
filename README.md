# Avdeep's AI Assistant 🤖

This project is an AI-powered chatbot designed to answer questions about my professional background. It leverages the power of Large Language Models (LLMs) and natural language processing (NLP) to provide insightful and engaging responses.

**Try it out here:** [https://aibot.asoberoi.com/](https://aibot.asoberoi.com/)

## Key Features

* **Personalized Responses:** Trained on my resume, projects, and skills to provide accurate and relevant information about my career journey.
* **Natural Language Understanding:** Understands complex questions and phrases related to my experience and expertise.
* **Engaging and Approachable:** Delivers information in a conversational and personable tone, reflecting my communication style.
* **Constantly Evolving:** Continuously learning and improving its knowledge base to provide even more comprehensive answers over time.

## Technologies Used

* **LangChain:** A framework for developing applications powered by language models.
* **NVIDIA NeMo Framework:** Used for training and deploying large language models.
* **Llama 3 70B (Instruct Model):** A powerful and efficient large language model from Meta, fine-tuned for instruction following.
* **FAISS (Facebook AI Similarity Search):** For efficient similarity search and retrieval of relevant information from my knowledge base.
* **Streamlit:** A Python framework for building interactive web applications.
* **Nginx:** Used as a reverse proxy for deployment.

## How it Works

1. **Data Preparation:** My resume and other relevant information were processed and structured to create a comprehensive knowledge base.
2. **Embeddings:** The knowledge base was transformed into numerical representations (embeddings) using NVIDIA's embedding model.
3. **Vector Database:** Embeddings are stored in a vector database (FAISS) for efficient similarity search.
4. **User Query:** When you ask a question, it's converted into an embedding.
5. **Similarity Search:** The vector database identifies the most similar embeddings from the knowledge base, representing the most relevant information.
6. **Response Generation:** The LLM (Llama 3) uses the retrieved information and the user's question to generate a natural and informative response.

## Future Enhancements

* **Multilingual Support:** Adding support for multiple languages to make the chatbot accessible to a wider audience.
* **Voice Integration:** Integrating voice recognition and synthesis for a more seamless and interactive user experience.
* **Contextual Memory:** Enabling the chatbot to remember previous interactions for more personalized and coherent conversations.

## Contributing

This project is just the beginning! I welcome contributions, feedback, and suggestions from the community to enhance its capabilities and explore the full potential of AI-powered personal assistants.

## Let's Connect!

* **LinkedIn:** [www.linkedin.com/in/avdeep-singh]
* **Email:** [avdeep2001@gmail.com] 
