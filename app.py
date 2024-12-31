import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
import logging

# Load environment variables from the .env file
load_dotenv()

# Access the variables using os.getenv()
groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize the LLM
langchain_llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Qdrant Client
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Initialize global vector database
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Define chat UI templates


# Button styles
button_style = """
<style>
    .small-button {
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
    }
    .small-button:hover {
        background-color: #0056b3;
    }

    /* Make the sidebar wider and main content narrower */
    .css-1v3fvcr {
        width: 30%;  /* Sidebar width */
    }

    .css-1d391kg {
        width: 70%;  /* Main content width */
    }
</style>
"""

# Function to process YouTube URL
def process_youtube_url(youtube_url):
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from {youtube_url}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks")

        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        collection_name = "youtube-video-index"
        qdrant_client.delete_collection(collection_name)
        logging.info(f"Deleted collection '{collection_name}' (if exists)")

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logging.info(f"Created collection '{collection_name}'")

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embeddings)
        vector_store.add_documents(split_docs)
        logging.info(f"Added {len(split_docs)} documents to collection '{collection_name}'")

        return vector_store
    except Exception as e:
        logging.error(f"Error processing YouTube URL: {str(e)}")
        raise

# Function to answer questions based on video content
def answer_question(question):
    if st.session_state.vector_db is None:
        return "Please process a YouTube video URL first."

    try:
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

        system_prompt = (
            "You are a video assistant tasked with answering questions based on the provided YouTube video context. "
            "Use the given context provided by the video author to provide accurate, concise answers in three sentences. "
            "If the context does not contain the answer, say you are not sure. "
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(langchain_llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        response = chain.invoke({"input": question})

        if "answer" not in response:
            raise ValueError("Response does not contain an 'answer' key.")

        st.session_state.chat_history.append((question, response['answer']))
        return response['answer']
    except Exception as e:
        error_message = f"Error: {str(e)}"
        st.session_state.chat_history.append((question, error_message))
        return error_message

# Main function
def main():
    st.set_page_config(page_title="YouTube Video Q&A", layout="wide")

    with st.sidebar:
        st.header("YouTube Video Q&A")
        youtube_url = st.text_input("Enter YouTube URL:")
        if st.button("Submit"):
            try:
                st.session_state.vector_db = process_youtube_url(youtube_url)
                default_question = "What is the video about?"
                summary = answer_question(default_question)
                st.session_state.summary = summary
                st.success("Video indexed successfully ✅! You can now ask questions about the video in the chatbot.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if "summary" in st.session_state:
        st.markdown(
            """
            <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h2 style="text-align: center; color: #007bff;">Video Summary</h2>
                <p style="font-size: 16px; line-height: 1.6; color: #555555; padding: 10px; background-color: #ffffff; border-radius: 10px;">
                    {summary}
                </p>
            </div>
            """.format(summary=st.session_state.summary),
            unsafe_allow_html=True,
        )

    if st.session_state.vector_db:
        st.header("Converse with your Video")

        # Initialize chat history if not already present
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Ask question input field at the bottom
        if prompt := st.chat_input("Ask a question about the video"):

            # Append user question to chat history (after it is submitted)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get the assistant's response
            answer = answer_question(prompt)

            # Append assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("user").write(prompt)
            # Display the assistant's response in chat
            st.chat_message("assistant").write(answer)

    # Add button styles
    st.markdown(button_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
