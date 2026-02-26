import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

INDEX_PATH = "insurance_faiss_index"


def build_vector_store_from_file(pdf_path, embeddings):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)
    return vector_store


def get_session_history(session_store, session_id: str):
    """Helper to get or create chat history for a session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


st.set_page_config(page_title="Mini RAG — Document Q&A")
st.title("Mini RAG — Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF to build the index", type=["pdf"])

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "store" not in st.session_state:
    st.session_state["store"] = {}
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = OpenAIEmbeddings(model="text-embedding-3-small")
if "model" not in st.session_state:
    st.session_state["model"] = ChatOpenAI(model="gpt-4o")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Building vector store from uploaded PDF — this may take a moment.")
    st.session_state["vector_store"] = build_vector_store_from_file(
        tmp_path, st.session_state["embeddings"]
    )
    os.unlink(tmp_path)
    st.success("Index built and saved to disk.")

else:
    # Try loading existing index if present
    if st.session_state["vector_store"] is None and os.path.exists(INDEX_PATH):
        try:
            st.session_state["vector_store"] = FAISS.load_local(
                INDEX_PATH,
                st.session_state["embeddings"],
                allow_dangerous_deserialization=True,
            )
            st.info("Loaded existing index from disk.")
        except Exception:
            st.warning(
                "Found index folder but failed to load. Upload a PDF to rebuild."
            )

st.markdown("---")

query = st.text_input("Ask a question about the document")
user_session_id = st.text_input("Session ID (optional)", value="default")

if st.button("Ask") and query:
    if st.session_state["vector_store"] is None:
        st.error(
            "No vector store available. Upload a PDF first or ensure index exists."
        )
    else:
        retrieved_docs = st.session_state["vector_store"].similarity_search(query, k=3)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful insurance assistant. Use ONLY the provided context to answer the question. If the answer is not in the context, say you don't know.\n\nContext:\n{context}",
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt | st.session_state["model"]

        memory_chain = RunnableWithMessageHistory(
            chain,
            lambda sid: get_session_history(st.session_state["store"], sid),
            input_messages_key="input",
            history_messages_key="history",
        )

        with st.spinner("Generating answer..."):
            response = memory_chain.invoke(
                {"input": query, "context": context_text},
                config={"configurable": {"session_id": user_session_id}},
            )

        # Defensive extraction of text
        try:
            answer_text = response.content
        except Exception:
            # fallback if response is a dict or other structure
            answer_text = str(response)

        st.subheader("Answer")
        st.write(answer_text)

        st.subheader("Retrieved Context")
        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Doc {i}**")
            st.write(doc.page_content[:1000])
