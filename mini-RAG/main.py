"""
Goal: Document Q&A (Mini RAG)

Task: Load a PDF or text file and allow Q&A.

Example: Upload LangChain docs → ask questions.

Skills:
    document loaders
    text splitting
    embeddings
    vector store
    retriever
"""

import os

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
loader = PyPDFLoader("INSURANCE_AGENTS_LIFE.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(INDEX_PATH):
    vector_store = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)

model = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful insurance assistant. "
            "Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

chain = prompt | model

store = {}


def get_session_history(session_id: str):
    """session_id can be user_id or any identifier to group messages into a conversation"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("Chatbot with memory. Type 'exit' to stop.\n")

USER_SESSION_ID = "default"

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    retrieved_docs = vector_store.similarity_search(user_input, k=3)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = memory_chain.invoke(
        {
            "input": user_input,
            "context": context_text,
        },
        config={"configurable": {"session_id": USER_SESSION_ID}},
    )

    print("Bot:", response.content)
