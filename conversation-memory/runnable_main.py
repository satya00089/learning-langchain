"""
Goal: Conversation Memory (RunnableWithMessageHistory)

Task:
Build a chatbot that remembers context:

User:

What is vector database?
Give example
Why useful?

Bot must remember topic = vector DB.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that explains technical topics clearly. "
            "Remember conversation context when answering follow-up questions.",
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

    response = memory_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": USER_SESSION_ID}},
    )

    print("Bot:", response.content)
