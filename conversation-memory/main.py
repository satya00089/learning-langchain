"""
Goal: Conversation Memory

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
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

history = []

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that explains technical topics clearly."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

chain = prompt | model

print("Chatbot with memory. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    # invoke with history
    response = chain.invoke({"input": user_input, "history": history})

    print("Bot:", response.content)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))
