"""
Goal: Conversation Memory (checkpointer memory)

Task:
Build a chatbot that remembers context:

User:

What is vector database?
Give example
Why useful?

Bot must remember topic = vector DB.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ===== Memory (LangGraph checkpointer) =====
memory = InMemorySaver()

# ===== Agent =====
agent = create_agent(
    model="gpt-4o",
    tools=[],  # no tools needed for simple chatbot
    checkpointer=memory,
    system_prompt=(
        "You are a helpful assistant that explains technical topics clearly. "
        "Remember conversation context when answering follow-up questions."
    ),
)

print("Agent chatbot with memory. Type 'exit' to stop.\n")

thread_id = "default"  # conversation session

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # agent returns full state; last message is assistant reply
    reply = result["messages"][-1].content.strip()

    print("Bot:", reply)
