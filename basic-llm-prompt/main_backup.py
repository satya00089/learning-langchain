"""
Goal: Understand LangChain model & prompt abstraction
Task:
    Build a Python script that:
        Takes a topic from user
        Generates a 5-point explanation
        Uses LangChain prompt template
"""

from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

SYSTEM_PROMPT = """
You are a help full chart bot. helps with user queries.
Generates a 5-point explanation on user topic.
"""

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt=SYSTEM_PROMPT
)


resp = agent.invoke(
    {"messages":  "What is langchain"}
)

print(resp)
