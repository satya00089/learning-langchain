"""
Goal: Understand LangChain model & prompt abstraction.

Task:
    Build a Python script that:
        Takes a topic from user
        Generates a 5-point explanation
        Uses LangChain prompt template
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

topic = input("Enter topic: ")

prompt = ChatPromptTemplate.from_template(
    "Generate a 5-point explanation on {topic}."
)

model = ChatOpenAI(model="gpt-4o")

chain = prompt | model

response = chain.invoke({"topic": topic})

print(response.content)
