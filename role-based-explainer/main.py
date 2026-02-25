"""
Goal:
Create a “Role-based explainer”:

Input:
    topic,
    role (teacher, pirate, lawyer, kid)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

user_topic = input("Enter topic: ")
user_role = input("Enter role: ")

prompt = ChatPromptTemplate.from_template(
    """
    Explain {topic}.
    Speak in the style of a {role}.
    Adjust tone, vocabulary, and examples to match the role.
    """
)

model = ChatOpenAI(model="gpt-4o")

chain = prompt | model

resp = chain.invoke({"topic": user_topic, "role": user_role})

print(resp.content)
