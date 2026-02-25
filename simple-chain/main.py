"""
Goal:
Build a 2-step chain:

Input: topic
Step1: Explain simply
Step2: Give real-world example
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# user input
topic = input("Enter topic: ").strip()

# step 1 — simple explanation
explanation_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

# step 2 — real-world example using explanation
example_prompt = ChatPromptTemplate.from_template(
    """Using this explanation:
{explanation}

Give a real-world example of {topic}."""
)

# model
model = ChatOpenAI(model="gpt-4o")

# chains
explanation_chain = explanation_prompt | model
example_chain = example_prompt | model

# step 1 execution
explanation_resp = explanation_chain.invoke({
    "topic": topic
})

explanation = explanation_resp.content.strip()

# optional safety check
if not explanation:
    raise ValueError("Explanation generation failed")

# step 2 execution
example_resp = example_chain.invoke({
    "topic": topic,
    "explanation": explanation
})

example = example_resp.content.strip()

# output
print("\nExplanation:\n")
print(explanation)

print("\nReal-world Example:\n")
print(example)
