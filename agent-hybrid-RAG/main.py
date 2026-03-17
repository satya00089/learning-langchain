"""
RAG + Agent Hybrid
=================
A router agent that inspects each query and decides:
  1. search_docs  → FAISS RAG over local Azure docs
  2. calculator   → safe math eval
  3. (direct)     → general LLM answer

Example queries
  • What is the max timeout of an Azure Function App on Consumption plan?
  • What is 2 to the power of 20?
  • What is a neural network?
"""

import math
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent


load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "azure_faiss_index"

# ── Embeddings & Vector Store ─────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def _load_docs():
    """Walk docs/ and load all .txt / .md / .pdf files."""
    documents = []
    for path in DOCS_DIR.rglob("*"):
        if path.suffix == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        elif path.suffix in (".txt", ".md"):
            documents.extend(TextLoader(str(path), encoding="utf-8").load())
    return documents


def _build_index() -> FAISS:
    print("Building FAISS index from docs/ …")
    docs = _load_docs()
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {DOCS_DIR}. "
            "Add .txt / .md / .pdf files before running."
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    print(f"Index saved → {INDEX_DIR}  ({len(chunks)} chunks)")
    return store


if INDEX_DIR.exists():
    vector_store = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
else:
    vector_store = _build_index()


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def search_docs(query: str) -> str:
    """
    Search the uploaded Azure documentation to answer questions about
    Azure services, configurations, limits, and features.
    Use this tool whenever the question is about Azure.
    """
    results = vector_store.similarity_search(query, k=4)
    if not results:
        return "No relevant information found in the documentation."
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'doc')}]\n{doc.page_content}"
        for doc in results
    )


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports standard arithmetic and all functions from Python's math module.
    Examples: '2 ** 20', 'sqrt(144)', '(3 + 5) * 12', 'log(1000, 10)'
    Do NOT include 'math.' prefix — just the function name, e.g. sqrt(9).
    """
    allowed_names = {
        name: getattr(math, name) for name in dir(math) if not name.startswith("_")
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


# ── System prompt (router instructions) ───────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant acting as a router agent.
You have access to multiple specialist tools:

1. search_docs  — use for ANY question about Azure services, Azure Functions,
   Azure configurations, limits, pricing, or anything covered in the
   uploaded documentation.

2. calculator   — use for ANY arithmetic, algebraic, or mathematical
   computation. Pass a clean Python math expression as the argument.

3. web_search   — use when the answer is not available in the uploaded
   documentation and you need to consult the public web. Prefer
   `search_docs` for Azure-specific questions.

For general knowledge questions that are NOT about Azure docs and NOT
mathematical, answer directly from your own knowledge.

Always cite the source file when you use search_docs. When using web_search,
cite the top web results (title + URL) you relied on.
"""


@tool
def web_search(query: str) -> str:
    """
    Search the public web and return top results (snippet + title + URL).
    Use this when the answer is not found in the uploaded Azure documentation.
    """
    search = DuckDuckGoSearchResults()
    return search.invoke(query)


# ── Build the ReAct router agent ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_docs, calculator, web_search],
)


# ── Interactive REPL ──────────────────────────────────────────────────────────
def run():
    print("\n" + "=" * 60)
    print(" RAG + Agent Hybrid  |  type 'exit' to quit")
    print("=" * 60)
    print("Try:")
    print("  • What is the max timeout of an Azure Function App?")
    print("  • How many instances can the Consumption plan scale to?")
    print("  • What is 2 to the power of 20?")
    print("  • What is a transformer model?")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() == "exit":
            print("Bye!")
            break

        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

        # Agent.invoke can return different shapes depending on how the
        # agent was created/configured. Handle common cases gracefully.
        if isinstance(response, dict) and "structured_response" in response:
            answer = response["structured_response"]
        elif isinstance(response, dict) and "messages" in response:
            # messages is typically a list-like with objects that have `.content`
            try:
                answer = response["messages"][-1].content
            except Exception:
                answer = str(response["messages"])
        else:
            answer = str(response)

        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    run()
