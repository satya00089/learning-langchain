"""
Goal: Multi-Document RAG — LLM Ecosystem Knowledge Base

Docs indexed:
  blogs/openai/     → OpenAI announcements (GPT-5, Codex, etc.)
  blogs/anthropic/  → Anthropic announcements (Claude 4.x)
  model-cards/      → GPT-4o, GPT-5, Claude Opus/Sonnet, Gemini, LLaMA 3
  papers/           → Attention, AutoDev, CoT, InstructGPT, LLaMA 3, Mixtral
  llm_benchmarks_2025_rag.txt → Benchmark comparisons

Skills demonstrated:
  metadata-enriched loading (provider / doc_type / source / page)
  folder-name → metadata auto-inference
  single unified FAISS vector store
  source-aware prompt with citations
  mixed-source retrieval (k=8)
"""

from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


DOCS_DIR = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / "llm_faiss_index"


def infer_metadata(file_path: Path) -> dict:
    """
    Derive provider / doc_type from directory path.

    Structure:
        docs/blogs/<provider>/file.pdf   → blog
        docs/model-cards/file.pdf        → model_card  (provider from filename)
        docs/papers/file.pdf             → paper
        docs/llm_benchmarks_2025_rag.txt → benchmark
    """
    parts = file_path.parts
    stem = file_path.stem.lower()

    # ---- doc_type + provider from folder ----
    if "blogs" in parts:
        doc_type = "blog"
        blogs_idx = list(parts).index("blogs")
        # folder immediately after "blogs/" is the provider name
        provider = (
            parts[blogs_idx + 1] if (blogs_idx + 1) < (len(parts) - 1) else "unknown"
        )

    elif "model-cards" in parts:
        doc_type = "model_card"
        # Infer provider from filename keywords
        if any(k in stem for k in ("gpt", "openai", "codex")):
            provider = "openai"
        elif any(k in stem for k in ("claude", "anthropic")):
            provider = "anthropic"
        elif any(k in stem for k in ("llama", "meta")):
            provider = "meta"
        elif any(k in stem for k in ("gemini", "google")):
            provider = "google"
        else:
            provider = "unknown"

    elif "papers" in parts:
        doc_type = "paper"
        provider = "research"

    else:
        doc_type = "benchmark"
        provider = "mixed"

    return {
        "provider": provider,
        "doc_type": doc_type,
        "source": file_path.name,
        "file_path": str(file_path),
    }


def load_all_documents(docs_dir: Path) -> list:
    """Walk the entire docs/ tree; load PDFs, TXT, and MD files."""
    all_docs = []

    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in (".pdf", ".txt", ".md"):
            continue

        metadata = infer_metadata(file_path)

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")

            docs = loader.load()

            for doc in docs:
                doc.metadata.update(metadata)
                # Prepend a short provider/type tag so embeddings distinguish
                # providers even when the topic is similar (e.g. "context window")
                tag = (
                    f"[{metadata['provider'].upper()} | "
                    f"{metadata['doc_type']} | "
                    f"{file_path.stem}]\n"
                )
                doc.page_content = tag + doc.page_content

            all_docs.extend(docs)
            print(
                f"  ✓ {file_path.name:<50s} "
                f"[{metadata['provider']:<12s} | {metadata['doc_type']}] "
                f"({len(docs)} page(s))"
            )

        except Exception as exc:
            print(f"  ✗ {file_path.name}: {exc}")

    return all_docs


def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Load or build a FAISS vector store from the documents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if INDEX_DIR.exists() and not force_rebuild:
        print(f"\nLoading existing index from  {INDEX_DIR} …")
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    print("\nBuilding vector index from scratch …\n")

    raw_docs = load_all_documents(DOCS_DIR)
    print(f"\nTotal pages loaded : {len(raw_docs)}")

    # Larger chunks preserve LLM spec context (avoid breaking mid-sentence)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Total chunks        : {len(chunks)}")

    print("\nEmbedding & indexing (this may take a minute) …")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Index saved to      : {INDEX_DIR}\n")

    return vectorstore


def format_with_citations(docs) -> str:
    """
    Wrap each retrieved chunk in a metadata header so the LLM
    can naturally cite (Provider | doc_type | filename | page).
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        provider = meta.get("provider", "unknown").upper()
        doc_type = meta.get("doc_type", "unknown")
        source = meta.get("source", "unknown")
        page = meta.get("page", "")

        header = f"[{i}] {provider} | {doc_type} | {source}"
        if page != "":
            header += f" | p.{int(page) + 1}"  # PyPDF pages are 0-indexed

        parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert AI/LLM assistant with deep knowledge of language models, \
research papers, and provider announcements.

Rules:
- Use ONLY the context below to answer.
- When multiple providers are relevant, compare them side-by-side.
- Cite every fact using the format: (Provider | doc_type | filename | page).
- If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer (with citations):"""
)


def build_rag_chain(vectorstore: FAISS):
    """Build a RAG chain using the provided vectorstore."""
    # MMR (Max Marginal Relevance) balances relevance AND diversity.
    # fetch_k=20 → candidate pool; k=8 → final diverse selection.
    # This prevents same-provider dominance when topics are similar.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6},
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    chain = (
        {
            "context": retriever | format_with_citations,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


SAMPLE_QUERIES = [
    "Compare Claude 4 and GPT-5 context window",
    "Which models are multimodal?",
    "What is Mixture-of-Experts and which models use it?",
    "Which model has the longest context window?",
    "Compare LLaMA 3 and GPT architecture",
]


def chat(chain, vectorstore: FAISS):
    print("\n" + "=" * 65)
    print("  Multi-Document LLM Ecosystem RAG")
    print("  Sources: OpenAI · Anthropic · Meta · Google · Research")
    print("=" * 65)
    print("\nSample questions:")
    for q in SAMPLE_QUERIES:
        print(f"  • {q}")
    print("\nCommands: 'exit' to quit | 'rebuild' to rebuild the index\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if question.lower() == "rebuild":
            print("Rebuilding index …")
            vectorstore = build_vectorstore(force_rebuild=True)
            chain = build_rag_chain(vectorstore)
            print("Index rebuilt. Ready.\n")
            continue

        print("\n[Searching across LLM ecosystem docs …]\n")

        # Debug: show exactly which sources were retrieved — verify cross-provider diversity
        debug_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6},
        )
        retrieved_docs = debug_retriever.invoke(question)
        print("Sources retrieved:")
        for d in retrieved_docs:
            m = d.metadata
            print(f"  {m.get('provider','?'):<12s} | {m.get('doc_type','?'):<12s} | {m.get('source','?')}")
        print()

        answer = chain.invoke(question)
        print(f"Assistant:\n{answer}\n")
        print("-" * 65 + "\n")


def main():
    load_dotenv()
    vectorstore = build_vectorstore()
    chain = build_rag_chain(vectorstore)
    chat(chain, vectorstore)


if __name__ == "__main__":
    main()
