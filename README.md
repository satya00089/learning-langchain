# learning-langchain — Mini-projects

A compact collection of progressive LangChain mini-projects to learn prompt, chaining, memory, and agents.

Quick setup
- Create and activate a Python virtual environment.
- Install dependencies (each example has its own `requirments.txt`):

```bash
pip install -r requirments.txt
```

- Add API keys: copy the provided `.env.example` in an example folder and set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

Progressive mini-projects (test knowledge by doing)
- **Basic prompt**: [basic-llm-prompt](basic-llm-prompt) — simple LangChain prompt template + Chat model. Run `python main.py` inside the folder.
- **Role-based explainer**: [role-based-explainer](role-based-explainer) — generate explanations in different voices (teacher, pirate, lawyer, kid). Run `python main.py`.
- **Simple 2-step chain**: [simple-chain](simple-chain) — build a 2-step chain (explain simply → give real-world example). Run `python main.py`.
- **Conversation memory**: [conversation-memory](conversation-memory) — multiple examples showing in-memory history, `RunnableWithMessageHistory`, and a checkpointer. Try `python main.py`, `python langchain_main.py`, or `python checkpointer_main.py`.
- **Mini RAG**: [mini-RAG](mini-RAG) — Document Q&A (load a PDF or text, build a FAISS index, ask questions). Run `python main.py` inside the folder.
- **Basic agent**: [basic-agent](basic-agent) — starter agent examples and small demos. See folder for scripts and run instructions.
- **Foundations & notebooks**: [lca-lc-foundations](lca-lc-foundations) — longer-form notebooks and exercises for deeper learning. Open notebooks in `notebooks/`.

Notes
- Examples are intentionally small and runnable; they keep secrets in memory by default. For persistence, replace in-memory stores with a DB-backed store.
- Filenames use `requirments.txt` (project typo) — install from each example's file when running that example.
- If you hit message-format errors (e.g., TypeError: 'AIMessage' object is not subscriptable), extract replies defensively by checking whether messages are dicts or message objects (see conversation-memory README for a snippet).

Want me to run any example now to verify it starts? Use the folder name and I can run it here.
