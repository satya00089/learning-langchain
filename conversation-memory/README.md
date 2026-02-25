# conversation-memory — Quick start

A simple chatbot example that keeps conversation history in memory.

Usage
- Create and activate a Python virtual environment (recommended).
- Install dependencies:

```bash
pip install -r requirments.txt
```

- Create an `.env` from the example and add your API key(s):

```bash
copy .env.example .env
# then edit .env to set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

There are two example runners in this folder:

- `main.py` — a minimal loop that appends `HumanMessage` and `AIMessage` to an in-memory list and invokes the prompt+model chain. Run with:

```bash
python main.py
```

- `langchain_main.py` — demonstrates `RunnableWithMessageHistory` + `InMemoryChatMessageHistory`. It stores per-session history and accepts a configurable `session_id` (the example uses `default`). To run:

```bash
python langchain_main.py
```

Change `session_id` in the script (or extend the script to accept it from user input) to keep separate conversation threads in memory.

Notes
- Uses `ChatOpenAI`, `ChatPromptTemplate` and the LangChain message/history helpers.
- `langchain_main.py` shows how to wrap a runnable with `RunnableWithMessageHistory` so history is automatically saved/loaded for a session.
- Both examples keep history only in-process (memory). For persistence across runs, replace `InMemoryChatMessageHistory` with a DB-backed store.

Additional runner: `checkpointer_main.py`

- `checkpointer_main.py` — shows using a LangGraph checkpointer (`InMemorySaver`) with `create_agent()` to keep conversation threads by `thread_id`. Run with:

```bash
python checkpointer_main.py
```

- `thread_id` behaves like a session key; change it to separate conversation threads.

Note about message shapes and a common TypeError
- Depending on the agent/runtime, `agent.invoke()` may return messages as plain dicts (e.g. `{"role":"assistant","content":"..."}`) or as message objects (e.g. `AIMessage` instances). Attempting to index an object like a dict can raise errors such as TypeError: 'AIMessage' object is not subscriptable.

- If you see that error, extract the assistant reply robustly, for example:

```py
# result may be dict-like or contain message objects
msgs = result.get("messages") if isinstance(result, dict) else getattr(result, "messages", None)
last = msgs[-1]
reply = last["content"] if isinstance(last, dict) else getattr(last, "content", str(last))
```

This README previously included `main.py` and `langchain_main.py` usage. The new `checkpointer_main.py` example demonstrates a checkpointer-backed agent; inspect the script to adapt `thread_id` or persist memory elsewhere.
