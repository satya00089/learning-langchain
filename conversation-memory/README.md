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
