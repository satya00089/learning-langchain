# simple-chain — Quick start

A small two-step LangChain example: simple explanation then a real-world example.

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

- Run the script:

```bash
python main.py
```

When prompted, enter a topic; the script prints a short explanation and a related real-world example.

Notes
- Uses `ChatOpenAI` and `ChatPromptTemplate` from LangChain.
- Example script: `main.py` in this folder.
