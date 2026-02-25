# role-based-explainer — Quick start

A tiny example that explains a topic in a chosen role (teacher, pirate, lawyer, kid).

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

When prompted, enter a topic and a role; the script prints an explanation in that role's style.

Notes
- Uses `ChatOpenAI` and `ChatPromptTemplate` from LangChain.
- Example script: `main.py` in this folder.
