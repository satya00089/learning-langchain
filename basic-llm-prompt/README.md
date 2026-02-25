# basic-llm-prompt — Quick start

A minimal example showing LangChain prompt + a Chat model.

Usage
- Create and activate a Python virtual environment (recommended).
- Install dependencies:

```bash
pip install -r requirments.txt
```

- Create an `.env` from the example and add your API key(s):

```bash
copy .env.example .env
# then edit .env to set OPENAI_API_KEY
```

- Run the script:

```bash
python main.py
```

The script will prompt for a topic and print a 5-point explanation.

Notes
- The project uses `ChatOpenAI` and `ChatPromptTemplate` (LangChain).
- The repository file for this example is `main.py` in this folder.
