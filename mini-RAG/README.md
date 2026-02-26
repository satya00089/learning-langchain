# mini-RAG — Document Q&A (Mini RAG)

Goal: Load a PDF or text file and enable Q&A over the document using embeddings and a FAISS vector store.

Quick start
- Create and activate a Python virtual environment.
- Install dependencies:

```bash
pip install -r requirments.txt
```

- Copy your PDF (for example `INSURANCE_AGENTS_LIFE.pdf`) into this folder.
- Set your OpenAI API key in the environment (for example `OPENAI_API_KEY`) or in a `.env` file.
- Run the example:

```bash
python main.py
```

Notes
- The script builds a FAISS index and saves it to the `insurance_faiss_index` folder; if that folder exists the index will be reused.
- The project uses the `OpenAIEmbeddings` and a chat model. Adjust model names or embeddings as needed.
- Filename typo: this repo uses `requirments.txt` for dependencies (match the file name when installing).

If you want, I can also add a minimal example PDF or a short test driver.
