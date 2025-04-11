# 📚 Academic Paper RAG System

A RAG-based academic paper QA system built with LangChain, Qwen (Tongyi), FAISS, and HuggingFace. This tool automatically converts PDF papers into structured Markdown using Mineru, and enables natural language Q&A over both text and table content via Retrieval-Augmented Generation.

---

## 🧭 Project Workflow Overview

1. 🧾 **Prepare the academic PDF file**
2. 🔍 **Use [Mineru](https://github.com/opendatalab/MinerU)** to convert PDF into Markdown:
   ```bash
   magic-pdf -p {some_pdf} -o {some_output_dir} -m auto
   ```
   ✔️ Output will be saved as: `some_output_dir/auto/some_pdf.md`
3. 🧹 **Run `convert_tables.py`** to convert HTML tables in the `.md` file into plain Markdown tables:
   ```bash
   python convert_tables.py
   ```
   ✔️ Output will be saved as: `data/some_pdf_cleaned.md`

4. 🤖 **Run `main.py` to start the QA system**
   ```bash
   python main.py
   ```

---

## 📂 Project Structure

```
.
├── data/
│   ├── some_pdf.md                # Original Markdown output from Mineru (with HTML tables)
│   └── some_pdf_cleaned.md        # Cleaned Markdown (with plain text tables)
├── convert_tables.py              # HTML → Markdown table conversion script
├── faiss_index/                   # Folder to store FAISS vector indexes
├── main.py                        # Main RAG QA pipeline
├── .env                           # Environment variables (for DashScope API Key)
└── README.md                      # Project documentation (this file)
```

---

## ⚙️ Installation

```bash
pip install langchain dashscope faiss-cpu python-dotenv sentence-transformers markdownify beautifulsoup4
```

---

## 🔑 Configure DashScope API Key

Create a `.env` file in the project root with the following content:

```ini
ALIYUN_API_KEY=your_dashscope_api_key
```

You can obtain the key from the [DashScope Console](https://dashscope.console.aliyun.com/).

---

## 📜 Example Query

After running `main.py`, you can interact in natural language:

```python
query = "What is the core method proposed in this paper?"
response = qa_chain.invoke(query)
print(response)
```

The system retrieves the most relevant Markdown chunks and generates a natural language response using Qwen.

---

## 🧠 Tech Stack

| Module                      | Description                                      |
| -------------------------- | ------------------------------------------------ |
| **LangChain**              | Chains for vector retrieval and QA              |
| **FAISS**                  | Fast similarity search via vector indexing       |
| **Qwen (Tongyi)**          | Large language model for answer generation       |
| **HuggingFace Embeddings** | Sentence embedding using `MiniLM-L6-v2`          |
| **Mineru**                 | High-quality PDF → Markdown parser               |
| **MarkdownHeaderTextSplitter** | Markdown section splitter for chunking      |

---

## 🛠️ About `convert_tables.py`

This script:

- Detects `<html>...</html>` or `<table>...</table>` blocks in the `.md` file
- Converts HTML tables to Markdown using `BeautifulSoup` + `markdownify`
- Keeps image captions or titles (actual image content is ignored)

Just run:

```bash
python convert_tables.py
```

The cleaned output (used by `main.py`) will be saved to `data/some_pdf_cleaned.md`.

---

## 💡 Tips

- You can ask questions in **Chinese**, even if the paper is written in **English**
- Supports questions about **tables**, as long as they're in Markdown format
- Ignores image content, but retains their **titles/descriptions** for context

---

## 🔮 TODO

- [ ] Enhance OCR for images and extract figure captions
- [ ] Structure table reasoning with schema-based extraction
- [ ] Add a frontend UI (e.g., Streamlit) for interactive QA

---

## 🤝 Acknowledgements

Thanks to the following open-source projects:

- [LangChain](https://github.com/langchain-ai/langchain)
- [Qwen (Tongyi)](https://modelscope.cn/models/qwen/Qwen1.5-Chat/summary)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Mineru](https://github.com/opendatalab/MinerU)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 📬 Contact

Feel free to open an issue or submit a PR if you have questions or suggestions 🙌
