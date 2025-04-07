# 📚 Academic Paper RAG System

一个基于 LangChain + 通义千问（Qwen）+ FAISS + HuggingFace 构建的学术论文问答系统，支持从 PDF 自动转换到结构化 Markdown，并通过 RAG（检索增强生成）方式对论文中的文本与表格内容进行自然语言问答。

---

## 🧭 项目流程总览

1. 🧾 **准备论文 PDF 文件**
2. 🔍 **使用 [Mineru](https://github.com/opendatalab/MinerU)** 工具将 PDF 转换为 Markdown 格式：
   ```bash
   magic-pdf -p {some_pdf} -o {some_output_dir} -m auto
   ```
     ✔️ 输出为 `some_output_dir/auto/some_pdf.md`
3. 🧹 **运行 `convert_tables.py` 脚本**：将 Mineru 输出中的 HTML 表格转换为纯文本 Markdown 表格，便于后续切分与嵌入
   ```bash
   python convert_tables.py
   ```
   ✔️ 输出为 `data/some_pdf_cleaned.md`

4. 🤖 **运行 `main.py` 构建 RAG 问答系统**
   ```bash
   python main.py
   ```

---

## 📂 项目结构

```
.
├── data/
│   ├── some_pdf.md            # Mineru 初始生成的 Markdown（含 HTML 表格）
│   └── some_pdf_cleaned.md        # 清洗后的 Markdown（纯文本表格）
├── convert_tables.py              # 表格格式转换脚本（HTML → Markdown）
├── faiss_index/                   # FAISS 索引存储目录
├── main.py                        # RAG 问答主逻辑
├── .env                           # 环境变量（用于通义 API Key）
└── README.md                      # 项目说明文件（本文件）
```

---

## ⚙️ 安装依赖

```bash
pip install langchain dashscope faiss-cpu python-dotenv sentence-transformers markdownify beautifulsoup4
```

---

## 🔑 配置通义千问 API Key

在项目根目录创建 `.env` 文件，并添加以下内容：

```ini
ALIYUN_API_KEY=你的阿里云 DashScope API Key
```

可在 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) 获取 Key。

---

## 📜 示例查询

运行 `main.py` 后，可直接用中文进行查询，例如：

```python
query = "请简要说明这篇论文提出的核心方法是什么？"
response = qa_chain.invoke(query)
print(response)
```

系统会检索最相关的 Markdown 内容片段并调用 Qwen 模型生成回答。

---

## 🧠 技术栈

| 模块          | 功能说明                          |
| ------------- | --------------------------------- |
| **LangChain** | 构建向量检索与问答链              |
| **FAISS**     | 文本向量索引和高效检索            |
| **Qwen (通义千问)** | 基于大模型生成自然语言回答       |
| **HuggingFace Embeddings** | 语义向量构建（`MiniLM-L6-v2`） |
| **Mineru**    | 高质量 PDF → Markdown 转换工具    |
| **MarkdownHeaderTextSplitter** | 按标题切分 Markdown 段落      |

---

## 🛠️ convert_tables.py 简介

该脚本会：

- 扫描 `.md` 文件中的 `<html>...</html>` HTML 标签
- 使用 `BeautifulSoup` 和 `markdownify` 自动将其转换为标准 Markdown 表格语法
- 保留图像的文字标题（图像内容不做嵌入）

你只需要执行：

```bash
python convert_tables.py
```

输出文件将保存在 `data/VideoRAG_cleaned.md`，供 `main.py` 使用。

---

## 💡 Tips

- 查询可以是中文，文档内容为英文也能精准检索
- 支持回答涉及表格的内容（表格需为纯文本）
- 图像部分会忽略图片，只保留描述/标题信息作为上下文

---

## 🔮 TODO

- [ ] 图像 OCR 与标题上下文增强
- [ ] 表格结构化分析增强推理能力
- [ ] 加入前端 UI（例如 Streamlit）用于交互式问答

---

## 🤝 致谢

感谢以下开源项目的强力支持：

- [LangChain](https://github.com/langchain-ai/langchain)
- [Qwen 通义千问](https://modelscope.cn/models/qwen/Qwen1.5-Chat/summary)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Mineru](https://github.com/opendatalab/MinerU)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 📬 联系

如有建议、改进或问题，欢迎提 Issue 或提交 PR 🙌
