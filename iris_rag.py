import pandas as pd

# 将 iris.csv 读取为 pandas DataFrame 并转换为 Markdown 表格
df = pd.read_csv("data/iris.csv")

# 如果你要添加一个 markdown 标题作为 H1，用于后续 header 切分：
md_text = "# Iris Dataset\n\n" + df.to_markdown(index=False)

from langchain.text_splitter import MarkdownHeaderTextSplitter

# 定义只按 "# Iris Dataset" 作为一级标题切分（模拟你的 md 结构）
headers_to_split_on = [
    ("#", "H1"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(md_text)

# 提取 page_content 字符串
raw_chunks = [doc.page_content.strip() for doc in docs]

merged_chunks = []
buffer = ""
min_chunk_length = 200

for chunk in raw_chunks:
    if len(chunk) < min_chunk_length:
        buffer += "\n\n" + chunk
    else:
        if buffer:
            merged_chunks.append(buffer.strip())
            buffer = ""
        merged_chunks.append(chunk.strip())

if buffer:
    merged_chunks.append(buffer.strip())

# 最终 chunks
chunks = merged_chunks


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 使用 HuggingFace Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 构建向量数据库
vectorstore = FAISS.from_texts(chunks, embeddings)

# 保存索引
vectorstore.save_local("faiss_index_iris")

# 加载
vectorstore = FAISS.load_local(
    "faiss_index_iris", 
    embeddings, 
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Integrate Qwen API (generator)
from langchain_community.chat_models import ChatTongyi
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import dashscope
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Read API Key from environment variables
api_key = os.getenv("ALIYUN_API_KEY")
if not api_key:
    raise ValueError("ALIYUN_API_KEY not found. Please check your .env configuration.")

llm = ChatTongyi(model="qwen-turbo",
                 api_key=api_key,
                 temperature=0)

# Custom prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an academic assistant. Below is the relevant content retrieved from a paper. Please answer the user's question based on this content.

[Paper Content]
{context}

[Question]
{question}

Please respond concisely and naturally. Do not provide scores, comparisons, or multiple answers.
If you cannot determine the answer, simply state that you are unsure.
"""
)

# Build RAG QA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# Execute a query
query = "每种鸢尾有多少个观测值？"

response = qa_chain.invoke(query)

print(response)
