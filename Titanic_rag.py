import pandas as pd
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

### Step 1. 读取 Excel 数据为 DataFrame
df = pd.read_excel("data/Titanic.xlsx")

# 可选：填充空值，清洗表格（避免 markdown 表格出错）
df.fillna("N/A", inplace=True)

# Step 2. 转为 markdown 格式文本 + 添加一级标题
md_text = "# Titanic Passenger Dataset\n\n" + df.to_markdown(index=False)

### Step 3. 按 Markdown Header 切分
headers_to_split_on = [
    ("#", "H1"),  # 模拟你的原始 md 切分方式
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(md_text)

# Step 4. 提取文本 + 合并过短片段（与你之前相同）
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

chunks = merged_chunks

# 嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 构建向量数据库
vectorstore = FAISS.from_texts(chunks, embeddings)

# 保存
vectorstore.save_local("faiss_index_titanic")

# 加载向量库
vectorstore = FAISS.load_local(
    "faiss_index_titanic", 
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
query = "乘客的存活率是多少？乘客的平均年龄是多少？男女乘客的比例是多少？哪一类乘客的存活率最高？"

response = qa_chain.invoke(query)

print(response)
