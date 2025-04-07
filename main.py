from langchain.text_splitter import MarkdownHeaderTextSplitter

# 读取 Markdown 文本
with open("data/VideoRAG_cleaned.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# 指定标题分级规则
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# 使用 MarkdownHeaderTextSplitter 切分
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(md_text)

# 提取文本内容
raw_chunks = [doc.page_content.strip() for doc in docs]

# 合并过短片段的逻辑
merged_chunks = []
buffer = ""

min_chunk_length = 200  # 最小长度阈值

for chunk in raw_chunks:
    if len(chunk) < min_chunk_length:
        buffer += "\n\n" + chunk  # 累加到缓存
    else:
        if buffer:
            merged_chunks.append(buffer.strip())
            buffer = ""
        merged_chunks.append(chunk.strip())

# 添加最后缓存内容
if buffer:
    merged_chunks.append(buffer.strip())

# 最终 chunks 结果
chunks = merged_chunks

#Embedding
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = [embeddings.embed_query(chunk) for chunk in chunks]

#Vector Store
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("faiss_index")

#Retriever
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True  # 允许 pickle 反序列化
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # 取前 5 个最相关的片段


# #
# query = "Table 4 中记录了哪些数据？"
# docs = retriever.invoke(query)

# for doc in docs:
#     print(doc.page_content)




# for doc in retriever.get_relevant_documents("作者是否使用了 GPT-4 作为基础模型？"):
#     print(doc.page_content[:300])


#集成 Qwen API（生成器）
from langchain_community.chat_models import ChatTongyi
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import dashscope
from dotenv import load_dotenv

# 加载 .env 文件中的变量
load_dotenv()

# 从环境变量读取 API Key
api_key = os.getenv("ALIYUN_API_KEY")
if not api_key:
    raise ValueError("未找到 ALIYUN_API_KEY，请检查 .env 文件是否正确配置")

llm = ChatTongyi(model="qwen-turbo",
                  api_key=api_key,
                   temperature=0)



# 自定义 Prompt
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一位学术助手，下面是从论文中检索到的相关内容，请基于这些内容来回答用户的问题。

[论文内容]
{context}

[问题]
{question}

请用简洁自然语言回答，不要进行打分、比较、或生成多种答案。
如果无法确定答案，请直接说明你无法确定。
"""
)

# 构建 RAG 问答链，加入 prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)


# 进行查询
query = "论文中提到了哪些与以往工作的不同之处？"
response = qa_chain.invoke(query)

print(response)
