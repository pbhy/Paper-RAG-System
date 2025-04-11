from langchain.text_splitter import MarkdownHeaderTextSplitter

# Read the Markdown text
with open("data/VideoRAG_cleaned.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# Define header levels to split on
headers_to_split_on = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# Split using MarkdownHeaderTextSplitter
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(md_text)

# Extract plain text content
raw_chunks = [doc.page_content.strip() for doc in docs]

# Merge chunks that are too short
merged_chunks = []
buffer = ""

min_chunk_length = 200  # Minimum chunk length threshold

for chunk in raw_chunks:
    if len(chunk) < min_chunk_length:
        buffer += "\n\n" + chunk  # Accumulate into buffer
    else:
        if buffer:
            merged_chunks.append(buffer.strip())
            buffer = ""
        merged_chunks.append(chunk.strip())

# Add any remaining buffer content
if buffer:
    merged_chunks.append(buffer.strip())

# Final chunks to use
chunks = merged_chunks

# Embedding
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = [embeddings.embed_query(chunk) for chunk in chunks]

# Vector Store
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Retriever
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True  # Allow pickle deserialization
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks


# Example queries (uncomment to use)
# query = "What data is recorded in Table 4?"
# docs = retriever.invoke(query)

# for doc in docs:
#     print(doc.page_content)

# for doc in retriever.get_relevant_documents("Did the authors use GPT-4 as the base model?"):
#     print(doc.page_content[:300])


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
# query = "What are the key differences mentioned in the paper compared to previous works?"
query = "请总结作者在实验部分中得出的关键结论"
#请简要说明这篇论文提出的核心方法是什么？论文中提到了哪些与以往工作的不同之处？该方法适用于哪些场景？是否具有通用性？
#Table 1 展示了哪些类型的视频数据？各自的数据量如何？
#Table 2 的主要内容是什么？它在论文中起到了什么作用？
#Figure 1 中展示的场景在整个系统中起到什么作用？
#请总结作者在实验部分中得出的关键结论
response = qa_chain.invoke(query)

print(response)
