"""
FastAPI 服务入口。

用法：
    python -m scripts.serve
    # 或
    uvicorn scripts.serve:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.generate.chain import build_rag_chain
from src.generate.llm import get_llm
from src.vectorstore.chroma_store import load_vectorstore
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank import Reranker
from src.config import TOP_K
from src.utils.logger import setup_logger

logger = setup_logger("serve")

app = FastAPI(title="IRB 6700 RAG API")


# ========== 全局初始化==========
logger.info("正在初始化 RAG 链...")

vectorstore = load_vectorstore()

vector_retriever = VectorRetriever(vectorstore, default_k=TOP_K)
bm25_retriever = BM25Retriever(vectorstore, default_k=TOP_K)
hybrid = HybridRetriever(vector_retriever, bm25_retriever, default_k=TOP_K, fetch_k=20)

reranker = Reranker()

chain = build_rag_chain(hybrid, reranker)   # 调用工厂函数，拿到链

logger.info("RAG 链初始化完成")


# ========== 请求/响应模型 ==========
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# ========== 路由 ==========
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer = chain.invoke(req.question)
    return AskResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # python -m src.serve.api
    import uvicorn
    print("=== 服务启动 ===") 
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("=== 服务结束 ===")     

 