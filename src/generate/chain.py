"""
LCEL 链:把检索、重排序、提示词、LLM 串成一条链。

职责：
- 串联 HybridRetriever → Reranker → Prompt → LLM → 输出解析
- 提供 build_rag_chain() 入口
- 支持同步 invoke 和流式 stream

"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.config import TOP_K, RERANK_TOP_K
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank import Reranker
from src.generate.llm import get_llm

logger = logging.getLogger(__name__)


# ========== 提示词模板 ==========
RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是一个工业机器人维修手册的问答助手。
请只根据以下提供的上下文回答问题 ,并在回答末尾注明来源页码。
如果上下文里没有答案 ,请直接说“根据提供的资料 ,无法回答这个问题”。

上下文：
{context}

问题：{question}

回答："""
)


def format_docs(docs: List[Document]) -> str:
    """
    把检索到的 Document 列表拼成纯文本 ,每段带页码标注。
    """
    if not docs:
        return "（无相关上下文)"

    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("rerank_score")
        if score is not None:
            parts.append(f"[第 {page} 页 | 相关度 {score:.2f}]\n{doc.page_content}")
        else:
            parts.append(f"[第 {page} 页]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    llm=None,
):
    """
    组装 RAG 链。

    Args:
        hybrid_retriever: 混合检索器
        reranker: 重排序器
        llm: LLM 实例 ,None 则从 config 创建

    Returns:
        LCEL 链 ,调用 .invoke(question) 返回回答字符串
    """
    if llm is None:
        llm = get_llm()

    def retrieve_and_rerank(query: str) -> List[Document]:
        """先 Hybrid 粗排 ,再 Reranker 精排"""
        candidates = hybrid_retriever.retrieve(query, k=TOP_K)
        logger.info(f"[Chain] Hybrid 召回 {len(candidates)} 条")
        final_docs = reranker.rerank(query, candidates, top_k=RERANK_TOP_K)
        logger.info(f"[Chain] Rerank 后保留 {len(final_docs)} 条")
        return final_docs

    chain = (
        {
            "context": RunnableLambda(retrieve_and_rerank) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("RAG 链组装完成")
    return chain


if __name__ == "__main__":
    # 单测: python -m src.generate.chain
    from src.utils.logger import setup_logger
    from src.vectorstore.chroma_store import load_vectorstore
    from src.retrievers.vector_retriever import VectorRetriever
    from src.retrievers.bm25_retriever import BM25Retriever

    logger = setup_logger("chain_test")

    logger.info("=" * 60)
    logger.info("开始测试 RAG 链")
    logger.info("=" * 60)

    # 1. 加载向量库
    vectorstore = load_vectorstore()

    # 2. 构建检索器
    vector_retriever = VectorRetriever(vectorstore, default_k=TOP_K)
    bm25_retriever = BM25Retriever(vectorstore, default_k=TOP_K)
    hybrid = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        default_k=TOP_K,
        fetch_k=20,
    )

    # 3. 构建 Reranker
    reranker = Reranker()

    # 4. 组装链
    chain = build_rag_chain(hybrid, reranker)

    # 5. 测试问题
    questions = [
        "检查安装的先决条件是什么",
        "安装机器人的第 2 步是什么？",
        "机器人中使用的材料弃置是指什么",
        "更换 IRB 6700 上的零件时",
        "校准术语",
    ]

    for q in questions:
        logger.info("=" * 60)
        logger.info(f"问题: {q}")
        logger.info("=" * 60)

        answer = chain.invoke(q)
        logger.info(f"回答:\n{answer}")