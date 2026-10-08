"""
重排序器：用 Cross-Encoder 对粗排召回的候选做精排。

职责：
- 接收 (query, 候选 Document 列表)，对每对 (query, doc) 打分
- 按分数从高到低排序，返回 Top-K
- 不负责检索，只负责"对已有候选重新排序"

说明：
- Cross-Encoder 把 query 和 doc 拼一起输入模型,比向量相似度准
- 返回的 Document 保留原 metadata,并加一个 rerank_score 方便调试
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document
from src.config import RERANKER_PATH, TOP_K

logger = logging.getLogger(__name__)


class Reranker:
    """
    重排序器:Cross-Encoder 精排。

    用法：
        reranker = Reranker()
        candidates = hybrid_retriever.retrieve(query, k=20)   # 粗排 20 条
        final = reranker.rerank(query, candidates, top_k=3)   # 精排取 3
    """

    def __init__(
        self,
        model_name: str = None,
    ):
        """
        Args:
            model_name: Cross-Encoder 模型名(reranker)
            max_length: 单对 (query, doc) 的最大 token 长度，超出截断
        """
        # model_name: str = "BAAI/bge-reranker-base", # 下载
        self.model_name = str(model_name or RERANKER_PATH)
        self._model = None
        logger.info(f"Reranker 初始化完成, model_name={self.model_name}")

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = TOP_K,
    ) -> List[Document]:
        """
        精排主入口。

        Args:
            query: 用户问题
            docs: 粗排召回的候选 Document 列表
            top_k: 精排后返回几条

        Returns:
            List[Document]: 按相关性从高到低排序的 Top-K,metadata 里附带 rerank_score
        """
        if not query or not query.strip():
            logger.warning("query 为空，返回空结果")
            return []

        if not docs:
            logger.warning("候选 docs 为空，返回空结果")
            return []

        logger.info(
            f"[Reranker] query='{query}', 候选 {len(docs)} 条, top_k={top_k}"
        )

        # 1. 懒加载模型
        model = self._get_model()

        # 2. 构造 (query, doc) 对
        pairs = [(query, doc.page_content or "") for doc in docs]

        # 3. 打分
        scores = model.predict(pairs)
        
        # 4. 按分数从高到低排序
        ranked: List[Tuple[Document, float]] = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # 5. 取 Top-K，把分数写进 metadata
        top_docs = []
        for doc, score in ranked[:top_k]:
            meta = dict(doc.metadata or {})
            meta["rerank_score"] = float(score)
            top_docs.append(Document(page_content=doc.page_content, metadata=meta))

        logger.info(f"[Reranker] 精排后返回 {len(top_docs)} 条")
        for i, doc in enumerate(top_docs):
            logger.info(
                f"[Reranker] 排名 {i+1}: 第 {doc.metadata.get('page')} 页, "
                f"score={doc.metadata.get('rerank_score'):.4f}"
            )

        return top_docs

    # ---------- 内部方法 ----------

    def _get_model(self):
        """
        懒加载 Cross-Encoder 模型。

        为什么懒加载：
        - 模型文件几百 MB,import 时就加载会让模块导入变慢
        - 首次 rerank 才加载，符合"用时才付代价"

        Returns:
            CrossEncoder: sentence_transformers 的 CrossEncoder 实例
        """
        if self._model is None:
            logger.info(f"首次加载 Cross-Encoder 模型: {self.model_name}")
            # 延迟 import，避免不用 rerank 时也依赖 sentence_transformers
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("模型加载完成")
        return self._model

    @staticmethod
    def _normalize_doc(doc: Document) -> Document:
        """
        规范化单条结果，保证关键 metadata 存在。
        (rerank 内部已经处理，这里保留以备扩展)

        Args:
            doc: 原始 Document

        Returns:
            Document: metadata 补齐后的 Document
        """
        meta = dict(doc.metadata or {})
        meta.setdefault("page", None)
        meta.setdefault("source", None)
        meta.setdefault("has_table", False)
        return Document(page_content=doc.page_content, metadata=meta)


if __name__ == "__main__":
    # 单测: python -m src.retrievers.rerank
    from src.utils.logger import setup_logger
    from src.vectorstore.chroma_store import load_vectorstore
    from src.retrievers.vector_retriever import VectorRetriever
    from src.retrievers.bm25_retriever import BM25Retriever
    from src.retrievers.hybrid_retriever import HybridRetriever

    logger = setup_logger("rerank_test")

    logger.info("=" * 60)
    logger.info("开始测试 Reranker")
    logger.info("=" * 60)

    vectorstore = load_vectorstore()

    # 1. 构建混合检索器（作为粗排）
    vector_retriever = VectorRetriever(vectorstore, default_k=20)
    bm25_retriever = BM25Retriever(vectorstore, default_k=20)
    hybrid = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        default_k=20,     # 粗排输出 20，给 rerank 当候选
        fetch_k=20,
    )

    # 2. 构建 reranker
    reranker = Reranker()

    # 3. 测试问题
    questions = [
        "检查安装的先决条件是什么",
        "检查安装机器人的先决条件第二部是什么",
        "安装适用的平台，或准备好机械臂的基座",
        "机器人中使用的材料弃置是指什么",
        "更换 IRB 6700 上的零件时",
        "校准术语"
    ]

    for q in questions:
        logger.info("=" * 60)
        logger.info(f"问题: {q}")
        logger.info("=" * 60)

        # 粗排：Hybrid 取 20 条候选
        candidates = hybrid.retrieve(q, k=20)

        # 精排：Rerank 取 Top-3
        results = reranker.rerank(q, candidates, top_k=3)

        for i, doc in enumerate(results):
            logger.info(f"--- 结果 {i+1} (第 {doc.metadata['page']} 页) ---")
            logger.info(f"\n{doc.page_content[:500]}")

    # 4. 过滤
    logger.info("=" * 60)
    logger.info("过滤测试: 只看表格 chunk")
    logger.info("=" * 60)
    table_candidates = hybrid.retrieve("安装步骤", k=20, filter={"is_table": True})
    table_results = reranker.rerank("安装步骤", table_candidates, top_k=3)
    for i, doc in enumerate(table_results):
        logger.info(f"--- 表格结果 {i+1} (第 {doc.metadata['page']} 页) ---")
        logger.info(f"\n{doc.page_content[:300]}")