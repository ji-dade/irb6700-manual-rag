"""
混合检索器：用 RRF(Reciprocal Rank Fusion)合并向量检索和 BM25 检索的结果。

职责：
- 同时调用 VectorRetriever 和 BM25Retriever
- 用 RRF 按"排名"合并两路结果，规避分数量纲不一致的问题
- 输入 query,输出 Top-K Document
- 支持按 metadata 过滤（下发给两个子检索器）
- 为 Rerank 提供一致的 retrieve() 接口

设计说明：
- RRF 只看排名，不看原始分数，规避"距离 vs BM25分"量纲冲突
- 同一文档可能被两路都召回，用 RRF 分数累加，让它排名更靠前
- 去重按 (page_content) 或 (page + 内容前缀) 判断，避免同一 chunk 重复占位
- 两路子检索器各自多召回一些(fetch_k),再融合,提升融合效果
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from src.config import TOP_K
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合检索器：向量 + BM25,用 RRF 合并。

    用法：
        retriever = HybridRetriever(vector_retriever, bm25_retriever, default_k=3)
        docs = retriever.retrieve("第 3 步的注释是什么")
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        default_k: int = TOP_K,
        rrf_k: int = 60,
        fetch_k: int = 20,
    ):
        """
        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25 检索器
            default_k: 默认返回条数
            rrf_k: RRF 平滑常数，默认 60(经验值)
            fetch_k: 每路子检索器先各召回多少条，再融合
        """
        if vector_retriever is None or bm25_retriever is None:
            raise ValueError("vector_retriever 和 bm25_retriever 不能为 None")

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.default_k = default_k
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

        logger.info(
            f"HybridRetriever 初始化完成, default_k={default_k}, "
            f"rrf_k={rrf_k}, fetch_k={fetch_k}"
        )

    def retrieve(self,query: str,k: Optional[int] = None,filter: Optional[Dict[str, Any]] = None,) -> List[Document]:
        """
        混合检索主入口。

        Args:
            query: 用户问题
            k: 返回条数,None 则用 default_k
            filter: metadata 过滤条件，下发给两个子检索器

        Returns:
            List[Document]: Top-K 结果,metadata 保留原始字段
        """
        if not query or not query.strip():
            logger.warning("query 为空，返回空结果")
            return []

        k = k or self.default_k

        logger.info(f"[HybridRetriever] query='{query}', k={k}, filter={filter}")

        # 1. 两路各自多召回一些（fetch_k），提高融合质量
        vec_docs = self.vector_retriever.retrieve(query, k=self.fetch_k, filter=filter)
        bm25_docs = self.bm25_retriever.retrieve(query, k=self.fetch_k, filter=filter)

        logger.info(
            f"[HybridRetriever] 向量召回 {len(vec_docs)} 条, "
            f"BM25 召回 {len(bm25_docs)} 条"
        )

        # 2. RRF 融合
        fused = self._rrf_fuse(vec_docs, bm25_docs)

        # 3. 取 Top-K
        top_docs = [doc for doc, _ in fused[:k]]

        logger.info(f"[HybridRetriever] 融合后命中 {len(top_docs)} 条")

        return [self._normalize_doc(doc) for doc in top_docs]

    # ---------- 内部方法 ----------

    def _rrf_fuse(self,vec_docs: List[Document],bm25_docs: List[Document],) -> List[tuple]:
        """
        RRF 融合两路结果。

        Args:
            vec_docs: 向量检索结果（已按相关性排序）
            bm25_docs: BM25 检索结果（已按相关性排序）

        Returns:
            List[Tuple[Document, float]]: 按 RRF 分从高到低排序的 (doc, score)
        """
        rrf_scores: Dict[str, float] = {}   # key -> RRF 分
        doc_map: Dict[str, Document] = {}   # key -> Document（去重用）

        # 向量路：rank 从 1 开始
        for rank, doc in enumerate(vec_docs, start=1):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map.setdefault(key, doc)

        # BM25 路
        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map.setdefault(key, doc)

        # 按 RRF 分从高到低排序
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], score) for key, score in ranked]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """
        生成文档去重 key。

        用 (page, page_content 前 100 字) 做 key:
        - 同一页的同一段内容，在两路里应视为同一文档
        - 只用 page 不行(一页可能切出多个 chunk)
        - 只用内容不行(不同页可能有重复模板文本)

        Args:
            doc: Document

        Returns:
            str: 唯一 key
        """
        page = doc.metadata.get("page", "unknown")
        content_head = (doc.page_content or "")[:100]
        return f"{page}||{content_head}"

    @staticmethod
    def _normalize_doc(doc: Document) -> Document:
        """
        规范化单条检索结果，保证关键 metadata 存在。

        Args:
            doc: 原始 Document

        Returns:
            Document: metadata 补齐后的 Document
        """
        meta = dict(doc.metadata or {})
        meta.setdefault("page", None)
        meta.setdefault("source", None)
        return Document(page_content=doc.page_content, metadata=meta)


if __name__ == "__main__":
    # 单测: python -m src.retrievers.hybrid_retriever
    from src.utils.logger import setup_logger
    from src.vectorstore.chroma_store import load_vectorstore

    logger = setup_logger("hybrid_retriever_test")

    logger.info("=" * 60)
    logger.info("开始测试 HybridRetriever")
    logger.info("=" * 60)

    vectorstore = load_vectorstore()

    # 1. 构建两个子检索器
    vector_retriever = VectorRetriever(vectorstore, default_k=3)
    bm25_retriever = BM25Retriever(vectorstore, default_k=3)

    # 2. 构建混合检索器
    retriever = HybridRetriever(
        vector_retriever,
        bm25_retriever,
        default_k=3,
        fetch_k=20,
    )

    # 3. 测试
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

        results = retriever.retrieve(q, k=3)
        for i, doc in enumerate(results):
            logger.info(f"--- 结果 {i+1} (第 {doc.metadata['page']} 页) ---")
            logger.info(f"\n{doc.page_content[:500]}")

    # 4. 演示 metadata 过滤：只看表格 chunk
    logger.info("=" * 60)
    logger.info("过滤测试: 只看表格 chunk")
    logger.info("=" * 60)
    table_results = retriever.retrieve("安装步骤", k=3, filter={"is_table": True})
    for i, doc in enumerate(table_results):
        logger.info(f"--- 表格结果 {i+1} (第 {doc.metadata['page']} 页) ---")
        logger.info(f"\n{doc.page_content[:300]}")