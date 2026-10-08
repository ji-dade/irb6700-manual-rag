"""
BM25 检索器：基于关键词的经典检索算法，封装成和 VectorRetriever 一致的接口。

职责：
- 从向量库拉取全部(chunk)构建内存 BM25 索引
- 用 jieba 对中文分词
- 输入 query,输出 Top-K Document
- 支持按 metadata 过滤
- 为 Hybrid / Rerank 提供一致的 retrieve() 接口

设计说明：
- BM25 是"字面匹配"，擅长编号、型号、专有名词；向量检索擅长语义。两者互补
- 分词质量直接决定 BM25 效果，中文必须分词，不能按空格 split
- 过滤放在打分之后：先算全部分数，再按 metadata 筛，避免破坏 BM25 的统计
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from src.config import TOP_K

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 关键词检索器：基于词频统计的字面匹配检索。

    用法：
        retriever = BM25Retriever(vectorstore, default_k=3)
        docs = retriever.retrieve("第 3 步的注释是什么")
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        default_k: int = TOP_K,
    ):
        """
        Args:
            vectorstore: 向量库，用来拉取全部 chunk(BM25 需要全量文本建索引)
            default_k: 默认返回条数
        """
        if vectorstore is None:
            raise ValueError("vectorstore 不能为 None")

        self.vectorstore = vectorstore
        self.default_k = default_k

        # 从向量库拉全部 chunk，构建 BM25 索引
        self.documents: List[Document] = self._load_all_documents()
        self.tokenized_corpus: List[List[str]] = [
            self._tokenize(doc.page_content) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(
            f"BM25Retriever 初始化完成, default_k={default_k}, "
            f"索引 chunk 数={len(self.documents)}"
        )

    def retrieve(self,query: str,k: Optional[int] = None,filter: Optional[Dict[str, Any]] = None,) -> List[Document]:
        """
        BM25 检索主入口。

        Args:
            query: 用户问题
            k: 返回条数,None 则用 default_k
            filter: metadata 过滤条件，如 {"is_table": True} 或 {"page": 3}

        Returns:
            List[Document]: Top-K 结果,metadata 保留原始字段
        """
        if not query or not query.strip():
            logger.warning("query 为空，返回空结果")
            return []

        k = k or self.default_k

        logger.info(f"[BM25Retriever] query='{query}', k={k}, filter={filter}")

        # 1. query 分词
        tokenized_query = self._tokenize(query)

        # 2. 给全部 chunk 打分
        scores = self.bm25.get_scores(tokenized_query)

        # 3. 按分数从高到低排序，得到 (doc, score) 列表
        ranked: List[Tuple[Document, float]] = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 4. 按 filter 过滤（打分后再筛，保证 BM25 统计不变）
        if filter:
            ranked = [
                (doc, score)
                for doc, score in ranked
                if self._match_filter(doc, filter)
            ]

        # 5. 取 Top-K
        top_docs = [doc for doc, _ in ranked[:k]]

        logger.info(f"[BM25Retriever] 命中 {len(top_docs)} 条")

        return [self._normalize_doc(doc) for doc in top_docs]

    # ---------- 内部方法 ----------

    def _load_all_documents(self) -> List[Document]:
        """
        从向量库拉取全部 chunk。

        Chroma 的 get() 返回 {"documents": [...], "metadatas": [...]}。
        需要把文本和 metadata 重新组装成 Document。

        Returns:
            List[Document]: 全部 chunk
        """
        try:
            data = self.vectorstore.get(include=["documents", "metadatas"])
        except Exception as e:
            # 不同 LangChain 版本 / 后端接口略有差异，兼容一下
            logger.error(f"拉取向量库全部文档失败: {e}")
            logger.info(f"get() 返回类型: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            raise

        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        logger.info(f"拉取到 {len(texts)} 条文档, {len(metadatas)} 条 metadata")

        docs = []
        for text, meta in zip(texts, metadatas):
            if not text:
                continue
            docs.append(Document(page_content=text, metadata=meta or {}))

        return docs

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        中文分词。jieba默认模式

        Args:
            text: 原始文本

        Returns:
            List[str]: 分词后的 token 列表
        """
        return [w for w in jieba.lcut(text) if w.strip()] # 把文本切成词列表,然后过滤空内容.

    @staticmethod
    def _match_filter(doc: Document, filter: Dict[str, Any]) -> bool:
        """
        简易 metadata 过滤：所有条件都满足才算命中。

        Args:
            doc: 待检查的 Document
            filter: 过滤条件，如 {"is_table": True, "page": 3}

        Returns:
            bool: 是否满足全部条件
        """
        for key, expected in filter.items():
            if doc.metadata.get(key) != expected:
                return False
        return True

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
    # 单测: python -m src.retrievers.bm25_retriever
    from src.utils.logger import setup_logger
    from src.vectorstore.chroma_store import load_vectorstore

    logger = setup_logger("bm25_retriever_test")

    logger.info("=" * 60)
    logger.info("开始测试 BM25Retriever")
    logger.info("=" * 60)

    vectorstore = load_vectorstore()

    # 1. 构建检索器
    retriever = BM25Retriever(vectorstore, default_k=3)

    # 2. 测试零件号
    results = retriever.retrieve("3HAC15556-1", k=3)

    for i, doc in enumerate(results):
        logger.info(f"--- 结果 {i+1} (第 {doc.metadata['page']} 页) ---")
        logger.info(doc.page_content[:200])

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

        results2 = retriever.retrieve(q, k=3)
        for i, doc in enumerate(results2):
            logger.info(f"--- 结果 {i+1} (第 {doc.metadata['page']} 页) ---")
            logger.info(f"\n{doc.page_content[:500]}")

    # 4. 只看表格 chunk
    logger.info("=" * 60)
    logger.info("过滤测试: 只看表格 chunk")
    logger.info("=" * 60)
    table_results = retriever.retrieve("安装步骤", k=3, filter={"is_table": True})
    for i, doc in enumerate(table_results):
        logger.info(f"--- 表格结果 {i+1} (第 {doc.metadata['page']} 页) ---")
        logger.info(f"\n{doc.page_content[:300]}")