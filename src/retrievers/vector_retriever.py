"""
向量检索器：把 Chroma 的相似度检索封装成统一接口。

职责：
- 封装 Chroma similarity_search,输入 query,输出 Top-K Document
- 统一返回格式 ,保留 page / source / has_table 等 metadata
- 支持按 metadata 过滤
- 为后续 BM25 / Hybrid / Rerank 提供一致的 retrieve() 接口

设计说明：
- 上层只依赖 retrieve(query, k) -> List[Document] ,不关心底层是向量还是关键词
- 元数据在检索结果里原样透传 ,方便引用来源、调试、重排
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from src.config import TOP_K

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    向量检索器：基于向量库做语义相似度检索。

    用法：
        retriever = VectorRetriever(vectorstore, TOP_K=3)
        docs = retriever.retrieve("检查安装的先决条件是什么")
    """

    def __init__(self,vectorstore: VectorStore,default_k: int = TOP_K,):   # config 读
        """
        Args:
            vectorstore: 已构建好的向量库(如 Chroma)需支持 similarity_search
            default_k: 默认返回条数,retrieve 时未指定 k 则用它
        """
        if vectorstore is None:
            raise ValueError("vectorstore 不能为 None")

        self.vectorstore = vectorstore
        self.default_k = default_k

        logger.info(f"VectorRetriever 初始化完成, default_k={default_k}")

    # 取回
    def retrieve(self,query: str,k: Optional[int] = None,filter: Optional[Dict[str, Any]] = None,) -> List[Document]:
        """
        向量检索主入口。

        Args:
            query: 用户问题
            k: 返回条数 ,None 则用 TOP_K
            filter: metadata 过滤条件 ,如 {"has_table": True} 或 {"page": 3}

        Returns:
            List[Document]: Top-K 结果 ,metadata 保留原始字段
        """
        if not query or not query.strip():# 包括字符串首尾的空白字符
            logger.warning("query 为空 ,返回空结果")
            return []

        k = k or self.default_k

        logger.info(f"[VectorRetriever] query='{query}', k={k}, filter={filter}")

        try:
            # 把 query 转成向量, 在 Chroma 里找最相似的 k 条遍历所有向量，算余弦相似度排序，取前 k 条
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter,
            )
        except TypeError:
            # 某些向量库后端不支持 filter ,降级为不带 filter
            logger.warning("当前向量库不支持 filter ,已忽略过滤条件")
            results = self.vectorstore.similarity_search(query, k=k)

        logger.info(f"[VectorRetriever] 命中 {len(results)} 条")

        # 补齐 metadata 字段
        return [self._normalize_doc(doc) for doc in results]

    @staticmethod # 静态方法装饰器
    def _normalize_doc(doc: Document) -> Document:
        """
        规范化单条检索结果 ,保证关键 metadata 存在。

        Args:
            doc: 原始 Document

        Returns:
            Document: metadata 补齐后的 Document
        """
        meta = dict(doc.metadata or {})
        # 补齐 metadata 字段
        meta.setdefault("page", None)
        meta.setdefault("source", None)
        meta.setdefault("has_table", meta.get("is_table", False))       
        return Document(page_content=doc.page_content, metadata=meta)


if __name__ == "__main__":
    # 单测: python -m src.retrievers.vector_retriever
    from src.utils.logger import setup_logger
    from src.vectorstore.chroma_store import load_vectorstore

    logger = setup_logger("vector_retriever_test")

    logger.info("=" * 60)
    logger.info("开始测试 VectorRetriever")
    logger.info("=" * 60)

    vectorstore = load_vectorstore()

    # 2. 构建检索器
    retriever = VectorRetriever(vectorstore,default_k=3) # 实验拿3条

    # 3. 测试问题
    questions = [
        "检查安装的先决条件是什么",
        "检查安装机器人的先决条件第二部是什么",
        "安装适用的平台 ,或准备好机械臂的基座",
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

    # 4. 测试 metadata 过滤：只看表格页
    logger.info("=" * 60)
    logger.info("过滤测试: 只看表格 chunk")
    logger.info("=" * 60)
    table_results = retriever.retrieve("安装步骤", k=3, filter={"is_table": True})
    for i, doc in enumerate(table_results):
        logger.info(f"--- 表格结果 {i+1} (第 {doc.metadata['page']} 页) ---")
        logger.info(f"\n{doc.page_content[:300]}")