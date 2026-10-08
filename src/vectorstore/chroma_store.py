"""
Chroma 向量库封装：入库、持久化、加载

职责：
- 创建/加载 Chroma 实例
- 批量入库
- 提供 get_vectorstore() 接口
"""

import logging
from pathlib import Path
from typing import List
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_DIR, EMBEDDING_MODEL
from src.vectorstore.embedder import get_embedder

logger = logging.getLogger(__name__)


def build_vectorstore(
    splits: List[Document],
    persist_dir: str = None,
    batch_size: int = 50,
    model_name: str = None,
) -> Chroma:
    """
    创建向量库并批量入库。

    Args:
        splits: 切分后的 Document 列表
        persist_dir: 持久化目录，默认从 config 读
        batch_size: 每批入库数量
        model_name: 嵌入模型名

    Returns:
        Chroma 实例
    """
    persist_dir = persist_dir or str(CHROMA_DIR)
    model_name = model_name or EMBEDDING_MODEL

    logger.info(f"开始入库: {len(splits)} 块 → {persist_dir}")
    logger.info(f"嵌入模型: {model_name}")
    #调嵌入模型
    embedder = get_embedder(model_name)
    # 
    vectorstore = Chroma(#  连接或创建持久化目录 返回Chroma
        embedding_function=embedder,
        persist_directory=persist_dir,
    )
    # 切块分批+入库
    for i in tqdm(range(0, len(splits), batch_size), desc="入库"): # 进度条库
        batch = splits[i:i + batch_size]# 切片
        vectorstore.add_documents(batch)

    logger.info(f"入库完成: {len(splits)} 块")
    return vectorstore


def load_vectorstore(persist_dir: str = None,model_name: str = None,) -> Chroma:
    """
    加载已有的向量库（不重新嵌入）

    Args:
        persist_dir: 持久化目录
        model_name: 嵌入模型名

    Returns:
        Chroma 实例
    """
    persist_dir = persist_dir or str(CHROMA_DIR)
    model_name = model_name or EMBEDDING_MODEL

    logger.info(f"加载向量库: {persist_dir}")

    embedder = get_embedder(model_name)# 把“问题”转成向量

    return Chroma(# 带着问题检索
        embedding_function=embedder,
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    # 单脚本测试: python -m src.vectorstore.chroma_store
    from src.loaders.pdf_loader import load_pdf
    from src.cleaners.footer_cleaner import clean_documents
    from src.chunking.splitter import split_documents
    from src.config import PDF_PATH
    from src.utils.logger import setup_logger

    logger = setup_logger("chroma_store_test")

    docs = load_pdf(str(PDF_PATH))
    cleaned = clean_documents(docs)
    splits = split_documents(cleaned)

    vectorstore = build_vectorstore(splits[:150])
    logger.info(f"测试入库完成，共 {len(splits[:150])} 块")