"""
离线入库管道：加载 → 清洗 → 切分 → 入库。

职责：
- 串联 loaders → cleaners → chunking → vectorstore
- 提供统一的 ingest_pipeline() 入口
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.config import PDF_PATH, CHROMA_DIR
from src.loaders.pdf_loader import load_pdf
from src.cleaners.footer_cleaner import clean_documents
from src.chunking.splitter import split_documents
from src.vectorstore.chroma_store import build_vectorstore
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def ingest_pipeline(pdf_path: str = None,persist_dir: str = None,limit: Optional[int] = None,batch_size = 50):
    """
    链接完整的离线入库流程。

    Args:
        pdf_path: PDF 路径，默认从 config 读
        persist_dir: 向量库持久化目录，默认从 config 读
        limit: 只入库前 N 块(测试用),None 表示全量

    Returns:
        Chroma 实例
    """
    pdf_path = pdf_path or str(PDF_PATH)
    persist_dir = persist_dir or str(CHROMA_DIR)

    logger.info("=" * 60)
    logger.info("开始离线入库管道")
    logger.info("=" * 60)
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"持久化目录: {persist_dir}")
    logger.info(f"入库限制: {limit if limit else '全量'}")

    # 1. 加载
    docs = load_pdf(pdf_path)
    logger.info(f"[1/4] 加载完成: {len(docs)} 页")

    # 2. 清洗
    cleaned = clean_documents(docs)
    logger.info(f"[2/4] 清洗完成: {len(cleaned)} 页")

    # 3. 切分
    splits = split_documents(cleaned)
    logger.info(f"[3/4] 切分完成: {len(splits)} 块")

    # 4. 入库
    if limit: #有值时对列表做切片
        splits = splits[:limit]
        logger.info(f"[4/4] 限制入库: {len(splits)} 块")

    vectorstore = build_vectorstore(splits, persist_dir=persist_dir, batch_size=batch_size)
    logger.info(f"[4/4] 入库完成: {len(splits)} 块")

    logger.info("=" * 60)
    logger.info("离线入库管道完成")
    logger.info("=" * 60)

    return vectorstore