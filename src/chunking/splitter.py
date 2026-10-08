"""
切分器：普通页按字符切，表格页单独块
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_documents(docs: List[Document]) -> List[Document]:
    """
    切分 Document 列表。

    普通页：用 RecursiveCharacterTextSplitter 按字符切
    表格页：保留表格结构 单独块
    """
    logger.info(f"开始切分，共 {len(docs)} 页")

    text_splitter = RecursiveCharacterTextSplitter( # 中文字符
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    splits = []
    for doc in docs:
        # 1. 正文切分
        text_chunks = text_splitter.split_documents([doc])
        splits.extend(text_chunks)

        # 2. 表格单独成块，不切
        table_texts = doc.metadata.get("table_texts", [])
        if table_texts:
            table_content = "\n".join(table_texts)# 元素取出加换行符
            table_doc = Document(
                page_content=table_content,
                metadata={
                    "page": doc.metadata["page"],
                    "source": doc.metadata["source"],
                    "is_table": True,
                }
            )
            splits.append(table_doc)

    return splits

if __name__ == "__main__":
    # 单脚本测试: python -m src.chunking.splitter
    from src.config import PDF_PATH
    from src.loaders.pdf_loader import load_pdf
    from src.cleaners.footer_cleaner import clean_documents
    from src.utils.logger import setup_logger

    logger = setup_logger("splitter_test")

    logger.info("=" * 60)
    logger.info("开始测试 splitter")
    logger.info("=" * 60)

    docs = load_pdf(str(PDF_PATH))
    cleaned_docs = clean_documents(docs)
    splits = split_documents(cleaned_docs)

    logger.info(f"总块数: {len(splits)}")

    # 表格块统计（用 metadata 判断）
    table_chunks = [s for s in splits if s.metadata.get("is_table")]
    logger.info(f"表格块数: {len(table_chunks)}")
    for chunk in table_chunks[:3]:
        logger.info(f"页 {chunk.metadata['page']}, 长度 {len(chunk.page_content)}")
        logger.info(f"\n{chunk.page_content[:300]}")

    # 普通块统计
    text_chunks = [s for s in splits if not s.metadata.get("is_table")]
    logger.info(f"普通块数: {len(text_chunks)}")
    for chunk in text_chunks[:3]:
        logger.info(f"页 {chunk.metadata['page']}, 长度 {len(chunk.page_content)}")
        logger.info(f"\n{chunk.page_content[:300]}")