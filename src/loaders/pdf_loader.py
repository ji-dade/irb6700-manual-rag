"""
PDF 加载器：把 PDF 文件转成 LangChain Document 列表。

职责：
- 用 pdfplumber 逐页读取
- 提取正文(extract_text)
- 提取表格并转成 Markdown 风格文本(extract_tables)
- 返回 Document 列表,每页一个 Document
"""

import logging
import pdfplumber
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> List[Document]:
    """
    加载单个 PDF 文件,返回 Document 列表。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        List[Document]: 每页一个 Document,metadata 包含 page 和 source
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    logger.info(f"开始加载 PDF: {pdf_path.name}")

    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            text, table_texts = _extract_page_content(page)

            # 整页为空,跳过
            if not text.strip() and not table_texts:
                continue

            # 构造 metadata
            metadata = {
                "page": page_num + 1,
                "source": str(pdf_path),
                "total_pages": total_pages,
                "has_table": bool(table_texts),
                "is_table": False
            }

            # 有表格才追加
            if table_texts:
                metadata["table_texts"] = table_texts

            docs.append(Document(
                page_content=text,
                metadata=metadata,
            ))
            

    logger.info(f"加载完成: {pdf_path.name}, 共 {len(docs)} 页")
    return docs


def _extract_page_content(page) -> Tuple[str, List[str]]:
    """
    提取单页内容,返回 (正文, 表格文本列表)。

    Args:
        page: pdfplumber 的 Page 对象

    Returns:
        Tuple[str, List[str]]: (正文, 表格文本列表)
    """
    text = page.extract_text() or ""
    table_texts = _extract_tables(page)
    return text, table_texts

def _extract_tables(page) -> List[str]:
    """
    提取单页的所有表格,转成 "列1 | 列2 | 列3" 格式。
    """
    tables = page.extract_tables()
    table_texts = []

    for table in tables:
        clean_table = [
            [cell if cell else "" for cell in row]
            for row in table
            if any(row)
        ]
        for row in clean_table:
            table_texts.append(" | ".join(row))

    return table_texts


def load_pdfs(pdf_dir: str) -> List[Document]:
    """
    批量加载目录下所有 PDF。
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"不是有效目录: {pdf_dir}")

    all_docs = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")

    for pdf_file in pdf_files:
        docs = load_pdf(str(pdf_file))
        all_docs.extend(docs)

    logger.info(f"批量加载完成,共 {len(all_docs)} 页")
    return all_docs


if __name__ == "__main__":
    # 单job测试: python -m src.loaders.pdf_loader
    from src.config import PDF_PATH
    from src.utils.logger import setup_logger

    logger = setup_logger("pdf_loader_test")

    logger.info("=" * 60)
    logger.info("开始测试 pdf_loader")
    logger.info("=" * 60)
    logger.info(f"PDF 路径: {PDF_PATH}")

    docs = load_pdf(str(PDF_PATH))

    logger.info(f"总页数: {len(docs)}")
    logger.info(f"第一页预览:\n{docs[0].page_content[:300]}")
    logger.info(f"第一页 metadata: {docs[0].metadata}")

    # 抽查第 100、400 页
    for i in [99, 399]:
        if i < len(docs):
            logger.info(f"--- 第 {i+1} 页预览 ---")
            logger.info(f"\n{docs[i].page_content[:500]}")