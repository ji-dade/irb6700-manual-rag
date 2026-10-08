"""
页脚清洗器：删除 PDF 页脚里的固定文本。

职责：
- 删除页脚固定行（页码、书名、文档编号、版权行）
- 删除末尾的“续前页”
- 保留页脚后面的章节标题
"""

import re
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ========== 正则模式 ==========
# 页码可能在书名前，也可能在书名后
FOOTER_PATTERN_1 = re.compile(
    r'\n?下一页继续\s*\n'
    r'(?:\d{1,3}\s*产品手册\s*-?\s*IRB\s*6700|产品手册\s*-?\s*IRB\s*6700\s*\d{1,3})\s*\n'
    r'3HAC044266-010\s*修订\s*:?\s*AL\s*\n'
    r'©\s*版权所有\s*2013\s*-\s*2026\s*ABB。保留所有权利。\s*\n?'
)

FOOTER_PATTERN_2 = re.compile(
    r'\n(?:\d{1,3}\s*产品手册\s*-?\s*IRB\s*6700|产品手册\s*-?\s*IRB\s*6700\s*\d{1,3})\s*\n'
    r'3HAC044266-010\s*修订\s*:?\s*AL\s*\n'
    r'©\s*版权所有\s*2013\s*-\s*2026\s*ABB。保留所有权利。\s*\n?'
)

TRAILING_PATTERN = re.compile(r'\n续前页\s*$')


def clean_footer(text: str) -> str:
    """
    清洗单页文本的页脚。

    Args:
        text: 单页的 page_content

    Returns:
        str: 清洗后的文本
    """
    # 删除带“下一页继续”的页脚
    text = FOOTER_PATTERN_1.sub('\n', text)
    # 删除不带“下一页继续”的页脚
    text = FOOTER_PATTERN_2.sub('\n', text)
    # 删除末尾的“续前页”
    text = TRAILING_PATTERN.sub('', text)
    return text.strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    """
    批量清洗 Document 列表。

    Args:
        docs: 原始 Document 列表

    Returns:
        List[Document]: 清洗后的 Document 列表
    """
    logger.info(f"开始清洗页脚，共 {len(docs)} 页")

    cleaned_docs = []
    for doc in docs:
        content = clean_footer(doc.page_content)
        cleaned_docs.append(
            type(doc)(page_content=content, metadata=doc.metadata)
        )

    logger.info(f"清洗完成，共 {len(cleaned_docs)} 页")
    return cleaned_docs

if __name__ == "__main__":
    # 单脚本测试: python -m src.cleaners.footer_cleaner
    from src.config import PDF_PATH
    from src.loaders.pdf_loader import load_pdf
    from src.utils.logger import setup_logger

    logger = setup_logger("footer_cleaner_test")

    logger.info("=" * 60)
    logger.info("开始测试 footer_cleaner")
    logger.info("=" * 60)

    docs = load_pdf(str(PDF_PATH))
    cleaned_docs = clean_documents(docs)

    # 对比清洗前后
    footer_page = None
    for i, doc in enumerate(docs):
        if "产品手册" in doc.page_content and "下一页继续" in doc.page_content:
            footer_page = i
            break

    if footer_page is not None:
        logger.info(f"=== 第 {footer_page + 1} 页 清洗前末尾 ===")
        logger.info(repr(docs[footer_page].page_content[-300:]))
        logger.info(f"=== 第 {footer_page + 1} 页 清洗后末尾 ===")
        logger.info(repr(cleaned_docs[footer_page].page_content[-300:]))
    else:
        logger.info("没有找到包含页脚的页")