"""
LLM 封装：统一创建 LLM 客户端。

用法：
    from src.generate.llm import get_llm
    llm = get_llm()
"""

import os
import logging
from langchain_openai import ChatOpenAI
from src.config import LLM_MODEL, LLM_BASE_URL

logger = logging.getLogger(__name__)


def get_llm():
    """创建 LLM 客户端(DeepSeek)"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(f"环境变量API_KEY未设置")

    logger.info(f"创建 LLM: {LLM_MODEL} @ {LLM_BASE_URL}")

    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=api_key,
        temperature=0.1,
    )