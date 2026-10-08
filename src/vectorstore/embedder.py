"""
嵌入模型封装：调用 Ollama 的客户端/统一创建和管理 Embedding 客户端
    
职责：
- 根据配置创建嵌入模型客户端
- 提供统一的 get_embedder() 接口

"""

import logging
from langchain_ollama import OllamaEmbeddings
from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embedder(model_name: str = None) -> OllamaEmbeddings:
    """
    创建嵌入模型客户端

    Args:
        model_name: 模型名，默认从 config 读

    Returns:
        OllamaEmbeddings 实例
    """
    model_name = model_name or EMBEDDING_MODEL
    logger.info(f"创建嵌入模型: {model_name}")

    return OllamaEmbeddings(model=model_name)


if __name__ == "__main__":
    # 单脚本测试: python -m src.vectorstore.embedder
    from src.utils.logger import setup_logger

    logger = setup_logger("embedder_test")

    embedder = get_embedder()
    vec = embedder.embed_query("测试一句话")
    logger.info(f"向量维度: {len(vec)}")
    logger.info(f"前 5 个值: {vec[:5]}")