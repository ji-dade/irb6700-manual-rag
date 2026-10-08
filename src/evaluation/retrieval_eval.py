"""
检索评估：读 questions.json,跑检索,算 Recall@K 和 MRR。

职责：
- 提供 eval_retriever() 纯评估函数
- 提供 load_questions() 问题集加载
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from src.config import PROJECT_ROOT, TOP_K
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank import Reranker
from src.vectorstore.chroma_store import load_vectorstore

logger = logging.getLogger(__name__)

QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.json"


def load_questions(path: str = None) -> List[Dict]:
    """
    加载问题集。

    Args:
        path: JSON 路径,None 则用默认

    Returns:
        List[Dict]: 问题列表
    """
    path = Path(path) if path else QUESTIONS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_retriever(retriever, questions, k=5, name="retriever") -> Dict:
    """
    评估一个检索器。

    指标：
    - Recall@K: 前 K 条里有没有命中 expected_pages
    - MRR: 第一个命中的排名倒数

    Args:
        retriever: 有 retrieve(query, k) 方法的检索器
        questions: 问题列表
        k: 取前 K 条
        name: 检索器名字

    Returns:
        Dict: {name, recall, mrr, k}
    """
    hit_count = 0
    mrr_sum = 0.0
    refusal_count = 0

    for q in questions:
        expected = set(q["expected_pages"])
        results = retriever.retrieve(q["question"], k=k)
        result_pages = [doc.metadata.get("page") for doc in results]

        if q["type"] == "refusal":
            refusal_count += 1
            continue

        hit = any(p in expected for p in result_pages)
        if hit:
            hit_count += 1
            for rank, p in enumerate(result_pages, start=1):
                if p in expected:
                    mrr_sum += 1.0 / rank
                    break

    n = len([q for q in questions if q["type"] != "refusal"])
    recall = hit_count / n if n > 0 else 0
    mrr = mrr_sum / n if n > 0 else 0

    logger.info(f"[{name}] Recall@{k}={recall:.3f}, MRR={mrr:.3f}, "
                f"命中 {hit_count}/{n}, 拒答型 {refusal_count} 条跳过")

    return {"name": name, "recall": recall, "mrr": mrr, "k": k}