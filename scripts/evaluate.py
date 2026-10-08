"""
构建检索器
评估入口脚本：交互式提问 + 评估导出 CSV。

用法：
    python -m scripts.evaluate                    # 提示选择
    python -m scripts.evaluate --interactive      # 交互
    python -m scripts.evaluate --eval             # 评估
    python -m scripts.evaluate --eval --k 10      # 自定义 K
"""

import argparse
import csv
import logging
from datetime import datetime

from src.config import PROJECT_ROOT, TOP_K, RERANK_TOP_K
from src.vectorstore.chroma_store import load_vectorstore
from src.retrievers.vector_retriever import VectorRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank import Reranker
from src.evaluation.retrieval_eval import load_questions, eval_retriever
from src.generate.chain import build_rag_chain
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"


# ========== 构建 ==========
def _build_all():
    """构建四种检索器/hybrid + reranker"""
    vectorstore = load_vectorstore()

    vector_retriever = VectorRetriever(vectorstore, default_k=TOP_K)
    bm25_retriever = BM25Retriever(vectorstore, default_k=TOP_K)
    hybrid = HybridRetriever(vector_retriever, bm25_retriever,
                             default_k=TOP_K, fetch_k=20)
    reranker = Reranker()

    class HybridRerankRetriever:
        def __init__(self, hybrid, reranker):
            self.hybrid = hybrid
            self.reranker = reranker

        def retrieve(self, query, k=5):
            candidates = self.hybrid.retrieve(query, k=20)
            return self.reranker.rerank(query, candidates, top_k=k)

    retrievers = {
        "向量": vector_retriever,
        "BM25": bm25_retriever,
        "Hybrid": hybrid,
        "Hybrid+Rerank": HybridRerankRetriever(hybrid, reranker),
    }

    return retrievers, hybrid, reranker


# ========== 模式一：交互式提问（走 LCEL 链） ==========
def interactive_mode(chain):
    """交互式提问：走完整 LCEL 链"""
    print()
    print("=" * 60)
    print("RAG 交互式提问（完整 LCEL 链）")
    print("=" * 60)
    print(f"链路: Hybrid 粗排 {TOP_K} 条 → Rerank 精排 {RERANK_TOP_K} 条 → LLM 生成")
    print("输入 'quit' / 'q' / 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            question = input("\n请输入相关问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not question:
            print("问题不能为空。")
            continue

        if question.lower() in ("quit", "q", "exit"):
            print("再见。")
            break

        print(f"\n[链路执行中] 检索 → 重排 → 生成...")
        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f"\n[错误] {e}")
            continue

        print()
        print("=" * 60)
        print(f"问题: {question}")
        print("=" * 60)
        print(f"\n回答:\n{answer}")
        print()


# ========== 模式二：评估 + CSV ==========
def eval_mode(questions, k=5):
    """评估四种检索器，输出对比表 + CSV"""
    logger.info("=" * 60)
    logger.info(f"开始检索评估(Recall@{k} / MRR)")
    logger.info("=" * 60)

    retrievers, _, _ = _build_all()

    results = []
    for name, retriever in retrievers.items():
        results.append(eval_retriever(retriever, questions, k=k, name=name))

    logger.info("=" * 60)
    logger.info("评估结果对比")
    logger.info("=" * 60)
    logger.info(f"{'方法':<20} {'Recall@' + str(k):<12} {'MRR':<12}")
    logger.info("-" * 50)
    for r in results:
        logger.info(f"{r['name']:<20} {r['recall']:<12.3f} {r['mrr']:<12.3f}")

    export_csv(results, k)
    return results


def export_csv(results, k):
    """导出评估结果到 CSV"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"retrieval_eval_{timestamp}.csv"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方法", f"Recall@{k}", "MRR"])
        for r in results:
            writer.writerow([r["name"], f"{r['recall']:.4f}", f"{r['mrr']:.4f}"])

    logger.info(f"评估结果已保存: {csv_path}")
    return csv_path

# ========== 入口 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="RAG 评估 + 交互式提问")
    parser.add_argument("--interactive", action="store_true", help="交互式提问")
    parser.add_argument("--eval", action="store_true", help="跑评估，导出 CSV")
    parser.add_argument("--k", type=int, default=5, help="Recall@K 的 K")
    parser.add_argument("--questions", type=str, default=None, help="问题集路径")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger("src")

    # 提示选择
    if not args.interactive and not args.eval:
        print()
        print("=" * 60)
        print("请选择运行模式")
        print("=" * 60)
        print("  1. 交互式提问 (interactive)")
        print("  2. 评估 (eval)")
        print("=" * 60)

        while True:
            choice = input("请输入运行模式 [1/2]: ").strip().lower()
            if choice in ("1", "interactive"):
                args.interactive = True
                break
            elif choice in ("2", "eval"):
                args.eval = True
                break
            else:
                print("无效输入，请选择 1、2")

    # 交互式：走 LCEL 链
    if args.interactive:
        _, hybrid, reranker = _build_all()
        chain = build_rag_chain(hybrid, reranker)
        interactive_mode(chain)

    # 评估
    if args.eval:
        questions = load_questions(args.questions)
        logger.info(f"加载 {len(questions)} 个问题")
        eval_mode(questions, k=args.k)


if __name__ == "__main__":
    main()