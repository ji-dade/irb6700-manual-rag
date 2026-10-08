"""
入库入口脚本。

用法：
    python -m scripts.ingest                    # 全量入库
    python -m scripts.ingest --limit 150        # 只入库前 150/N 块
    python -m scripts.ingest --persist-dir ./data/chroma_db_bge  # 换目录
    python -m scripts.ingest --batch-size 20    # 每批入库 20/N 块
"""

import argparse
from src.pipeline import ingest_pipeline
from src.config import PDF_PATH, CHROMA_DIR
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 入库脚本")
    parser.add_argument("--pdf", type=str, default=str(PDF_PATH), help="PDF 路径")
    parser.add_argument("--persist-dir", type=str, default=str(CHROMA_DIR), help="向量库目录")
    parser.add_argument("--limit", type=int, default=None, help="只入库前 N 块")
    parser.add_argument("--batch-size", type=int, default=50, help="每批入库块数")
    return parser.parse_args()


def main():
# python -m scripts.ingest --persist-dir ./data/chroma_db_bge --batch-size 20
    args = parse_args()

    # 初始化日志
    setup_logger("src")   # 配置 "src" 父 logger

    # 调 pipeline
    ingest_pipeline(
        pdf_path=args.pdf,
        persist_dir=args.persist_dir,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()