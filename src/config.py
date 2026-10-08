
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db_bge"

# PDF 文件
PDF_FILENAME = "3HAC044266 PM IRB 6700-zh-cn.pdf"
PDF_PATH = RAW_PDF_DIR / PDF_FILENAME

# 切分参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ---- Embedding（Ollama）----
EMBEDDING_MODEL = "bge-m3"       
OLLAMA_BASE_URL = "http://localhost:11434"

# ---- Reranker（本地加载）----
RERANKER_PATH = PROJECT_ROOT / "models" / "bge-reranker-base"

# 检索参数
TOP_K = 10
RERANK_TOP_K = 3

# llm
LLM_MODEL = "deepseek-v4-pro"
LLM_BASE_URL = "https://api.deepseek.com"



