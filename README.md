# IRB 6700 维修手册 RAG 系统

基于 pdfplumber + Chroma + bge-m3 + BM25 + Cross-Encoder + DeepSeek 的工业设备维修手册检索增强生成（RAG）系统。

针对 ABB IRB 6700 机器人产品手册构建，实现从 PDF 解析、表格提取、混合检索、重排序到 LLM 生成回答的完整流程。
## 技术栈

Python 3.11 / pdfplumber / Chroma / bge-m3 / jieba + rank_bm25 / bge-reranker-base（sentence-transformers）/LangChain LCEL/DeepSeek API / FastAPI

## 目录结构
```
RAG_System/
├── data/
│   ├── raw_pdfs/                    # 原始 PDF
│   ├── eval/                        # 评估问题集（questions.json）
│   ├── chroma_test/                 # 测试向量库（bge-m3，150块）
│   └── chroma_db_bge/               # 向量库（bge-m3，1669 块）
├── models/                          # bge-reranker-base 
├── reports/                         # 评估产出（CSV）
├── logs/                            # 运行日志
├── scripts/
│   ├── ingest.py                    # 入库入口
│   └── evaluate.py                  # 交互式提问 + 评估入口
├── serve/
│   └── api.py                        # FastAPI 路由
├── src/
│   ├── config.py                    # 配置（路径、chunk_size、模型名、TOP_K）
│   ├── pipeline.py                  # 离线入库管道
│   ├── loaders/
│   │   └── pdf_loader.py            # pdfplumber 加载 + 表格提取
│   ├── cleaners/
│   │   └── footer_cleaner.py        # 页脚清洗
│   ├── chunking/
│   │   └── splitter.py              # 切分（正文切分，表格单独成块）
│   ├── vectorstore/
│   │   ├── embedder.py              # Embedding 封装（bge-m3）
│   │   └── chroma_store.py          # Chroma 入库 + 持久化
│   ├── retrievers/
│   │   ├── vector_retriever.py      # 向量检索
│   │   ├── bm25_retriever.py        # BM25 关键词检索
│   │   ├── hybrid_retriever.py      # RRF 混合召回
│   │   └── rerank.py                # Cross-Encoder 重排序
│   ├── generate/
│   │   ├── llm.py                   # LLM 封装（DeepSeek）
│   │   └── chain.py                 # LCEL 链组装 + RAG 提示词
│   ├── evaluation/
│   │   └── retrieval_eval.py        # 检索评估（Recall@K、MRR）
├── requirements.txt
├── LICENSE
└── README.md
```
离线阶段(pipeline.py + scripts/ingest.py)
在线阶段(chain.py + scripts/evaluate.py)

## 快速开始
### 本地运行

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate        # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 拉取嵌入模型（本地 Ollama）
ollama pull bge-m3
### Embedding（Ollama）

# 4. 下载Reranker到本地
从 HuggingFace 下载 BAAI/bge-reranker-base 到 ./models/bge-reranker-base

# 5. 设置 DeepSeek API Key
export DEEPSEEK_API_KEY="sk-..."
```
## 测试入库
python -m scripts.ingest --persist-dir ./data/chroma_db_test --limit 150

## 全量入库
python -m scripts.ingest --persist-dir ./data/chroma_db_bge

## 交互式提问
python -m scripts.evaluate --interactive

## 检索评估(生成对比表并导出-reports/)
python -m scripts.evaluate --eval --k 5

## API
- `GET /docs` — Swagger UI（FastAPI 自带）
- `POST /ask` 
### 请求示例
```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question": "如何执行泄漏测试"}'
```
## 测试
```bash
- 单元测试（TestClient，进程内，不启动服务）
```

## 结果
方法,Recall@5,MRR
向量,0.8824,0.8235
BM25,0.7059,0.6275
Hybrid,0.8824,0.7353
Hybrid+Rerank,0.8824,0.8235

以上结果由 `python -m scripts.evaluate --eval` 生成。
## 环境依赖
- Python 3.11
- Ollama（本地跑 bge-m3）
- DeepSeek API Key
依赖见 `requirements.txt`

## 已知限制
- 表格提取为组合方式(文本+表格).
- 正文和表格分块存储：正文切分后独立成块，表格单独成块。大多数问题能通过 Top-3 覆盖到两者，但如果问题同时需要“章节标题 + 表格内容”，而 Top-3 里只命中了其中一个块，LLM 可能无法完整回答。
- 只支持单 PDF：目前针对 IRB 6700 手册调优，换其他手册需要重新配置页脚清洗正则.
- tokenizers fork 警告：Cross-Encoder 加载时出现，不影响功能.

## License
MIT