# HyMed-Agent

基于千问 API、Neo4j、FAISS、LangGraph 和 SQLite 的医疗 Hybrid RAG 问答系统。

系统通过受控 ReAct 循环动态选择知识图谱和向量检索，并由 Harness
限制工具预算、重复调用和医疗输出风险。Skills 为不同医疗任务提供按需加载的执行策略。

> 医疗回答仅供知识参考，不替代医生诊断和治疗。

## 核心流程

```text
用户问题
  -> Harness 初始化预算、超时和执行轨迹
  -> 保存并加载相关长期记忆
  -> 结合最近对话改写省略式追问
  -> 选择相关医疗 Skills
  -> ReAct 决定下一步工具
      -> Neo4j：结构化医疗事实
      -> FAISS：长文本和相似医疗问答
  -> Evidence Checker 判断证据是否充分
      -> 不充分且预算允许：继续 ReAct
      -> 充分：生成回答
  -> Output Guardrail 检查诊断、剂量和过敏冲突
```

## 项目结构

```text
RAGQnASystem/
├── app.py                         # Streamlit 应用入口
├── medical_rag/                   # 在线应用代码
│   ├── core/                      # 配置、日志、计算设备
│   │   ├── config.py
│   │   ├── devices.py
│   │   └── logging.py
│   ├── clients/                   # 外部服务客户端
│   │   ├── qwen.py
│   │   └── neo4j.py
│   ├── retrieval/                 # Hybrid RAG 检索
│   │   ├── documents.py
│   │   ├── vector_store.py
│   │   ├── query_router.py
│   │   ├── graph_intents.py
│   │   └── tools.py
│   ├── workflow/
│   │   ├── hybrid_rag.py          # 旧固定流程，保留用于回归比较
│   │   └── react_rag.py           # 当前 ReAct LangGraph 工作流
│   ├── agent/                     # ReAct 决策协议
│   ├── harness/                   # 预算、Guardrail、证据检查
│   ├── skills/                    # 医疗任务 SKILL.md
│   ├── memory/
│   │   ├── store.py               # SQLite 持久化
│   │   └── service.py             # 提取和相关性过滤
│   ├── ner/                       # BERT + BiRNN 实体识别
│   └── ui/                        # Streamlit 页面与样式
├── scripts/                       # 离线构建和数据处理任务
│   ├── build_knowledge_graph.py
│   ├── build_ner_training_data.py
│   ├── build_vector_index.py
│   ├── export_huatuo_dataset.py
│   └── enrich_medical_dataset.py
├── tests/                         # 自动化回归测试
├── experiments/                   # 未接入在线应用的实验代码
│   ├── finetune/
│   └── ner_notebooks/
├── docs/assets/                   # README 图片
├── data/                          # 原始数据和训练语料
├── model/                         # 本地模型与 NER 权重
├── vector_index/                  # 生成的 FAISS 索引
├── tmp_data/                      # 用户状态和运行缓存
├── local_config.example.py        # 私密配置模板
└── requirements.txt
```

## 安装

```bash
conda create -n ragag python=3.10
conda activate ragag
pip install -r requirements.txt
```

## 配置千问

推荐在项目根目录创建 `local_config.py`：

```python
"""本机私密配置，不提交到 Git。"""

QWEN_API_KEY = "你的 DashScope API Key"
```

也可以使用环境变量：

```bash
export QWEN_API_KEY="你的 DashScope API Key"
```

读取优先级：

```text
QWEN_API_KEY / DASHSCOPE_API_KEY 环境变量
  > local_config.py
  > 空值
```

## 启动服务

### 1. 启动 Neo4j

如果使用项目内置 Neo4j：

```bash
./neo4j-community-5.26.26/bin/neo4j start
```

默认连接配置：

```text
Bolt: bolt://localhost:7687
Browser: http://localhost:7474
Database: neo4j
```

### 2. 启动应用

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

浏览器访问：

```text
http://127.0.0.1:8501
```

## 构建数据

这些操作属于离线任务，不需要在每次启动应用时执行。

### 构建 Neo4j 知识图谱

```bash
python -m scripts.build_knowledge_graph \
  --website bolt://localhost:7687 \
  --user neo4j \
  --password "你的密码" \
  --dbname neo4j
```

脚本会询问是否清空现有 Neo4j 数据，并生成 `data/ent_aug/` 实体词表。

### 生成 NER 训练数据

```bash
python -m scripts.build_ner_training_data
```

输出：

```text
data/ner_data_aug.txt
```

### 训练 NER

```bash
python -m medical_rag.ner.train --epochs 30 --batch-size 60
```

训练完成后进入交互测试：

```bash
python -m medical_rag.ner.train --interactive
```

### 导出 Huatuo QA 数据

```bash
python -m scripts.export_huatuo_dataset \
  --dataset-path /path/to/huatuo_encyclopedia_qa \
  --split train \
  --output /data0/grl_data/llm/rag/huatuo.jsonl \
  --limit 5000
```

### 构建 FAISS 索引

```bash
python -m scripts.build_vector_index \
  --corpus-path /data0/grl_data/llm/rag/huatuo.jsonl \
  --corpus-format jsonl_qa \
  --embedding-model /data0/grl_data/llm/rag/bge-m3 \
  --embedding-device cuda:5
```

## 配置项

配置定义位于 `medical_rag/core/config.py`，均支持环境变量覆盖。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j Bolt 地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户 |
| `NEO4J_PASSWORD` | 项目默认值 | 建议通过环境变量覆盖 |
| `NEO4J_DBNAME` | `neo4j` | Neo4j 数据库 |
| `QWEN_MODEL` | `qwen-turbo` | 千问模型 |
| `QWEN_TEMPERATURE` | `0.2` | 生成温度 |
| `NER_MODEL_NAME` | `model/chinese-roberta-wwm-ext` | NER 基础模型 |
| `NER_CHECKPOINT` | `best_roberta_rnn_model_ent_aug` | NER 权重名 |
| `COMPUTE_DEVICE` | `cuda:5` | NER 训练和推理设备 |
| `EMBEDDING_MODEL_NAME` | `/data0/grl_data/llm/rag/bge-m3` | Embedding 模型 |
| `EMBEDDING_DEVICE` | `cuda:5` | Embedding 设备 |
| `EMBEDDING_BATCH_SIZE` | `16` | Embedding 批大小 |
| `EMBEDDING_MAX_SEQ_LENGTH` | `1024` | Embedding 最大长度 |
| `VECTOR_TOP_K` | `5` | FAISS 返回数量 |
| `RERANKER_ENABLED` | `1` | 是否启用两阶段 Reranker |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Reranker 模型名或本地路径 |
| `RERANKER_DEVICE` | `cuda:5` | Reranker 推理设备 |
| `RERANKER_CANDIDATE_K` | `20` | FAISS 送入 Reranker 的候选数 |
| `RERANKER_BATCH_SIZE` | `8` | Reranker 推理批大小 |
| `RERANKER_MAX_LENGTH` | `1024` | Reranker 最大输入长度 |
| `RERANKER_SCORE_THRESHOLD` | `0.5` | Evidence Checker 的重排分数阈值 |
| `FAISS_INDEX_DIR` | `vector_index` | FAISS 索引目录 |
| `MEMORY_RECENT_TURNS` | `5` | 短期记忆对话轮数 |
| `MEMORY_DB_PATH` | `tmp_data/user_memory.db` | 长期记忆数据库 |
| `MEMORY_MAX_RELEVANT` | `6` | 单轮注入的长期记忆上限 |
| `AGENT_MAX_TOOL_CALLS` | `3` | 单轮最大工具调用次数 |
| `AGENT_MAX_REWRITES` | `1` | 单轮最大问题改写次数 |
| `AGENT_TIMEOUT_SECONDS` | `40` | Agent 运行超时 |
| `AGENT_MAX_SAME_TOOL_CALLS` | `1` | 相同工具与参数允许次数 |
| `AGENT_VECTOR_SCORE_THRESHOLD` | `0.55` | FAISS 证据相关性阈值 |
| `AGENT_MAX_TOP_K` | `8` | Harness 允许的最大 `top_k` |

## ReAct、Harness 与 Skills

当前在线流程使用 `medical_rag/workflow/react_rag.py`。

- **ReAct**：模型每轮只能选择一个检索工具或结束检索。
- **Harness**：验证工具白名单、参数、预算、重复调用和超时，并记录完整 trace。
- **Evidence Checker**：图谱只返回“药物治疗”等泛化结果时，要求继续 FAISS 检索。
- **Reranker**：FAISS 召回候选后使用 CrossEncoder 精排，再判断证据是否充分。
- **风险分级**：急症、具体用药和治疗问题无证据时拒绝自由补全；低风险健康教育问题允许受限兜底。
- **Output Guardrail**：阻止直接确诊、无证据剂量和明确过敏冲突；允许一次受证据约束的修复。
- **Skills**：从磁盘按需加载，不是额外工具，也不是额外 Agent。

内置 Skills：

```text
disease_treatment
medication_advice
prognosis_and_risk
```

每个 Skill 位于 `medical_rag/skills/<name>/SKILL.md`，包含适用任务、建议工具、
记忆要求和证据边界。管理员可在“Agent 决策过程”中查看 Skill 选择、ReAct 决策、
工具 Guardrail、Observation、Evidence Checker 和最终输出检查。

设备支持：

```text
auto / cpu / cuda / cuda:0 / cuda:1 ...
```

## 长期记忆

系统只保存用户明确陈述的四类信息：

- 病史
- 过敏
- 当前或长期用药
- 回答偏好

不会保存模型推测、检索结果、助手回答、疑问式诊断或普通一次性症状。

用户可以在侧边栏查看、逐条删除或清空自己的长期记忆。SQLite 数据按登录用户名隔离。

## 测试

```bash
python -m pytest -q
python -m compileall -q app.py medical_rag scripts tests
```

## Agent 真实评测

评测会调用千问 API、Neo4j、FAISS 和 NER 模型：

```bash
python scripts/evaluate_real_agent.py
```

可先运行少量样例检查环境：

```bash
python scripts/evaluate_real_agent.py --limit 3
```

也可以单独运行难例集：

```bash
python scripts/evaluate_real_agent.py --cases evaluation/cases/hard.jsonl
```

评测报告写入 `evaluation_reports/real/`，默认测试集位于
`evaluation/cases/real_medical_agent.jsonl`。测试集只包含问题与人工标注，
不包含预生成证据或回答。

## 实验代码

`experiments/` 中的微调脚本和 notebook 不参与当前线上流程，仅用于保留研究过程。运行这些实验前，请单独阅读对应目录中的 README 和依赖说明。
