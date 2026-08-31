# SupportFlow

SupportFlow 是一个范围明确的个人作品原型，面向英文账单与退款客服工单。系统支持四类结构化意图（`BILLING_QUESTION`、`REFUND_REQUEST`、`DUPLICATE_CHARGE` 和 `REFUND_STATUS`）以及五种模拟操作。三个受约束的 Agent 分别负责分诊、解决方案生成和风险审查，其输出包含提取事实、风险、理由、证据与不确定性；检索、策略校验、审批绑定和执行则保持确定性。

## 运行固定模型流程

```bash
uv sync --extra dev
LANGGRAPH_STRICT_MSGPACK=true uv run pytest -q
uv run python -m supportflow.cli demo-golden
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.cli demo-restart --runtime .supportflow/final-restart
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.eval.runner --dataset data/eval/tickets.jsonl --policies data/policies --output artifacts
LANGGRAPH_STRICT_MSGPACK=true uv run python -m supportflow.evidence export --runtime .supportflow/evidence-export --evaluation artifacts/eval-87eced57d8de89b24b3d6b1f470220761ffc443f48145bff011fc7735649e014.json --output artifacts/evidence-manifest-v1.json
uv run streamlit run src/supportflow/ui/app.py
```

`demo-golden` 使用固定本地模型和确定性的 token-hash embedding，完成重复扣费处理流程。该 embedding 是测试、命令行演示、评测和工作台的默认可复现实现。检索器根据分诊意图优先选择相关策略章节，融合精确余弦相似度与 BM25 排名，最多返回五条有效证据。只有当所选证据覆盖该意图要求的策略类型时，`EvidenceBundle.sufficient` 才为 `true`；否则系统会在生成解决方案前停止。

`SentenceTransformerEmbeddingProvider` 是可选的应用服务实现，只会从本地缓存加载 `sentence-transformers/all-MiniLM-L6-v2`（`local_files_only=True`），不会隐式下载模型。`demo-restart` 仅允许清理名为 `demo-restart` 或 `final-restart` 的目录，用于演示持久化暂停、跨进程审批和重复执行防护。

当调用方没有提供上游 `input_revision` 时，服务层和存储层会根据标准化后的工单内容生成修订标识，并排除 UI 接收时间戳。因此，即使分别构造内容相同的工作台提交，也会重新打开同一个等待中或已完成的运行。若调用方提供来源修订标识，系统会原样保留，并将其作为权威版本。

## 可选的真实模型适配器

系统只允许替换三个 Agent 背后的 `StructuredModel`。检索仍由 `RagRetriever` 完成；`PolicyGate`、审批哈希检查和 `DurableExecutor` 继续保持确定性与模拟执行边界。

```bash
uv sync --extra real-llm
export SUPPORTFLOW_LLM_MODEL="your-compatible-model"
export OPENAI_API_KEY="your-key"
# 可选：配置兼容 OpenAI API 的服务商
export OPENAI_BASE_URL="https://provider.example/v1"
uv run --extra real-llm python -m supportflow.cli demo-golden --model-adapter openai
```

适配器会把三个 Agent 的结果类型转换为服务商兼容的严格 JSON Schema：所有 DTO 字段均为必填、对象保持封闭、嵌套定义会被内联，最终 Schema 不包含引用、联合类型或判别器。解决方案阶段使用一个扁平且封闭的操作 DTO；解析完成后，只将五种白名单操作映射为领域模型。

模型超时或结构化输出格式错误时，系统只重试一次；SDK 自带重试被关闭，因此完整预算就是两次调用，并会在重启或重新进入节点后保持原子化记录。检索同样只重试一次。当进程在安全的模型或检索结果产生后、存储日志提交前中断时，待处理节点会在同一个持久化预算内重放；已经写入日志的结果只做状态协调，不会重复运行。每个模拟操作拥有三次持久化的瞬态失败尝试预算。

所有重试轨迹和终态错误只记录安全的错误类别，不保存服务商原始负载。非法操作不会重试。任何重试耗尽或非法结果都会停在 `NEEDS_ATTENTION`，无法绕过审批。服务商密钥和授权请求头不会进入日志。自动化测试仅使用本地替身，不发起网络请求。没有用户提供的凭据时，本仓库**不会运行真实模型演示**。

## 证据与边界

演示步骤见 [docs/demo-script.md](docs/demo-script.md)，完整证据链见 [docs/evidence-boundary.md](docs/evidence-boundary.md)，已提交的脱敏记录见 [artifacts/evidence-manifest-v1.json](artifacts/evidence-manifest-v1.json)。证据清单由一次全新的确定性公共服务运行生成。其 `source_revision` 是对可执行的 `src/supportflow` 源码树、冻结的 `data` 输入、`pyproject.toml` 和 `uv.lock` 计算得到的 SHA-256。生成产物被排除在摘要之外，因此可以精确绑定来源，同时避免“提交包含自身提交哈希”的循环依赖。

本项目使用模拟执行器，并未连接真实支付系统或客服系统。冻结评测采用固定的假模型输出，**不代表真实模型质量评测**。项目没有业务基线，也没有经过测量的业务提升指标；当前分数仅用于审计所提供的 30 条作品集工单，以及确定性的安全与恢复约束。
