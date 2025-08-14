# Comprehensive-intelligent-search-toolkit
# 🔍 综合智能搜索工具集

**Kimi AI + Bocha + RAG 优化 + LLM 智能摘要 + 链接噪声治理（完整修复版）**
**Kimi AI + Bocha + RAG + LLM Map-Reduce Summaries + Link Noise Mitigation (Fixed)**

> version: **3.8.2** · license: **MIT** · required\_open\_webui\_version: **0.4.0**

**作者 / Author:** JiangNanGenius
**GitHub:** [https://github.com/JiangNanGenius](https://github.com/JiangNanGenius)

本项目集成 Kimi AI 基础搜索、Bocha 专业搜索、网页读取、LLM 智能摘要（并发 Map‑Reduce）、RAG 向量化、语义重排序与链接噪声治理。修复并发/重试、分片重叠、健壮 JSON 解析等问题，支持优雅回退与并发 LLM 调用。

This project integrates Kimi AI basic search, Bocha/LangSearch professional search, web reading via Jina, LLM-powered concurrent Map‑Reduce summarization, RAG embeddings, semantic re-ranking, and link-noise mitigation. It fixes chunk overlap and robustness issues, supports graceful fallbacks and concurrent LLM calls.

---

## 目录 · Table of Contents

* [特性 Features](#特性-features)
* [架构概览 Architecture](#架构概览-architecture)
* [环境要求 Requirements](#环境要求-requirements)
* [安装 Installation](#安装-installation)
* [配置 Configuration](#配置-configuration)
* [快速上手 Quick Start](#快速上手-quick-start)
* [函数与示例 API & Examples](#函数与示例-api--examples)
* [事件流（状态/引用） Events (Status & Citations)](#事件流状态引用-events-status--citations)
* [返回格式与示例 Response Format](#返回格式与示例-response-format)
* [性能调优 Performance Tuning](#性能调优-performance-tuning)
* [隐私与安全 Privacy & Security](#隐私与安全-privacy--security)
* [常见问题 FAQ](#常见问题-faq)
* [变更日志 Changelog](#变更日志-changelog)
* [许可 License](#许可-license)

---

## 特性 Features

* **Kimi AI 基础搜索**：面向中文的快速检索与摘要产出
* **Bocha / LangSearch 专业搜索**：更强的中英文网页检索能力
* **网页读取（Jina Reader）**：并发抓取正文（带链接/图片摘要标记）
* **LLM 智能摘要（并发 Map‑Reduce）**：

  * 句子感知/链接感知/表格感知的**语义安全分片**
  * **并发 map**（每片提取多条详细摘要）+ **reduce**（去重聚合）
  * 健壮 JSON 提取与优雅回退（原文片段回退、多段直出等）
* **RAG 向量化**（Doubao ARK）：查询/摘要均可向量化，余弦相似度筛选
* **语义重排序**（Bocha Rerank）：多维打分融合
* **链接噪声治理**：移除纯链接段、可选整块保留代码/表格/链接
* **并发与重试**：LLM 并发、指数退避重试、超时控制
* **引用治理**：可选择持久化引用并在后续请求回放（可关闭）
* **事件回调**：`__event_emitter__` 输出**状态**与**引用**事件，便于前端/日志集成

---

## 架构概览 Architecture

```
User Query
   │
   ├─ Kimi AI basic search ───┐
   ├─ Bocha / LangSearch ─────┼─► Source list (URLs, snippets)
   └─ Given URL list ─────────┘
            │
            ├─ Jina Reader (fetch HTML→text)
            │
            ├─ Semantic Chunking (sentence/link/table/code aware)
            │
            ├─ LLM Map (concurrent) → detailed summaries per chunk
            ├─ LLM Reduce → dedup & merge top-N
            │
            ├─ RAG Embeddings (Doubao ARK) → cosine similarity
            ├─ Semantic Re-rank (Bocha) → relevance scoring
            └─ Score Fusion → Filter / Top-N
```

---

## 环境要求 Requirements

* **Python** ≥ 3.9
* **Open WebUI** ≥ 0.4.0（若需集成到 Open WebUI）
* 依赖包：`openai>=1.0.0, requests, beautifulsoup4, numpy, aiohttp, pydantic`

---

## 安装 Installation

```bash
pip install -U openai requests beautifulsoup4 numpy aiohttp pydantic
```

> 你可将本文件保存为模块（例如 `search_toolkit.py`）或按你的项目结构组织。

---

## 配置 Configuration

所有配置集中在 `Tools.Valves`（Pydantic 模型）中。你可以**在运行时动态设置**（推荐），无需改源码默认值。

### 必要的 API Key（按需启用）

| 用途                 | 字段                   | 说明                     |
| ------------------ | -------------------- | ---------------------- |
| Kimi / Moonshot    | `MOONSHOT_API_KEY`   | 供 Kimi 基础搜索与（可选）摘要模型调用 |
| Bocha 中文搜索/AI搜索/重排 | `BOCHA_API_KEY`      | 中文检索、AI搜索以及语义重排序       |
| LangSearch 英文搜索    | `LANGSEARCH_API_KEY` | 英文检索能力                 |
| Doubao ARK 向量化     | `ARK_API_KEY`        | RAG 向量生成               |
| Jina Reader        | `JINA_API_KEY`       | 抓取网页正文（`r.jina.ai`）    |

> **提示**：代码默认未读取环境变量。你可在使用示例中以 `os.getenv(...)` 方式装填到 `valves` 字段。

### 常用开关（节选）

* `ENABLE_SMART_SUMMARY=True`：启用 LLM 智能摘要（Map‑Reduce）
* `TARGET_CHUNK_CHARS=2800` / `OVERLAP_SENTENCES=3`：语义分片目标与重叠
* `LLM_MAX_CONCURRENCY=3` / `LLM_RETRIES=2`：并发与重试
* `ENABLE_RAG_ENHANCEMENT=True` / `SIMILARITY_THRESHOLD=0.08`
* `ENABLE_SEMANTIC_RERANK=True` / `RERANK_MODEL="gte-rerank"`
* `RETURN_CONTENT_IN_RESULTS=True` / `RETURN_CONTENT_MAX_CHARS=-1`（不截断）
* `PERSIST_CITATIONS=True`（持久化引用，后续请求回放）

> **更安全默认建议**（可选）：将 `RETURN_CONTENT_MAX_CHARS=800`、`CITATION_DOC_MAX_CHARS=800`、`CITATION_CHUNK_SIZE=400`、`PERSIST_CITATIONS=False`。

---

## 快速上手 Quick Start

```python
# quickstart.py
import os, asyncio
from search_toolkit import Function  # 假设文件名为 search_toolkit.py

async def main():
    f = Function()
    v = f.tools.valves

    # 装填密钥（推荐从环境变量读取）
    v.MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")
    v.BOCHA_API_KEY    = os.getenv("BOCHA_API_KEY", "")
    v.LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY", "")
    v.ARK_API_KEY      = os.getenv("ARK_API_KEY", "")
    v.JINA_API_KEY     = os.getenv("JINA_API_KEY", "")

    # 可选：更安全的输出限制
    v.RETURN_CONTENT_MAX_CHARS = 800
    v.CITATION_DOC_MAX_CHARS   = 800
    v.CITATION_CHUNK_SIZE      = 400
    v.PERSIST_CITATIONS        = False

    # 1) Kimi 基础搜索
    res_kimi = await f.kimi_ai_search("大语言模型最新研究进展", context="NLP综述")
    print("Kimi:", res_kimi[:500], "...\n")

    # 2) 中文/英文专业搜索
    res_zh = await f.search_chinese_web("国产多模态大模型 对比 评测 2025")
    res_en = await f.search_english_web("Retrieval-augmented generation best practices 2025")
    print("ZH:", res_zh[:300], "...\n")
    print("EN:", res_en[:300], "...\n")

    # 3) 网页读取 + 智能摘要（Map-Reduce）
    urls = [
        "https://arxiv.org/abs/2302.00000",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]
    res_read = await f.web_scrape(urls, user_request="总结RAG与智能体框架的关键要点与实践建议")
    print("Scrape:", res_read[:500], "...\n")

    # 4) AI 智能搜索（Bocha）
    res_ai = await f.search_ai_intelligent("国内外RAG评测基准与开源工具")
    print("AI Search:", res_ai[:500], "...\n")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 函数与示例 API & Examples

> 所有函数均为 **async**，返回 **JSON 字符串**（已包含结果与摘要/打分）。

### 1) `kimi_ai_search(search_query: str, context: str = "", __event_emitter__=None)`

* **作用**：Kimi AI 基础搜索，按要求的“标题+来源”格式解析结果
* **返回**：包含 `summary`、`results`（标题/URL/片段/可选content）

```python
await f.kimi_ai_search("强化学习 经典教材", context="入门路径")
```

### 2) `search_chinese_web(query: str, __event_emitter__=None)`

* **作用**：Bocha 中文网页搜索（可选 RAG、重排序）
* **返回**：`summary`（阈值、保留数）+ `results`（含 rag\_similarity / rerank\_score）

```python
await f.search_chinese_web("SFT与RLHF 区别 对比 详解")
```

### 3) `search_english_web(query: str, __event_emitter__=None)`

* **作用**：LangSearch 英文网页搜索（可选 RAG、重排序）

```python
await f.search_english_web("Mixture-of-Experts scaling laws 2025")
```

### 4) `web_scrape(urls: List[str], user_request: str, __event_emitter__=None)`

* **作用**：并发网页抓取 + 语义安全分片 + LLM Map‑Reduce 智能摘要
* **返回**：`stats`（pages\_fetched、summaries\_final等）+ `summaries`（多维打分）

```python
await f.web_scrape(
    ["https://arxiv.org/abs/2403.00000", "https://example.com/blog/rag"],
    user_request="提取关键定义、指标、最佳实践与坑点"
)
```

### 5) `web_scrape_raw(urls: List[str], __event_emitter__=None)`

* **作用**：直接抓取原始文本（不做摘要/打分）
* **返回**：每个 URL 的 `title`、`content` 原文片段

```python
await f.web_scrape_raw(["https://example.com/a", "https://example.com/b"])
```

### 6) `search_ai_intelligent(query: str, __event_emitter__=None)`

* **作用**：Bocha AI 搜索（带 AI 回答与追问），同时返回来源列表
* **返回**：`search_results`、`ai_answers`、`follow_up_questions`

```python
await f.search_ai_intelligent("RAG 在企业知识库落地的常见难题与解决方案")
```

---

## 事件流（状态/引用） Events (Status & Citations)

可传入 `__event_emitter__`（`Callable[[dict], Any]`）接收运行过程事件：

* `type: "status"`：阶段状态、进度、动作标识、（可选）URL 列表
* `type: "citation"`：引用片段（可控制是否持久化回放）

**示例：**

```python
async def emitter(evt: dict):
    t = evt.get("type")
    data = evt.get("data", {})
    if t == "status":
        print("[STATUS]", data.get("description"))
    elif t == "citation":
        print("[CITATION]", data["source"]["url"])

await f.web_scrape(
    urls=["https://example.com"],
    user_request="总结页面核心要点",
    __event_emitter__=emitter
)
```

> **安全提示**：事件里可能包含 URL 等元数据。若担心隐私，请在前端/日志侧做掩码或在代码层关闭上报。

---

## 返回格式与示例 Response Format

### 典型 `web_scrape` 返回（节选）

```json
{
  "request": "提取关键定义、指标、最佳实践与坑点",
  "stats": {
    "pages_fetched": 2,
    "summaries_final": 12,
    "version": "concurrent_fixed_v3.8.2",
    "chunking_strategy": "句子/链接/表格感知（修复重叠）",
    "summarization_strategy": "并发Map-Reduce"
  },
  "summaries": [
    {
      "title": "页面标题 · 综合摘要",
      "url": "https://example.com/post",
      "rag_similarity": 0.42,
      "rerank_score": 0.81,
      "answerability": 0.78,
      "final_score": 0.57,
      "key_points": ["定义", "评测指标", "风险"],
      "covers_aspects": ["技术", "工程实践", "评估"],
      "details": ["具体数字/日期/术语..."],
      "extract_method": "concurrent_reduce",
      "snippet": "（摘要前300字符）…",
      "content": "（完整摘要文本，长度受 RETURN_CONTENT_MAX_CHARS 控制）"
    }
  ],
  "errors": []
}
```

---

## 性能调优 Performance Tuning

* **并发**：`LLM_MAX_CONCURRENCY`（默认 3）
* **重试与退避**：`LLM_RETRIES`、`LLM_BACKOFF_BASE_SEC`
* **分片大小**：`TARGET_CHUNK_CHARS`、`MAX_CHUNK_CHARS`、`OVERLAP_SENTENCES`
* **RAG 阈值**：`SIMILARITY_THRESHOLD`（可根据数据杂讯程度调高/调低）
* **重排序数量**：`RERANK_TOP_N`
* **输出长度**：`RETURN_CONTENT_MAX_CHARS` 控制返回文本体量

---

## 隐私与安全 Privacy & Security

> 本仓库默认不包含真实密钥；你需要在运行时注入。下面是**强烈建议**的安全实践：

1. **密钥管理**：通过环境变量或密钥管理器装载，不要写入代码/仓库。
2. **外传控制**：开启 `PRIVACY_MODE`（可在你调用层自定义），对可能含 PII 的文本**脱敏**后再发送到第三方 API。
3. **内容回放限制**：将 `RETURN_CONTENT_MAX_CHARS`、`CITATION_DOC_MAX_CHARS` 设为有限值；`PERSIST_CITATIONS=False`。
4. **URL 白名单**：仅允许 `http/https`，拒绝 `file:`、`data:` 等，并拒绝内网网段/localhost（在你调用层做校验）。
5. **日志脱敏**：调试时对 token、`Bearer ...`、`jina_...` 等做正则替换为 `[REDACTED]`。
6. **合规**：遵循各第三方 API 的服务条款与使用政策。

---

## 常见问题 FAQ

**Q1：必须提供所有 API Key 吗？**
A：不需要。你可以只用部分能力；缺失的功能会优雅降级（如禁用 RAG / 重排序 / AI 搜索）。

**Q2：为什么要 async？**
A：因为抓取/LLM/向量化/重排序都涉及网络 IO，并发提升吞吐与延迟表现。

**Q3：摘要为什么有时直接回退到原文片段？**
A：当 LLM Map‑Reduce 或 JSON 解析失败时，自动触发回退策略，保证**有结果**而非报错。

**Q4：摘要里会不会泄露太多原文？**
A：由 `RETURN_CONTENT_MAX_CHARS`、`CITATION_DOC_MAX_CHARS` 与 `CITATION_CHUNK_SIZE` 控制；建议设置合理上限。

**Q5：如何集成到 Open WebUI？**
A：满足 `required_open_webui_version >= 0.4.0`，在你的函数调用层传入 `__event_emitter__`，即可将状态/引用事件用于前端流式呈现。

---

## 变更日志 Changelog

**v3.8.2**

* 修复：语义分片**重叠逻辑**与边界推进问题
* 修复：并发 Map‑Reduce 的**健壮 JSON 解析**与优雅回退
* 增强：**链接噪声治理**（移除纯链接段，保留表格/代码块）
* 增强：并发、重试、退避与超时处理
* 增强：RAG/重排序/可回答性综合打分，支持阈值过滤与 Wikipedia 放宽策略
* 增强：事件流与引用持久化（可配置）

---

## 许可 License

**MIT License**

```
Copyright (c) JiangNanGenius

Permission is hereby granted, free of charge, to any person obtaining a copy...
（MIT 许可全文）
```

---

### English README (Summary)

> The English content below mirrors the Chinese sections above.

#### Features

* Kimi basic search; Bocha/LangSearch pro search
* Web reading via Jina
* LLM concurrent Map‑Reduce summarization with semantic chunking (sentence/link/table/code aware)
* RAG embeddings (Doubao ARK), semantic re-ranking (Bocha)
* Link-noise mitigation, graceful fallbacks, retries/backoff
* Status & citation events for UI integration

#### Architecture

(See the diagram above.)

#### Requirements

* Python ≥ 3.9; Open WebUI ≥ 0.4.0 (optional)
* `openai>=1.0.0, requests, beautifulsoup4, numpy, aiohttp, pydantic`

#### Installation

```bash
pip install -U openai requests beautifulsoup4 numpy aiohttp pydantic
```

#### Configuration

Populate `Tools.Valves` fields at runtime (recommended). API keys (as needed): `MOONSHOT_API_KEY`, `BOCHA_API_KEY`, `LANGSEARCH_API_KEY`, `ARK_API_KEY`, `JINA_API_KEY`.
Suggested safer defaults: `RETURN_CONTENT_MAX_CHARS=800`, `CITATION_DOC_MAX_CHARS=800`, `CITATION_CHUNK_SIZE=400`, `PERSIST_CITATIONS=False`.

#### Quick Start

```python
import os, asyncio
from search_toolkit import Function

async def main():
    f = Function(); v = f.tools.valves
    v.MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY","")
    v.BOCHA_API_KEY    = os.getenv("BOCHA_API_KEY","")
    v.LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY","")
    v.ARK_API_KEY      = os.getenv("ARK_API_KEY","")
    v.JINA_API_KEY     = os.getenv("JINA_API_KEY","")

    print(await f.kimi_ai_search("Latest LLM research", context="NLP"))
    print(await f.search_chinese_web("国产多模态大模型 评测 2025"))
    print(await f.search_english_web("RAG best practices 2025"))
    print(await f.web_scrape(["https://arxiv.org/abs/2302.00000"], "Summarize key ideas"))
    print(await f.search_ai_intelligent("RAG evaluation benchmarks"))
asyncio.run(main())
```

#### API & Examples

* `kimi_ai_search(query, context="", __event_emitter__=None)`
* `search_chinese_web(query, __event_emitter__=None)`
* `search_english_web(query, __event_emitter__=None)`
* `web_scrape(urls, user_request, __event_emitter__=None)`
* `web_scrape_raw(urls, __event_emitter__=None)`
* `search_ai_intelligent(query, __event_emitter__=None)`

#### Events

Provide `__event_emitter__` to receive `"status"` and `"citation"` events for UI logs/streaming.

#### Response

JSON strings including summaries, scores (rag\_similarity, rerank\_score, answerability), and stats.

#### Performance

Tune: `LLM_MAX_CONCURRENCY`, `LLM_RETRIES`, `TARGET_CHUNK_CHARS`, `SIMILARITY_THRESHOLD`, `RERANK_TOP_N`, `RETURN_CONTENT_MAX_CHARS`.

#### Privacy & Security

Manage secrets via environment variables; limit content lengths; disable citation persistence if needed; whitelist URLs; redact logs.

#### Changelog

**v3.8.2** — fixed chunk overlap, robust Map‑Reduce JSON parsing, improved noise mitigation, concurrency, scoring fusion, and event handling.

#### License

MIT

---

> 需要把 README 拆分为独立的 `README_zh.md` 和 `README_en.md` 版本、或需要我根据你的仓库实际文件名改示例导入路径，告诉我即可，我直接整理好给你。
