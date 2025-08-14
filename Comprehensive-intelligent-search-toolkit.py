"""
title: 🔍 综合智能搜索工具集 - Kimi AI + Bocha + RAG优化 + LLM智能摘要 + 链接噪声治理 (完整修复版)
author: JiangNanGenius
Github: https://github.com/JiangNanGenius
description: 集成Kimi AI基础搜索、Bocha专业搜索、网页读取，支持LLM智能摘要提取、RAG向量化、语义重排序的智能搜索工具集，强化链接噪声治理和优雅回退，修复语法错误和分片重叠问题，实现并发LLM调用
required_open_webui_version: 0.4.0
requirements: openai>=1.0.0, requests, beautifulsoup4, numpy, aiohttp
version: 3.8.2
license: MIT
"""

import os
import requests
import json
import asyncio
import aiohttp
import numpy as np
import hashlib
from pydantic import BaseModel, Field
from typing import Callable, Any, Optional, List, Dict
from urllib.parse import urlparse
import unicodedata
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime
import traceback
from uuid import uuid4
from collections import defaultdict


class Tools:
    class Valves(BaseModel):
        # 调试配置
        DEBUG_MODE: bool = Field(
            default=False, description="🐛 调试模式开关 - 显示详细日志和错误信息"
        )

        # Kimi AI 配置
        MOONSHOT_API_KEY: str = Field(
            default="", description="🌙 Moonshot API密钥 (用于Kimi AI基础搜索功能)"
        )
        MOONSHOT_BASE_URL: str = Field(
            default="https://api.moonshot.cn/v1", description="🌙 Moonshot API基础URL"
        )
        KIMI_MODEL: str = Field(
            default="moonshot-v1-auto", description="🤖 Kimi使用的模型"
        )
        KIMI_TEMPERATURE: float = Field(default=0.3, description="🌡️ Kimi模型温度参数")

        # Bocha 配置
        BOCHA_API_KEY: str = Field(
            default="YOUR_BOCHA_API_KEY",
            description="🔑 Bocha AI API密钥 (用于专业中文搜索和AI搜索)",
        )
        LANGSEARCH_API_KEY: str = Field(
            default="YOUR_LANGSEARCH_API_KEY",
            description="🗝️ LangSearch API密钥 (用于专业英文搜索)",
        )

        # 豆包向量化配置
        ARK_API_KEY: str = Field(
            default="YOUR_ARK_API_KEY",
            description="🎯 豆包ARK API密钥 (用于文本向量化)",
        )
        EMBEDDING_BASE_URL: str = Field(
            default="https://ark.cn-beijing.volces.com/api/v3",
            description="🎯 豆包向量化API基础URL",
        )
        EMBEDDING_MODEL: str = Field(
            default="doubao-embedding-large-text-250515",
            description="📊 向量化模型名称",
        )

        # LLM智能摘要配置
        ENABLE_SMART_SUMMARY: bool = Field(
            default=True, description="🧠 是否启用LLM智能摘要提取"
        )
        SUMMARY_MIN_CHARS: int = Field(default=300, description="📏 单条摘要最小字符数")
        SUMMARY_MAX_CHARS: int = Field(default=800, description="📏 单条摘要最大字符数")
        SUMMARY_TEMPERATURE: float = Field(
            default=0.2, description="🌡️ 摘要提取温度参数"
        )
        SUMMARY_COUNT_PER_PAGE: int = Field(default=20, description="📄 每页摘要数量")

        # 语义安全分片配置
        TARGET_CHUNK_CHARS: int = Field(
            default=2800, description="🎯 目标分片大小（字符）"
        )
        MAX_CHUNK_CHARS: int = Field(default=3500, description="⛔ 单片最大字符")
        OVERLAP_SENTENCES: int = Field(default=3, description="🔗 相邻分片的句子重叠数")
        MAX_TOTAL_CHUNKS: int = Field(default=32, description="📚 每页最多处理的分片数")

        # 并发控制配置
        LLM_MAX_CONCURRENCY: int = Field(
            default=3, description="🕊️ 分片并发摘要的最大并发数"
        )
        LLM_RETRIES: int = Field(default=2, description="🔁 LLM调用失败的重试次数")
        LLM_BACKOFF_BASE_SEC: float = Field(
            default=0.8, description="⏳ 重试退避基数（秒）"
        )

        # 分片保护策略
        PRESERVE_TABLES: bool = Field(default=True, description="📊 分片时整块保留表格")
        PRESERVE_CODEBLOCKS: bool = Field(
            default=True, description="🧩 分片时整块保留代码块"
        )
        PRESERVE_LINKS: bool = Field(
            default=True, description="🔗 分片时保证链接不被打散"
        )
        DENOISE_LINK_SECTIONS: bool = Field(
            default=True, description="🧹 移除纯链接/导航段落"
        )

        # 摘要策略
        MAP_SUMMARY_PER_CHUNK: int = Field(
            default=7, description="🧭 map 阶段：每个分片要提取的摘要条数"
        )
        REDUCE_SUMMARY_LIMIT: int = Field(
            default=20, description="🧰 reduce 阶段：整页保留的摘要条数上限"
        )
        ENABLE_DETAILED_EXTRACTION: bool = Field(
            default=True, description="📝 是否启用详细信息提取模式"
        )
        ENCOURAGE_COMPREHENSIVE: bool = Field(
            default=True, description="🎯 是否鼓励全面覆盖各个方面"
        )

        # LLM配置
        SEGMENTER_API_KEY: str = Field(
            default="", description="🔧 LLM API密钥 (留空则使用Moonshot密钥)"
        )
        SEGMENTER_BASE_URL: str = Field(
            default="", description="🔧 LLM API基础URL (留空则使用Moonshot URL)"
        )
        SEGMENTER_MODEL: str = Field(
            default="moonshot-v1-auto", description="🤖 LLM使用的模型"
        )
        SEGMENTER_TEMPERATURE: float = Field(default=0.1, description="🌡️ LLM温度参数")

        # 评分权重配置
        ENABLE_ANSWERABILITY: bool = Field(
            default=True, description="💡 是否启用可回答性评分"
        )
        ANSWERABILITY_WEIGHT: float = Field(
            default=0.25, description="⚖️ 可回答性分数权重"
        )
        SUMMARY_RELEVANCE_WEIGHT: float = Field(
            default=0.35, description="🎯 LLM摘要相关度权重"
        )
        RERANK_WEIGHT: float = Field(default=0.4, description="⚖️ 重排序分数权重")
        RAG_WEIGHT: float = Field(default=0.3, description="⚖️ RAG相似度权重")

        # 搜索端点配置
        CHINESE_WEB_SEARCH_ENDPOINT: str = Field(
            default="https://api.bochaai.com/v1/web-search",
            description="🇨🇳 中文网页搜索API端点",
        )
        ENGLISH_WEB_SEARCH_ENDPOINT: str = Field(
            default="https://api.langsearch.com/v1/web-search",
            description="🌐 英文网页搜索API端点",
        )
        AI_SEARCH_ENDPOINT: str = Field(
            default="https://api.bochaai.com/v1/ai-search",
            description="🤖 AI智能搜索API端点",
        )
        RERANK_ENDPOINT: str = Field(
            default="https://api.bochaai.com/v1/rerank",
            description="🎯 语义重排序API端点",
        )

        # RAG和重排序配置
        ENABLE_RAG_ENHANCEMENT: bool = Field(
            default=True, description="🧠 是否启用RAG向量化优化"
        )
        ENABLE_SEMANTIC_RERANK: bool = Field(
            default=True, description="🎯 是否启用语义重排序"
        )
        RERANK_MODEL: str = Field(default="gte-rerank", description="🎯 重排序模型名称")
        SIMILARITY_THRESHOLD: float = Field(
            default=0.08, description="📊 RAG相似度阈值"
        )
        EMIT_ONLY_RAG_PASS: bool = Field(
            default=True, description="🎯 仅返回通过阈值的RAG结果"
        )
        RERANK_TOP_N: int = Field(default=25, description="🎯 重排序返回结果数量")

        # 内容返回控制
        RETURN_CONTENT_IN_RESULTS: bool = Field(
            default=True, description="📄 是否在结果JSON中携带content字段"
        )
        RETURN_CONTENT_MAX_CHARS: int = Field(
            default=-1, description="📏 返回content的最大字符数，<=0表示不截断"
        )
        CITATION_DOC_MAX_CHARS: int = Field(
            default=-1, description="📋 引用中文档最大字符数，<=0表示不截断"
        )
        CITATION_CHUNK_SIZE: int = Field(
            default=0, description="🔗 引用分片大小，<=0表示不分片"
        )
        UNIQUE_REFERENCE_NAMES: bool = Field(
            default=True, description="🎯 引用名唯一，避免UI合并/折叠"
        )
        PERSIST_CITATIONS: bool = Field(
            default=True, description="💾 多次调用时保留并重发历史引用"
        )
        PERSIST_CITATIONS_MAX: int = Field(
            default=100, description="📚 历史引用最多保存条数"
        )
        RAW_OUTPUT_FORMAT: str = Field(
            default="json", description="📄 raw输出格式: json或text"
        )

        # 通用配置
        CHINESE_SEARCH_COUNT: int = Field(default=15, description="🇨🇳 中文搜索结果数量")
        ENGLISH_SEARCH_COUNT: int = Field(default=15, description="🌐 英文搜索结果数量")
        AI_SEARCH_COUNT: int = Field(default=15, description="🤖 AI搜索结果数量")
        CITATION_LINKS: bool = Field(
            default=True, description="🔗 是否发送引用链接和元数据"
        )
        FRESHNESS: str = Field(
            default="noLimit",
            description="⏰ 搜索时间范围 (noLimit, oneDay, oneWeek, oneMonth, oneYear)",
        )
        MAX_RETRIES: int = Field(default=3, description="🔄 最大重试次数")

        # Jina网页读取配置
        JINA_API_KEY: str = Field(
            default="jina_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            description="🌐 Jina API密钥 (用于网页读取)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.kimi_client = None
        self.segmenter_client = None
        self.embedding_cache = {}
        self.citation = False
        self.run_seq = 0
        self.citations_history = []

    # ======================== Kimi AI 搜索 ========================
    async def kimi_ai_search(
        self,
        search_query: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌙 Kimi AI基础搜索"""

        # === 内嵌工具函数 ===
        def get_segmenter_client():
            api_key = self.valves.SEGMENTER_API_KEY or self.valves.MOONSHOT_API_KEY
            base_url = self.valves.SEGMENTER_BASE_URL or self.valves.MOONSHOT_BASE_URL
            if not api_key:
                raise ValueError("需要API密钥")
            if (
                self.segmenter_client is None
                or self.segmenter_client.api_key != api_key
                or getattr(self.segmenter_client, "base_url_stored", None) != base_url
            ):
                self.segmenter_client = OpenAI(base_url=base_url, api_key=api_key)
                self.segmenter_client.base_url_stored = base_url
            return self.segmenter_client

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            return text if (max_chars is None or max_chars <= 0) else text[:max_chars]

        def split_text_chunks(text: str, size: int) -> List[str]:
            if text is None:
                return [""]
            if size is None or size <= 0:
                return [text]
            return [text[i : i + size] for i in range(0, len(text), size)]

        async def emit_citation_data(r: Dict, __event_emitter__, run_id: str, idx: int):
            if not (__event_emitter__ and self.valves.CITATION_LINKS):
                return

            full_doc = r.get("content") or ""
            doc_for_emit = take_text(full_doc, self.valves.CITATION_DOC_MAX_CHARS)
            chunks = split_text_chunks(doc_for_emit, self.valves.CITATION_CHUNK_SIZE)

            base_title = (r.get("title") or "") or (r.get("url") or "Source")
            base_url = (r.get("url") or "").strip()

            for ci, chunk in enumerate(chunks, 1):
                if self.valves.UNIQUE_REFERENCE_NAMES:
                    src_name = f"{base_title} | {base_url} | {run_id}#{idx}-{ci}-{uuid4().hex[:6]}"
                else:
                    src_name = base_url or base_title

                payload = {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "title": base_title,
                                "date_accessed": datetime.now().isoformat(),
                            }
                        ],
                        "source": {
                            "name": src_name,
                            "url": base_url or "",
                            "type": r.get("source_type", "webpage"),
                        },
                    },
                }
                await __event_emitter__(payload)

                if self.valves.PERSIST_CITATIONS:
                    self.citations_history.append(payload)
                    if len(self.citations_history) > self.valves.PERSIST_CITATIONS_MAX:
                        self.citations_history = self.citations_history[
                            -self.valves.PERSIST_CITATIONS_MAX :
                        ]

        # 获取run_id并重放历史引用
        run_id = next_run_id("kimi")
        if self.valves.PERSIST_CITATIONS and __event_emitter__:
            for old in self.citations_history:
                await __event_emitter__(old)

        def debug_log(message: str, error=None):
            if self.valves.DEBUG_MODE:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG {timestamp}] {message}")
                if error:
                    print(f"[DEBUG ERROR] {str(error)}")
                    print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")

        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": status,
                            "description": description,
                            "done": done,
                            "action": f"kimi_search:{run_id}",
                        },
                    }
                )

        def parse_search_results(content: str):
            debug_log(f"解析搜索结果内容: {content[:200]}...")

            source_pattern = r"\[来源：(https?://[^\]]+)\]"
            sources = re.findall(source_pattern, content)

            reference_pattern = (
                r"参考网站链接：\s*\n((?:\d+\.\s*\[https?://[^\]]+\]\([^\)]+\)\s*\n?)+)"
            )
            reference_match = re.search(reference_pattern, content, re.MULTILINE)

            if reference_match:
                reference_links = reference_match.group(1)
                link_pattern = r"\[(https?://[^\]]+)\]\([^\)]+\)"
                additional_sources = re.findall(link_pattern, reference_links)
                sources.extend(additional_sources)

            unique_sources = list(set(sources))
            debug_log(f"找到 {len(unique_sources)} 个唯一来源")

            sections = re.split(r"\n\d+\.\s*\*\*([^*]+)\*\*：", content)
            search_results = []

            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    title = sections[i].strip()
                    content_part = sections[i + 1].strip()

                    part_urls = re.findall(source_pattern, content_part)
                    main_url = (
                        part_urls[0]
                        if part_urls
                        else (unique_sources[0] if unique_sources else "")
                    )

                    clean_content = re.sub(
                        r"\[来源：[^\]]+\]", "", content_part
                    ).strip()

                    search_results.append(
                        {
                            "content": clean_content,
                            "title": title,
                            "url": main_url,
                            "site_name": (
                                main_url.split("/")[2] if main_url else "Unknown"
                            ),
                            "date_published": datetime.now().strftime("%Y-%m-%d"),
                            "source_type": "Kimi AI基础搜索",
                        }
                    )

            if not search_results and unique_sources:
                search_results.append(
                    {
                        "content": re.sub(r"\[来源：[^\]]+\]", "", content).strip(),
                        "title": search_query,
                        "url": unique_sources[0],
                        "site_name": (
                            unique_sources[0].split("/")[2]
                            if unique_sources
                            else "Unknown"
                        ),
                        "date_published": datetime.now().strftime("%Y-%m-%d"),
                        "source_type": "Kimi AI基础搜索",
                    }
                )

            debug_log(f"解析完成，得到 {len(search_results)} 个结果")
            return search_results, unique_sources

        def get_kimi_client():
            if (
                self.kimi_client is None
                or self.kimi_client.api_key != self.valves.MOONSHOT_API_KEY
            ):
                if not self.valves.MOONSHOT_API_KEY:
                    raise ValueError("Moonshot API密钥是必需的")
                self.kimi_client = OpenAI(
                    base_url=self.valves.MOONSHOT_BASE_URL,
                    api_key=self.valves.MOONSHOT_API_KEY,
                )
            return self.kimi_client

        def chat_with_kimi(messages: list):
            client = get_kimi_client()
            completion = client.chat.completions.create(
                model=self.valves.KIMI_MODEL,
                messages=messages,
                temperature=self.valves.KIMI_TEMPERATURE,
            )
            return completion.choices[0]

        try:
            debug_log(f"开始Kimi AI搜索: {search_query}, 上下文: {context}")

            if context:
                enhanced_query = f"在'{context}'的背景下，搜索关于'{search_query}'的信息。请提供详细且有引用来源的回答。"
                await emit_status(
                    f"🌙 开始Kimi AI搜索: {search_query} (背景: {context})"
                )
            else:
                enhanced_query = (
                    f"搜索关于'{search_query}'的信息。请提供详细且有引用来源的回答。"
                )
                await emit_status(f"🌙 开始Kimi AI搜索: {search_query}")

            system_prompt = """你是Kimi AI，一个基础的搜索助手。
请按照以下格式要求回答：

**回答结构：**
1. **标题1**：内容描述。[来源：完整URL]
2. **标题2**：内容描述。[来源：完整URL]  
3. **标题3**：内容描述。[来源：完整URL]

**引用格式：**
- 在每个信息点后使用：[来源：完整URL]
- 在回答末尾列出所有参考网站链接

请基于搜索结果提供相关信息。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_query},
            ]

            retry_count = 0
            while retry_count < self.valves.MAX_RETRIES:
                try:
                    debug_log(f"发送Kimi请求，重试次数: {retry_count}")
                    choice = chat_with_kimi(messages)
                    await emit_status("✅ 搜索完成，正在处理结果...")

                    content = choice.message.content
                    debug_log(f"Kimi最终回答: {content[:500]}...")

                    search_results, sources = parse_search_results(content)

                    for idx, r in enumerate(search_results):
                        await emit_citation_data(r, __event_emitter__, run_id, idx)

                    results_data = []
                    for r in search_results:
                        result_item = {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": take_text(r.get("content", ""), 300),
                        }
                        if self.valves.RETURN_CONTENT_IN_RESULTS:
                            result_item["content"] = take_text(
                                r.get("content", ""),
                                self.valves.RETURN_CONTENT_MAX_CHARS,
                            )
                        results_data.append(result_item)

                    result = {
                        "summary": {
                            "total_results": len(search_results),
                            "total_sources": len(sources),
                            "search_query": search_query,
                            "context": context,
                            "search_type": "🌙 Kimi AI基础搜索",
                            "timestamp": datetime.now().isoformat(),
                        },
                        "results": results_data,
                    }

                    await emit_status(
                        "🎉 Kimi AI搜索完成！", status="complete", done=True
                    )
                    return json.dumps(result, ensure_ascii=False, indent=2)

                except Exception as e:
                    retry_count += 1
                    debug_log(
                        f"Kimi请求失败，重试 {retry_count}/{self.valves.MAX_RETRIES}", e
                    )
                    if retry_count < self.valves.MAX_RETRIES:
                        await emit_status(
                            f"⚠️ 重试 {retry_count}/{self.valves.MAX_RETRIES}: {str(e)}"
                        )
                        await asyncio.sleep(1)
                    else:
                        raise e

            raise Exception("搜索过程异常结束")

        except Exception as e:
            debug_log("Kimi AI搜索失败", e)
            await emit_status(
                f"❌ Kimi AI搜索失败: {str(e)}", status="error", done=True
            )

            error_result = {
                "search_results": [],
                "error": str(e),
                "summary": {
                    "total_results": 0,
                    "total_sources": 0,
                    "search_query": search_query,
                    "context": context,
                    "search_type": "🌙 Kimi AI基础搜索",
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                },
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

    # ======================== 专业中文搜索 ========================
    async def search_chinese_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🇨🇳 专业中文网页搜索工具"""

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            return text if (max_chars is None or max_chars <= 0) else text[:max_chars]

        def split_text_chunks(text: str, size: int) -> List[str]:
            if text is None:
                return [""]
            if size is None or size <= 0:
                return [text]
            return [text[i : i + size] for i in range(0, len(text), size)]

        async def emit_citation_data(r: Dict, __event_emitter__, run_id: str, idx: int):
            if not (__event_emitter__ and self.valves.CITATION_LINKS):
                return

            full_doc = r.get("content") or ""
            doc_for_emit = take_text(full_doc, self.valves.CITATION_DOC_MAX_CHARS)
            chunks = split_text_chunks(doc_for_emit, self.valves.CITATION_CHUNK_SIZE)

            base_title = (r.get("title") or "") or (r.get("url") or "Source")
            base_url = (r.get("url") or "").strip()

            for ci, chunk in enumerate(chunks, 1):
                if self.valves.UNIQUE_REFERENCE_NAMES:
                    src_name = f"{base_title} | {base_url} | {run_id}#{idx}-{ci}-{uuid4().hex[:6]}"
                else:
                    src_name = base_url or base_title

                payload = {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "title": base_title,
                                "date_accessed": datetime.now().isoformat(),
                            }
                        ],
                        "source": {
                            "name": src_name,
                            "url": base_url or "",
                            "type": r.get("source_type", "webpage"),
                        },
                    },
                }
                await __event_emitter__(payload)

                if self.valves.PERSIST_CITATIONS:
                    self.citations_history.append(payload)
                    if len(self.citations_history) > self.valves.PERSIST_CITATIONS_MAX:
                        self.citations_history = self.citations_history[
                            -self.valves.PERSIST_CITATIONS_MAX :
                        ]

        async def get_text_embedding(text: str) -> Optional[List[float]]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not self.valves.ARK_API_KEY:
                return None

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]

            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.ARK_API_KEY}",
                    "Content-Type": "application/json",
                }

                clean_text = text.strip()[:4000]
                if not clean_text:
                    return None

                payload = {
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": [clean_text],
                    "encoding_format": "float",
                }

                response = requests.post(
                    f"{self.valves.EMBEDDING_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0]["embedding"]
                    self.embedding_cache[text_hash] = embedding
                    return embedding
                else:
                    return None
            except Exception:
                return None

        def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
            try:
                v1 = np.array(vec1)
                v2 = np.array(vec2)
                return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            except Exception:
                return 0.0

        run_id = next_run_id("zh-web")
        if self.valves.PERSIST_CITATIONS and __event_emitter__:
            for old in self.citations_history:
                await __event_emitter__(old)

        def debug_log(message: str, error=None):
            if self.valves.DEBUG_MODE:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG {timestamp}] {message}")
                if error:
                    print(f"[DEBUG ERROR] {str(error)}")
                    print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")

        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": status,
                            "description": description,
                            "done": done,
                            "action": f"zh_search:{run_id}",
                        },
                    }
                )

        async def enhance_results_with_rag(results: List[Dict]) -> List[Dict]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not results:
                debug_log("RAG未启用或结果为空")
                return results

            try:
                await emit_status(f"🧠 正在进行RAG向量化优化 ({len(results)} 个结果)")
                debug_log(f"开始RAG优化，查询: {query}, 结果数: {len(results)}")

                query_embedding = await get_text_embedding(query)
                if not query_embedding:
                    debug_log("查询向量化失败，返回原结果")
                    return results

                enhanced_results = []
                for i, result in enumerate(results):
                    content = result.get("content", "")
                    if not content:
                        debug_log(f"结果 {i} 内容为空，跳过")
                        continue

                    content_embedding = await get_text_embedding(content)
                    if content_embedding:
                        similarity = calculate_similarity(
                            query_embedding, content_embedding
                        )
                        result["rag_similarity"] = similarity
                        result["rag_enhanced"] = True
                        debug_log(f"结果 {i} 相似度: {similarity:.3f}")
                        enhanced_results.append(result)
                    else:
                        result["rag_similarity"] = 0.0
                        result["rag_enhanced"] = False
                        enhanced_results.append(result)
                        debug_log(f"结果 {i} 向量化失败，保留原结果")

                enhanced_results.sort(
                    key=lambda x: x.get("rag_similarity", 0), reverse=True
                )
                debug_log(f"RAG优化完成，保留 {len(enhanced_results)} 个结果")
                await emit_status(f"✅ RAG优化完成")
                return enhanced_results

            except Exception as e:
                debug_log("RAG优化失败", e)
                return results

        async def rerank_results(results: List[Dict]) -> List[Dict]:
            if (
                not self.valves.ENABLE_SEMANTIC_RERANK
                or not results
                or not self.valves.BOCHA_API_KEY
            ):
                debug_log("语义重排序未启用或配置不完整")
                return results

            try:
                await emit_status(f"🎯 正在进行语义重排序 ({len(results)} 个结果)")
                debug_log(f"开始语义重排序，查询: {query}")

                documents = []
                for i, result in enumerate(results):
                    content = result.get("content", "")[:4000]
                    if content:
                        documents.append(content)

                if not documents:
                    debug_log("没有有效文档，跳过重排序")
                    return results

                headers = {
                    "Authorization": f"Bearer {self.valves.BOCHA_API_KEY}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": self.valves.RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": min(self.valves.RERANK_TOP_N, len(documents)),
                    "return_documents": False,
                }

                response = requests.post(
                    self.valves.RERANK_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                if "data" in data and "results" in data["data"]:
                    rerank_results_data = data["data"]["results"]
                    reranked_results = []

                    for rerank_item in rerank_results_data:
                        index = rerank_item.get("index", 0)
                        relevance_score = rerank_item.get("relevance_score", 0.0)
                        if index < len(results):
                            result = results[index].copy()
                            result["rerank_score"] = relevance_score
                            result["rerank_enhanced"] = True
                            reranked_results.append(result)

                    debug_log(f"重排序完成，返回 {len(reranked_results)} 个结果")
                    await emit_status(f"✅ 语义重排序完成")
                    return reranked_results
                else:
                    debug_log("重排序响应格式异常")
                    return results

            except Exception as e:
                debug_log("语义重排序失败", e)
                return results

        try:
            debug_log(f"开始中文网页搜索: {query}")
            await emit_status(f"🔍 正在进行专业中文网页搜索: {query}")

            headers = {
                "Authorization": f"Bearer {self.valves.BOCHA_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "query": query,
                "freshness": self.valves.FRESHNESS,
                "summary": True,
                "count": self.valves.CHINESE_SEARCH_COUNT,
            }

            await emit_status("⏳ 正在连接专业中文搜索服务器...")
            resp = requests.post(
                self.valves.CHINESE_WEB_SEARCH_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            source_context_list = []
            len_raw = 0

            if "data" in data and "webPages" in data["data"]:
                web_pages = data["data"]["webPages"]
                if "value" in web_pages and isinstance(web_pages["value"], list):
                    len_raw = len(web_pages["value"])
                    await emit_status(f"📄 正在处理 {len_raw} 个中文网页结果...")

                    for i, item in enumerate(web_pages["value"]):
                        url = item.get("url", "")
                        snippet = item.get("snippet", "")
                        summary = item.get("summary", "")
                        name = item.get("name", "")
                        site_name = item.get("siteName", "")
                        date_published = item.get("datePublished", "")

                        content = summary or snippet
                        if not content:
                            debug_log(f"中文搜索结果 {i} 内容为空，跳过")
                            continue

                        result_item = {
                            "content": content,
                            "title": name,
                            "url": url,
                            "site_name": site_name if site_name else "",
                            "date_published": date_published if date_published else "",
                            "source_type": "专业中文网页",
                        }
                        source_context_list.append(result_item)

            debug_log(f"原始中文搜索结果数: {len(source_context_list)}")

            if self.valves.ENABLE_RAG_ENHANCEMENT:
                source_context_list = await enhance_results_with_rag(
                    source_context_list
                )

            if self.valves.ENABLE_SEMANTIC_RERANK:
                source_context_list = await rerank_results(source_context_list)

            if self.valves.ENABLE_RAG_ENHANCEMENT and self.valves.EMIT_ONLY_RAG_PASS:
                _thr = float(self.valves.SIMILARITY_THRESHOLD)
                source_context_list = [
                    r
                    for r in source_context_list
                    if float(r.get("rag_similarity") or 0.0) >= _thr
                ]

            for idx, r in enumerate(source_context_list):
                await emit_citation_data(r, __event_emitter__, run_id, idx)

            await emit_status(
                status="complete",
                description=f"🎉 中文网页搜索完成，找到 {len(source_context_list)} 个结果",
                done=True,
            )

            results_data = []
            for r in source_context_list:
                result_item = {
                    "title": (r.get("title") or ""),
                    "url": r.get("url"),
                    "rag_similarity": float(r.get("rag_similarity") or 0.0),
                    "rerank_score": float(r.get("rerank_score") or 0.0),
                    "snippet": take_text(r.get("content", ""), 300),
                }
                if self.valves.RETURN_CONTENT_IN_RESULTS:
                    result_item["content"] = take_text(
                        r.get("content", ""), self.valves.RETURN_CONTENT_MAX_CHARS
                    )
                results_data.append(result_item)

            return json.dumps(
                {
                    "summary": {
                        "rag_threshold": self.valves.SIMILARITY_THRESHOLD,
                        "kept": len(source_context_list),
                        "total": len_raw,
                        "search_type": "🇨🇳 专业中文网页",
                    },
                    "results": results_data,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            debug_log("中文网页搜索失败", e)
            error_details = {
                "error": str(e),
                "type": "❌ 专业中文网页搜索错误",
                "debug_info": (
                    traceback.format_exc() if self.valves.DEBUG_MODE else None
                ),
            }
            await emit_status(
                status="error", description=f"❌ 搜索出错: {str(e)}", done=True
            )
            return json.dumps(error_details, ensure_ascii=False, indent=2)

    # ======================== 专业英文搜索 ========================
    async def search_english_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 专业英文网页搜索工具"""

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            return text if (max_chars is None or max_chars <= 0) else text[:max_chars]

        def split_text_chunks(text: str, size: int) -> List[str]:
            if text is None:
                return [""]
            if size is None or size <= 0:
                return [text]
            return [text[i : i + size] for i in range(0, len(text), size)]

        async def emit_citation_data(r: Dict, __event_emitter__, run_id: str, idx: int):
            if not (__event_emitter__ and self.valves.CITATION_LINKS):
                return

            full_doc = r.get("content") or ""
            doc_for_emit = take_text(full_doc, self.valves.CITATION_DOC_MAX_CHARS)
            chunks = split_text_chunks(doc_for_emit, self.valves.CITATION_CHUNK_SIZE)

            base_title = (r.get("title") or "") or (r.get("url") or "Source")
            base_url = (r.get("url") or "").strip()

            for ci, chunk in enumerate(chunks, 1):
                if self.valves.UNIQUE_REFERENCE_NAMES:
                    src_name = f"{base_title} | {base_url} | {run_id}#{idx}-{ci}-{uuid4().hex[:6]}"
                else:
                    src_name = base_url or base_title

                payload = {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "title": base_title,
                                "date_accessed": datetime.now().isoformat(),
                            }
                        ],
                        "source": {
                            "name": src_name,
                            "url": base_url or "",
                            "type": r.get("source_type", "webpage"),
                        },
                    },
                }
                await __event_emitter__(payload)

                if self.valves.PERSIST_CITATIONS:
                    self.citations_history.append(payload)
                    if len(self.citations_history) > self.valves.PERSIST_CITATIONS_MAX:
                        self.citations_history = self.citations_history[
                            -self.valves.PERSIST_CITATIONS_MAX :
                        ]

        async def get_text_embedding(text: str) -> Optional[List[float]]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not self.valves.ARK_API_KEY:
                return None

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]

            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.ARK_API_KEY}",
                    "Content-Type": "application/json",
                }

                clean_text = text.strip()[:4000]
                if not clean_text:
                    return None

                payload = {
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": [clean_text],
                    "encoding_format": "float",
                }

                response = requests.post(
                    f"{self.valves.EMBEDDING_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0]["embedding"]
                    self.embedding_cache[text_hash] = embedding
                    return embedding
                else:
                    return None
            except Exception:
                return None

        def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
            try:
                v1 = np.array(vec1)
                v2 = np.array(vec2)
                return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            except Exception:
                return 0.0

        run_id = next_run_id("en-web")
        if self.valves.PERSIST_CITATIONS and __event_emitter__:
            for old in self.citations_history:
                await __event_emitter__(old)

        def debug_log(message: str, error=None):
            if self.valves.DEBUG_MODE:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG {timestamp}] {message}")
                if error:
                    print(f"[DEBUG ERROR] {str(error)}")
                    print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")

        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": status,
                            "description": description,
                            "done": done,
                            "action": f"en_search:{run_id}",
                        },
                    }
                )

        async def enhance_results_with_rag(results: List[Dict]) -> List[Dict]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not results:
                return results

            try:
                await emit_status(f"🧠 RAG优化 ({len(results)} 个结果)")
                query_embedding = await get_text_embedding(query)
                if not query_embedding:
                    return results

                enhanced_results = []
                for i, result in enumerate(results):
                    content = result.get("content", "")
                    if not content:
                        continue

                    content_embedding = await get_text_embedding(content)
                    if content_embedding:
                        similarity = calculate_similarity(
                            query_embedding, content_embedding
                        )
                        result["rag_similarity"] = similarity
                        result["rag_enhanced"] = True
                        enhanced_results.append(result)
                    else:
                        result["rag_similarity"] = 0.0
                        result["rag_enhanced"] = False
                        enhanced_results.append(result)

                enhanced_results.sort(
                    key=lambda x: x.get("rag_similarity", 0), reverse=True
                )
                await emit_status(f"✅ RAG优化完成")
                return enhanced_results

            except Exception as e:
                debug_log("RAG优化失败", e)
                return results

        async def rerank_results(results: List[Dict]) -> List[Dict]:
            if (
                not self.valves.ENABLE_SEMANTIC_RERANK
                or not results
                or not self.valves.BOCHA_API_KEY
            ):
                return results

            try:
                await emit_status(f"🎯 语义重排序 ({len(results)} 个结果)")
                documents = [
                    result.get("content", "")[:4000]
                    for result in results
                    if result.get("content")
                ]

                if not documents:
                    return results

                headers = {
                    "Authorization": f"Bearer {self.valves.BOCHA_API_KEY}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": self.valves.RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": min(self.valves.RERANK_TOP_N, len(documents)),
                    "return_documents": False,
                }

                response = requests.post(
                    self.valves.RERANK_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                if "data" in data and "results" in data["data"]:
                    rerank_results_data = data["data"]["results"]
                    reranked_results = []

                    for rerank_item in rerank_results_data:
                        index = rerank_item.get("index", 0)
                        relevance_score = rerank_item.get("relevance_score", 0.0)
                        if index < len(results):
                            result = results[index].copy()
                            result["rerank_score"] = relevance_score
                            result["rerank_enhanced"] = True
                            reranked_results.append(result)

                    await emit_status(f"✅ 语义重排序完成")
                    return reranked_results
                else:
                    return results

            except Exception as e:
                debug_log("语义重排序失败", e)
                return results

        try:
            await emit_status(f"🔍 英文网页搜索: {query}")

            headers = {
                "Authorization": f"Bearer {self.valves.LANGSEARCH_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "query": query,
                "freshness": self.valves.FRESHNESS,
                "summary": True,
                "count": self.valves.ENGLISH_SEARCH_COUNT,
            }

            await emit_status("⏳ 连接英文搜索服务器...")
            resp = requests.post(
                self.valves.ENGLISH_WEB_SEARCH_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            source_context_list = []
            len_raw = 0

            if "data" in data and "webPages" in data["data"]:
                web_pages = data["data"]["webPages"]
                if "value" in web_pages and isinstance(web_pages["value"], list):
                    len_raw = len(web_pages["value"])
                    await emit_status(f"📄 处理 {len_raw} 个英文结果...")

                    for i, item in enumerate(web_pages["value"]):
                        content = item.get("summary", "") or item.get("snippet", "")
                        if not content:
                            continue

                        result_item = {
                            "content": content,
                            "title": item.get("name", ""),
                            "url": item.get("url", ""),
                            "site_name": item.get("siteName", ""),
                            "date_published": item.get("datePublished", ""),
                            "source_type": "专业英文网页",
                        }
                        source_context_list.append(result_item)

            if self.valves.ENABLE_RAG_ENHANCEMENT:
                source_context_list = await enhance_results_with_rag(
                    source_context_list
                )

            if self.valves.ENABLE_SEMANTIC_RERANK:
                source_context_list = await rerank_results(source_context_list)

            if self.valves.ENABLE_RAG_ENHANCEMENT and self.valves.EMIT_ONLY_RAG_PASS:
                _thr = float(self.valves.SIMILARITY_THRESHOLD)
                source_context_list = [
                    r
                    for r in source_context_list
                    if float(r.get("rag_similarity") or 0.0) >= _thr
                ]

            for idx, r in enumerate(source_context_list):
                await emit_citation_data(r, __event_emitter__, run_id, idx)

            await emit_status(
                status="complete",
                description=f"🎉 英文网页搜索完成，找到 {len(source_context_list)} 个结果",
                done=True,
            )

            results_data = []
            for r in source_context_list:
                result_item = {
                    "title": (r.get("title") or ""),
                    "url": r.get("url"),
                    "rag_similarity": float(r.get("rag_similarity") or 0.0),
                    "rerank_score": float(r.get("rerank_score") or 0.0),
                    "snippet": take_text(r.get("content", ""), 300),
                }
                if self.valves.RETURN_CONTENT_IN_RESULTS:
                    result_item["content"] = take_text(
                        r.get("content", ""), self.valves.RETURN_CONTENT_MAX_CHARS
                    )
                results_data.append(result_item)

            return json.dumps(
                {
                    "summary": {
                        "rag_threshold": self.valves.SIMILARITY_THRESHOLD,
                        "kept": len(source_context_list),
                        "total": len_raw,
                        "search_type": "🌐 专业英文网页",
                    },
                    "results": results_data,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            debug_log("英文网页搜索失败", e)
            error_details = {
                "error": str(e),
                "type": "❌ 英文网页搜索错误",
                "debug_info": (
                    traceback.format_exc() if self.valves.DEBUG_MODE else None
                ),
            }
            await emit_status(
                status="error", description=f"❌ 搜索出错: {str(e)}", done=True
            )
            return json.dumps(error_details, ensure_ascii=False, indent=2)

    # ======================== 智能网页读取功能（完整修复版） ========================
    async def web_scrape(
        self,
        urls: List[str],
        user_request: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 智能网页读取工具 (完整修复版)"""

        # === 内嵌语义安全分片工具函数 ===
        def _protect_blocks_and_links(text: str):
            """保护代码块、表格、链接"""
            holders = {"code": {}, "tables": {}, "md": {}, "url": {}}

            if self.valves.PRESERVE_CODEBLOCKS:
                code_pat = re.compile(r"```.*?```", re.S)

                def _code_sub(m):
                    key = f"⟦CODE{len(holders['code'])}⟧"
                    holders["code"][key] = m.group(0)
                    return key

                text = code_pat.sub(_code_sub, text)

            if self.valves.PRESERVE_TABLES:
                lines = text.splitlines()
                out, i = [], 0
                while i < len(lines):
                    line = lines[i]
                    if line.strip().startswith("|") and "|" in line:
                        start = i
                        i += 1
                        while i < len(lines) and (
                            lines[i].strip().startswith("|")
                            or re.match(r"^\s*[:\-\|\s]+$", lines[i])
                        ):
                            i += 1
                        block = "\n".join(lines[start:i])
                        key = f"⟦TBL{len(holders['tables'])}⟧"
                        holders["tables"][key] = block
                        out.append(key)
                    else:
                        out.append(line)
                        i += 1
                text = "\n".join(out)

            if self.valves.PRESERVE_LINKS:

                def _md_sub(m):
                    key = f"⟦MD{len(holders['md'])}⟧"
                    holders["md"][key] = m.group(0)
                    return key

                text = re.sub(r"\[[^\]]+\]\([^)]+\)", _md_sub, text)

                def _url_sub(m):
                    key = f"⟦URL{len(holders['url'])}⟧"
                    holders["url"][key] = m.group(0)
                    return key

                text = re.sub(r"https?://[^\s\)\]]+", _url_sub, text)

            return text, holders

        def _restore_placeholders(text: str, holders: dict) -> str:
            """恢复占位符"""
            for k, v in holders.get("url", {}).items():
                text = text.replace(k, v)
            for k, v in holders.get("md", {}).items():
                text = text.replace(k, v)
            for k, v in holders.get("tables", {}).items():
                text = text.replace(k, v)
            for k, v in holders.get("code", {}).items():
                text = text.replace(k, v)
            return text

        def _split_sentences_zh_en(text: str) -> List[str]:
            """中英混合句子切分"""
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = text.replace("\n\n", "⟦PARA⟧")
            text = re.sub(r"([。！？；…])", r"\1⟦SPLIT⟧", text)
            text = re.sub(r"([.!?;])(\s+)(?=[A-Z0-9\"'])", r"\1⟦SPLIT⟧", text)
            text = text.replace("⟦PARA⟧", "⟦SPLIT⟧")

            parts = [p.strip() for p in text.split("⟦SPLIT⟧") if p.strip()]

            if self.valves.DENOISE_LINK_SECTIONS:
                cleaned = []
                for s in parts:
                    tokens = s.split()
                    link_like = sum(
                        1
                        for t in tokens
                        if t.startswith("http")
                        or t.startswith("⟦URL")
                        or t.startswith("⟦MD")
                    )
                    if len(tokens) <= 4 and link_like >= max(2, int(len(tokens) * 0.8)):
                        continue
                    cleaned.append(s)
                parts = cleaned

            return parts

        def _pack_sentences_to_chunks(sentences: List[str]) -> List[dict]:
            """将句子打包成分片（修复重叠逻辑）"""
            tgt = max(800, int(self.valves.TARGET_CHUNK_CHARS))
            hard = max(tgt, int(self.valves.MAX_CHUNK_CHARS))
            ovl = max(0, int(self.valves.OVERLAP_SENTENCES))

            chunks = []
            i = 0

            while i < len(sentences) and len(chunks) < int(
                self.valves.MAX_TOTAL_CHUNKS
            ):
                buf, size, start = [], 0, i

                while i < len(sentences):
                    s = sentences[i]
                    s_len = len(s) + 1

                    if size + s_len > tgt:
                        if not buf:
                            s_cut = s[:hard]
                            buf.append(s_cut)
                            sentences[i] = s[hard:]
                        break

                    buf.append(s)
                    size += s_len
                    i += 1

                if buf:
                    text = " ".join(buf).strip()
                    end = start + len(buf) - 1
                    chunks.append({"text": text, "start_sent": start, "end_sent": end})

                    # 修复分片重叠逻辑：正确的回退
                    if i < len(sentences):  # 只有在还有剩余句子时才回退
                        i = max(end - ovl + 1, start + 1)  # 确保至少前进一个句子
                else:
                    i += 1

            return chunks

        def smart_segment_text(raw_text: str) -> List[dict]:
            """语义安全分片入口"""
            protected, holders = _protect_blocks_and_links(raw_text)
            sentences = _split_sentences_zh_en(protected)
            chunks = _pack_sentences_to_chunks(sentences)

            for c in chunks:
                c["text"] = _restore_placeholders(c["text"], holders)

            return chunks

        # === 其他工具函数 ===
        def get_segmenter_client():
            api_key = self.valves.SEGMENTER_API_KEY or self.valves.MOONSHOT_API_KEY
            base_url = self.valves.SEGMENTER_BASE_URL or self.valves.MOONSHOT_BASE_URL
            if not api_key:
                raise ValueError("LLM需要API密钥")
            if (
                self.segmenter_client is None
                or self.segmenter_client.api_key != api_key
                or getattr(self.segmenter_client, "base_url_stored", None) != base_url
            ):
                self.segmenter_client = OpenAI(base_url=base_url, api_key=api_key)
                self.segmenter_client.base_url_stored = base_url
            return self.segmenter_client

        async def llm_call(
            messages: list, temperature: float = None, max_tokens: int = 4000
        ) -> str:
            """调用LLM（修复版：重试+线程池）"""
            client = get_segmenter_client()
            temp = (
                temperature
                if temperature is not None
                else self.valves.SUMMARY_TEMPERATURE
            )

            last_err = None
            for attempt in range(self.valves.LLM_RETRIES + 1):
                try:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=self.valves.SEGMENTER_MODEL,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    last_err = e
                    if attempt < self.valves.LLM_RETRIES:
                        await asyncio.sleep(
                            self.valves.LLM_BACKOFF_BASE_SEC * (attempt + 1)
                        )

            raise Exception(f"LLM调用失败: {last_err}")

        def is_wikipedia(u: str) -> bool:
            try:
                return "wikipedia.org" in (urlparse(u).netloc or "").lower()
            except Exception:
                return False

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            return text if (max_chars is None or max_chars <= 0) else text[:max_chars]

        def split_text_chunks(text: str, size: int) -> List[str]:
            if text is None:
                return [""]
            if size is None or size <= 0:
                return [text]
            return [text[i : i + size] for i in range(0, len(text), size)]

        async def emit_citation_data(r: Dict, __event_emitter__, run_id: str, idx: int):
            if not (__event_emitter__ and self.valves.CITATION_LINKS):
                return

            full_doc = r.get("content") or ""
            doc_for_emit = take_text(full_doc, self.valves.CITATION_DOC_MAX_CHARS)
            chunks = split_text_chunks(doc_for_emit, self.valves.CITATION_CHUNK_SIZE)

            base_title = (r.get("title") or "") or (r.get("url") or "Source")
            base_url = (r.get("url") or "").strip()

            for ci, chunk in enumerate(chunks, 1):
                if self.valves.UNIQUE_REFERENCE_NAMES:
                    src_name = f"{base_title} | {base_url} | {run_id}#{idx}-{ci}-{uuid4().hex[:6]}"
                else:
                    src_name = base_url or base_title

                payload = {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "title": base_title,
                                "date_accessed": datetime.now().isoformat(),
                            }
                        ],
                        "source": {
                            "name": src_name,
                            "url": base_url or "",
                            "type": r.get("source_type", "webpage"),
                        },
                    },
                }
                await __event_emitter__(payload)

                if self.valves.PERSIST_CITATIONS:
                    self.citations_history.append(payload)
                    if len(self.citations_history) > self.valves.PERSIST_CITATIONS_MAX:
                        self.citations_history = self.citations_history[
                            -self.valves.PERSIST_CITATIONS_MAX :
                        ]

        # 核心功能：修复版Map-Reduce智能摘要
        async def extract_targeted_summaries_enhanced(
            content: str,
            user_request: str,
            url: str,
            page_title: str,
            max_summaries: int = None,
        ) -> List[Dict]:
            """修复版：并发map + 健壮JSON解析"""
            if max_summaries is None:
                max_summaries = int(
                    self.valves.REDUCE_SUMMARY_LIMIT
                    or self.valves.SUMMARY_COUNT_PER_PAGE
                )

            def cleanup(text: str) -> str:
                t = re.sub(r"\n{4,}", "\n\n", text)
                t = re.sub(r"[ \t]{3,}", " ", t)
                t = re.sub(r"\[\d+\]", "", t)
                return t.strip()

            cleaned = cleanup(content)
            if not cleaned:
                return []

            chunks = smart_segment_text(cleaned)
            if not chunks:
                return []

            if len(chunks) > int(self.valves.MAX_TOTAL_CHUNKS):
                chunks = chunks[: int(self.valves.MAX_TOTAL_CHUNKS)]

            debug_log(f"语义安全分片完成：{len(chunks)} 片")

            # 动态决定每片条数
            target_total = max_summaries * 2
            per_chunk = max(3, min(10, int(round(target_total / max(1, len(chunks))))))

            sem = asyncio.Semaphore(self.valves.LLM_MAX_CONCURRENCY)
            map_summaries: List[Dict] = []

            def _extract_json_array(text: str) -> List[dict]:
                """健壮的JSON数组提取"""
                if not text:
                    return []

                t = text.strip()
                if t.startswith("```"):
                    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.I | re.M).strip()

                try:
                    obj = json.loads(t)
                    return obj if isinstance(obj, list) else []
                except Exception:
                    s, e = t.find("["), t.rfind("]")
                    if s != -1 and e != -1 and e > s:
                        try:
                            obj = json.loads(t[s : e + 1])
                            return obj if isinstance(obj, list) else []
                        except Exception:
                            return []
                    return []

            async def _map_one(idx: int, c: dict):
                """单个分片的摘要提取"""
                if self.valves.ENABLE_DETAILED_EXTRACTION:
                    detail_instruction = """
**详细信息提取要求：**
- 提取具体的数字、日期、人名、地名、专业术语
- 保留重要的定义、解释、公式、概念
- 包含历史背景、发展过程、影响意义
- 捕捉不同观点、争议、研究现状
- 记录重要的引用、参考资料、相关链接"""
                else:
                    detail_instruction = ""

                if self.valves.ENCOURAGE_COMPREHENSIVE:
                    comprehensive_instruction = f"""
**全面覆盖策略：**
- 从多个角度分析内容（技术、历史、文化、社会等）
- 优先提取能直接回答用户问题的信息
- 包含背景知识和相关概念
- 提取{per_chunk}条不同方面的摘要，避免重复
- 确保信息完整性和准确性"""
                else:
                    comprehensive_instruction = f"提取{per_chunk}条高质量摘要"

                sys_prompt = f"""你是专业信息提取专家。请基于给定片段内容，围绕用户需求进行全面深度的信息提取，输出JSON数组格式。

{detail_instruction}

{comprehensive_instruction}

**输出格式要求：**
- 每条摘要 {self.valves.SUMMARY_MIN_CHARS}–{self.valves.SUMMARY_MAX_CHARS} 字符
- 字段：summary(详细摘要全文)、position(位置描述)、key_points(5-8个关键词)、covers_aspects(3-5个涵盖方面)、relevance(0-1)、importance(0-1)、details(重要细节数组)
- 若片段不相关，返回 []
- 禁止编造信息，严格基于原文
- 优先选择信息密度高的内容

仅输出JSON数组，不要其他内容。"""

                user_prompt = f"""用户需求：{user_request}
页面标题：{page_title}
分片 {idx+1}/{len(chunks)} 内容：
{c['text']}

请进行全面深度的信息提取（{per_chunk}条）："""

                try:
                    async with sem:
                        resp = await llm_call(
                            [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=self.valves.SUMMARY_TEMPERATURE,
                            max_tokens=4000,
                        )

                    arr = _extract_json_array(resp)
                    out = []
                    for item in arr:
                        if not isinstance(item, dict):
                            continue
                        s = (item.get("summary") or "").strip()
                        if len(s) < int(self.valves.SUMMARY_MIN_CHARS):
                            continue

                        out.append(
                            {
                                "content": s[: int(self.valves.SUMMARY_MAX_CHARS)],
                                "title": f"{page_title} · 详细摘要",
                                "url": url,
                                "relevance": float(item.get("relevance", 0.7)),
                                "importance": float(item.get("importance", 0.7)),
                                "position": item.get("position", f"分片{idx+1}"),
                                "key_points": item.get("key_points", []),
                                "covers_aspects": item.get("covers_aspects", []),
                                "details": item.get("details", []),
                                "extract_method": "concurrent_map",
                                "source_type": "LLM智能摘要（并发map）",
                                "chunk_index": idx,
                            }
                        )
                    return out
                except Exception as e:
                    debug_log(f"分片 {idx+1} 并发摘要失败：{e}")
                    return []

            # 并发执行所有分片的摘要提取
            tasks = [_map_one(idx, c) for idx, c in enumerate(chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, list):
                    map_summaries.extend(r)
                else:
                    debug_log("map 子任务异常", r)

            if not map_summaries:
                return []

            debug_log(f"并发map摘要完成：共 {len(map_summaries)} 条，进入 reduce 汇总")

            # reduce阶段
            draft_items = []
            for m in map_summaries:
                aspects = ", ".join(m.get("covers_aspects", []))
                draft_items.append(
                    f"- [{m.get('position','')}] {aspects}: {m['content']}"
                )

            draft_text = "\n".join(draft_items)[:30000]

            reduce_sys = f"""你是高级信息聚合器。给定若干"分片级详细摘要"，请进行智能整合：

**聚合策略：**
1) 合并高重复度条目，保留信息更完整/更具体的一条
2) 优先覆盖"用户问题的关键方面"，其次补全背景/脉络
3) 优先信息密度高、包含数字/定义/因果/对比的条目
4) 选出最有价值的 {max_summaries} 条，避免主题重复
5) 保持原文事实，不要编造；如不确定则丢弃

**输出要求：**
- JSON数组格式，字段：summary、position、key_points、covers_aspects、relevance、importance、details
- 每条摘要保持详细和完整
- 按重要性和相关性排序
- 确保不遗漏关键信息

禁止编造，必须基于所给摘要。仅输出JSON数组。"""

            try:
                reduce_resp = await llm_call(
                    [
                        {"role": "system", "content": reduce_sys},
                        {
                            "role": "user",
                            "content": f"用户需求：{user_request}\n分片摘要列表：\n{draft_text}",
                        },
                    ],
                    temperature=0.05,
                    max_tokens=5000,
                )

                reduced = _extract_json_array(reduce_resp)
                if reduced:
                    final = []
                    for it in reduced[:max_summaries]:
                        s = (it.get("summary") or "").strip()
                        if not s:
                            continue

                        final.append(
                            {
                                "content": s[: int(self.valves.SUMMARY_MAX_CHARS)],
                                "title": f"{page_title} · 综合摘要",
                                "url": url,
                                "relevance": float(it.get("relevance", 0.8)),
                                "importance": float(it.get("importance", 0.8)),
                                "position": it.get("position", ""),
                                "key_points": it.get("key_points", []),
                                "covers_aspects": it.get("covers_aspects", []),
                                "details": it.get("details", []),
                                "extract_method": "concurrent_reduce",
                                "source_type": "LLM智能摘要（并发reduce）",
                            }
                        )

                    if final:
                        debug_log(f"并发reduce汇总成功，最终 {len(final)} 条摘要")
                        return final
            except Exception as e:
                debug_log(f"并发reduce汇总失败，回退 map 直出：{e}")

            # 回退策略
            map_summaries.sort(
                key=lambda x: (x.get("relevance", 0) + x.get("importance", 0)) / 2,
                reverse=True,
            )
            debug_log(f"回退到并发map摘要前 {max_summaries} 条")
            return map_summaries[:max_summaries]

        # RAG函数
        async def batch_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
            if not texts:
                return []
            if not (self.valves.ENABLE_RAG_ENHANCEMENT and self.valves.ARK_API_KEY):
                return [None for _ in texts]

            headers = {
                "Authorization": f"Bearer {self.valves.ARK_API_KEY}",
                "Content-Type": "application/json",
            }

            try:
                payload = {
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": [t[:4000] for t in texts],
                    "encoding_format": "float",
                }

                resp = await asyncio.to_thread(
                    requests.post,
                    f"{self.valves.EMBEDDING_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                vecs = [d["embedding"] for d in data.get("data", [])]
                if len(vecs) != len(texts):
                    fallback = []
                    for t in texts:
                        v = await get_single_embedding(t)
                        fallback.append(v)
                    return fallback

                return vecs
            except Exception:
                out = []
                for t in texts:
                    v = await get_single_embedding(t)
                    out.append(v)
                return out

        async def get_single_embedding(text: str) -> Optional[List[float]]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not self.valves.ARK_API_KEY:
                return None

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]

            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.ARK_API_KEY}",
                    "Content-Type": "application/json",
                }

                clean_text = text.strip()[:4000]
                if not clean_text:
                    return None

                payload = {
                    "model": self.valves.EMBEDDING_MODEL,
                    "input": [clean_text],
                    "encoding_format": "float",
                }

                response = await asyncio.to_thread(
                    requests.post,
                    f"{self.valves.EMBEDDING_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0]["embedding"]
                    self.embedding_cache[text_hash] = embedding
                    return embedding
                else:
                    return None
            except Exception:
                return None

        def cos_similarity(a: List[float], b: List[float]) -> float:
            try:
                va = np.array(a)
                vb = np.array(b)
                return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            except Exception:
                return 0.0

        async def internal_answerability_evaluation(
            content: str, user_request: str
        ) -> float:
            """可回答性评估"""
            if not self.valves.ENABLE_ANSWERABILITY:
                return 0.6

            try:
                content = content[:3000] if len(content) > 3000 else content
                system_prompt = """你是文本相关性评估专家。评分标准（宽松模式）：
- 1.0: 完美回答问题
- 0.8: 大部分回答问题  
- 0.6: 有一定相关性
- 0.4: 间接相关
- 0.2: 相关性较低
- 0.0: 完全无关

仅返回数字。"""

                user_prompt = f"""用户问题: {user_request}
文本内容: {content}

可回答性分数:"""

                response = await llm_call(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.05,
                )

                number_match = re.search(r"0\.\d+|1\.0|0|1", response.strip())
                if number_match:
                    score = float(number_match.group())
                    return max(0.0, min(1.0, score))
                else:
                    return 0.6
            except Exception as e:
                return 0.6

        run_id = next_run_id("web-scrape")
        if self.valves.PERSIST_CITATIONS and __event_emitter__:
            for old in self.citations_history:
                await __event_emitter__(old)

        def debug_log(message: str, error=None):
            if self.valves.DEBUG_MODE:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG {timestamp}] {message}")
                if error:
                    print(f"[DEBUG ERROR] {str(error)}")
                    print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")

        async def emit_status(
            description: str, done: bool, action: str, urls_list: List[str]
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "done": done,
                            "action": f"{action}:{run_id}",
                            "description": description,
                            "urls": urls_list,
                        },
                    }
                )

        try:
            debug_log(f"开始智能网页读取，URL数量: {len(urls)}")
            await emit_status(
                f"🧠 正在进行并发智能摘要提取 - {len(urls)} 个网页",
                False,
                "web_search",
                urls,
            )

            async def process_url(url):
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    "X-No-Cache": "true",
                    "X-With-Links-Summary": "true",
                    "Authorization": f"Bearer {self.valves.JINA_API_KEY}",
                }

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            jina_url, headers=headers, timeout=90
                        ) as response:
                            response.raise_for_status()
                            content = await response.text()

                    if not content or content.strip() == "":
                        return {
                            "content": "",
                            "title": url,
                            "url": url,
                            "error": "返回内容为空",
                            "status": "empty",
                        }

                    debug_log(f"成功读取URL {url}，内容长度: {len(content)}")

                    return {
                        "content": content,
                        "title": f"网页内容 - {url.split('/')[2] if '/' in url else url}",
                        "url": url,
                        "site_name": url.split("/")[2] if "/" in url else url,
                        "date_published": datetime.now().strftime("%Y-%m-%d"),
                        "source_type": "网页读取",
                        "status": "success",
                    }
                except Exception as e:
                    error_message = f"读取网页 {url} 时出错: {str(e)}"
                    debug_log(f"处理URL失败: {url}", e)
                    return {
                        "content": "",
                        "title": url,
                        "url": url,
                        "error": error_message,
                        "status": "error",
                    }

            tasks = [process_url(url) for url in urls]
            results = await asyncio.gather(*tasks)

            successful_results = []
            error_results = []

            for result in results:
                if result.get("status") == "success" and result.get("content"):
                    successful_results.append(result)
                else:
                    error_results.append(result)

            debug_log(
                f"处理完成，成功: {len(successful_results)}, 失败: {len(error_results)}"
            )

            if not successful_results:
                return json.dumps(
                    {
                        "request": user_request,
                        "error": "所有网页读取都失败",
                        "stats": {"kept": 0, "failed": len(urls)},
                        "summaries": [],
                        "errors": error_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # 智能摘要提取流程
            if self.valves.ENABLE_SMART_SUMMARY:
                await emit_status(
                    f"🧠 对 {len(successful_results)} 个页面进行并发摘要提取",
                    False,
                    "smart_summary",
                    urls,
                )

                all_summaries = []
                for i, page in enumerate(successful_results):
                    content = page.get("content", "")
                    url = page.get("url", "")
                    title = page.get("title", "")

                    debug_log(f"为页面 {i+1}/{len(successful_results)} 提取摘要: {url}")

                    try:
                        summaries = await extract_targeted_summaries_enhanced(
                            content=content,
                            user_request=user_request,
                            url=url,
                            page_title=title,
                            max_summaries=int(self.valves.REDUCE_SUMMARY_LIMIT),
                        )

                        debug_log(f"页面 {url} 提取到 {len(summaries)} 条摘要")
                        all_summaries.extend(summaries)

                        await emit_status(
                            f"📄 已处理 {i+1}/{len(successful_results)} 个页面，累计 {len(all_summaries)} 条摘要",
                            False,
                            "smart_summary",
                            urls,
                        )

                    except Exception as e:
                        debug_log(f"页面 {url} 摘要提取失败: {e}")
                        # 回退处理
                        content_chunks = [
                            content[i : i + self.valves.SUMMARY_MAX_CHARS]
                            for i in range(
                                0, len(content), self.valves.SUMMARY_MAX_CHARS
                            )[:3]
                        ]

                        for j, chunk in enumerate(content_chunks):
                            basic_summary = {
                                "content": chunk,
                                "title": f"{title} · 基础摘要{j+1}",
                                "url": url,
                                "relevance": 0.4,
                                "importance": 0.4,
                                "position": f"摘要提取失败-片段{j+1}",
                                "key_points": [],
                                "covers_aspects": [],
                                "details": [],
                                "extract_method": "basic_fallback_concurrent",
                                "source_type": "基础摘要(回退)",
                            }
                            all_summaries.append(basic_summary)

                debug_log(f"并发智能摘要提取完成，总计 {len(all_summaries)} 条摘要")

                if not all_summaries:
                    # 最终回退
                    all_summaries = []
                    for page in successful_results:
                        content = page.get("content", "")
                        content_chunks = [
                            content[i : i + 2000]
                            for i in range(0, len(content), 1500)[:5]
                        ]

                        for j, chunk in enumerate(content_chunks):
                            all_summaries.append(
                                {
                                    "content": chunk,
                                    "title": f"{page.get('title', '')} · 原始片段{j+1}",
                                    "url": page.get("url", ""),
                                    "relevance": 0.5,
                                    "importance": 0.5,
                                    "position": f"原始内容片段{j+1}",
                                    "key_points": [],
                                    "covers_aspects": [],
                                    "details": [],
                                    "extract_method": "multi_segment_fallback",
                                    "source_type": "原始内容（多段）",
                                }
                            )

                # RAG处理
                if self.valves.ENABLE_RAG_ENHANCEMENT and all_summaries:
                    await emit_status(
                        f"🎯 RAG向量化 {len(all_summaries)} 条摘要",
                        False,
                        "rag_enhancement",
                        urls,
                    )

                    summary_texts = [s["content"] for s in all_summaries]
                    query_vec = (await batch_embeddings([user_request]))[0]
                    summary_vecs = await batch_embeddings(summary_texts)

                    for i, summary in enumerate(all_summaries):
                        if (
                            i < len(summary_vecs)
                            and summary_vecs[i] is not None
                            and query_vec is not None
                        ):
                            similarity = cos_similarity(query_vec, summary_vecs[i])
                            summary["rag_similarity"] = similarity
                        else:
                            summary["rag_similarity"] = summary.get("relevance", 0.6)

                    all_summaries.sort(
                        key=lambda x: x.get("rag_similarity", 0), reverse=True
                    )

                # 语义重排序
                if (
                    self.valves.ENABLE_SEMANTIC_RERANK
                    and self.valves.BOCHA_API_KEY
                    and all_summaries
                ):
                    await emit_status(
                        f"🎯 语义重排序 {len(all_summaries)} 条摘要",
                        False,
                        "rerank",
                        urls,
                    )

                    try:
                        headers = {
                            "Authorization": f"Bearer {self.valves.BOCHA_API_KEY}",
                            "Content-Type": "application/json",
                        }

                        documents = [s["content"][:4000] for s in all_summaries]
                        payload = {
                            "model": self.valves.RERANK_MODEL,
                            "query": user_request,
                            "documents": documents,
                            "top_n": min(self.valves.RERANK_TOP_N, len(documents)),
                            "return_documents": False,
                        }

                        resp = await asyncio.to_thread(
                            requests.post,
                            self.valves.RERANK_ENDPOINT,
                            headers=headers,
                            json=payload,
                            timeout=60,
                        )
                        resp.raise_for_status()
                        data = resp.json()

                        if "data" in data and "results" in data["data"]:
                            rerank_results_data = data["data"]["results"]
                            reranked_summaries = []

                            for rerank_item in rerank_results_data:
                                index = rerank_item.get("index", 0)
                                relevance_score = rerank_item.get(
                                    "relevance_score", 0.0
                                )

                                if 0 <= index < len(all_summaries):
                                    summary = all_summaries[index].copy()
                                    summary["rerank_score"] = relevance_score
                                    reranked_summaries.append(summary)

                            all_summaries = reranked_summaries
                    except Exception as e:
                        debug_log(f"语义重排序失败: {e}")

                # 可回答性评分
                if self.valves.ENABLE_ANSWERABILITY and all_summaries:
                    await emit_status(f"💡 可回答性评分", False, "answerability", urls)

                    for i, summary in enumerate(all_summaries):
                        try:
                            answerability = await internal_answerability_evaluation(
                                summary["content"], user_request
                            )
                            summary["answerability"] = answerability
                        except Exception as e:
                            summary["answerability"] = 0.6

                # 综合分数计算
                for summary in all_summaries:
                    rag_similarity = float(summary.get("rag_similarity", 0.0))
                    rerank_score = float(summary.get("rerank_score", 0.0))
                    answerability = float(summary.get("answerability", 0.6))
                    llm_relevance = float(summary.get("relevance", 0.6))

                    final_score = (
                        self.valves.RAG_WEIGHT * rag_similarity
                        + self.valves.RERANK_WEIGHT * rerank_score
                        + self.valves.ANSWERABILITY_WEIGHT * answerability
                        + self.valves.SUMMARY_RELEVANCE_WEIGHT * llm_relevance
                    )
                    summary["final_score"] = final_score

                all_summaries.sort(key=lambda x: x.get("final_score", 0), reverse=True)

                # 阈值过滤
                if (
                    self.valves.ENABLE_RAG_ENHANCEMENT
                    and self.valves.EMIT_ONLY_RAG_PASS
                ):
                    threshold = self.valves.SIMILARITY_THRESHOLD

                    if any(is_wikipedia(r["url"]) for r in successful_results):
                        threshold = max(0.03, threshold * 0.4)
                        debug_log(f"检测到维基百科，放宽阈值到: {threshold}")

                    filtered_summaries = [
                        s
                        for s in all_summaries
                        if s.get("rag_similarity", 0) >= threshold
                        or s.get("final_score", 0) >= 0.3
                    ]

                    debug_log(
                        f"阈值过滤：{len(all_summaries)} -> {len(filtered_summaries)} 条摘要"
                    )
                    all_summaries = filtered_summaries

                final_summaries = all_summaries[: self.valves.RERANK_TOP_N]
                debug_log(f"最终保留 {len(final_summaries)} 条摘要")

                # 发送引用
                for idx, summary in enumerate(final_summaries):
                    await emit_citation_data(summary, __event_emitter__, run_id, idx)

                await emit_status(
                    f"🎉 智能摘要完成！提取了 {len(final_summaries)} 条摘要",
                    True,
                    "web_search",
                    urls,
                )

                # 构建返回体
                results_data = []
                for summary in final_summaries:
                    item = {
                        "title": summary.get("title") or "",
                        "url": summary.get("url"),
                        "rag_similarity": float(summary.get("rag_similarity", 0.0)),
                        "rerank_score": float(summary.get("rerank_score", 0.0)),
                        "answerability": float(summary.get("answerability", 0.6)),
                        "final_score": float(summary.get("final_score", 0.0)),
                        "key_points": summary.get("key_points", []),
                        "covers_aspects": summary.get("covers_aspects", []),
                        "details": summary.get("details", []),
                        "extract_method": summary.get("extract_method", ""),
                        "snippet": take_text(summary.get("content", ""), 300),
                    }
                    if self.valves.RETURN_CONTENT_IN_RESULTS:
                        item["content"] = take_text(
                            summary.get("content", ""),
                            self.valves.RETURN_CONTENT_MAX_CHARS,
                        )
                    results_data.append(item)

                return json.dumps(
                    {
                        "request": user_request,
                        "stats": {
                            "pages_fetched": len(successful_results),
                            "summaries_final": len(final_summaries),
                            "version": "concurrent_fixed_v3.8.2",
                            "chunking_strategy": "句子/链接/表格感知（修复重叠）",
                            "summarization_strategy": "并发Map-Reduce",
                        },
                        "summaries": results_data,
                        "errors": error_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            else:
                # 智能摘要未启用
                for idx, r in enumerate(successful_results):
                    await emit_citation_data(r, __event_emitter__, run_id, idx)

                results_data = []
                for r in successful_results:
                    result_item = {
                        "title": (r.get("title") or ""),
                        "url": r.get("url"),
                        "snippet": take_text(r.get("content", ""), 300),
                    }
                    if self.valves.RETURN_CONTENT_IN_RESULTS:
                        result_item["content"] = take_text(
                            r.get("content", ""), self.valves.RETURN_CONTENT_MAX_CHARS
                        )
                    results_data.append(result_item)

                return json.dumps(
                    {
                        "request": user_request,
                        "stats": {
                            "kept": len(successful_results),
                            "failed": len(error_results),
                        },
                        "results": results_data,
                        "errors": error_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        except Exception as e:
            debug_log("智能网页读取失败", e)
            return json.dumps(
                {
                    "request": user_request,
                    "error": str(e),
                    "stats": {"kept": 0, "failed": len(urls)},
                    "summaries": [],
                    "errors": [{"url": url, "error": "处理失败"} for url in urls],
                },
                ensure_ascii=False,
                indent=2,
            )

    # ======================== Raw网页读取功能 ========================
    async def web_scrape_raw(
        self,
        urls: List[str],
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 Raw网页读取工具"""

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            return text if (max_chars is None or max_chars <= 0) else text[:max_chars]

        def split_text_chunks(text: str, size: int) -> List[str]:
            if text is None:
                return [""]
            if size is None or size <= 0:
                return [text]
            return [text[i : i + size] for i in range(0, len(text), size)]

        async def emit_citation_data(r: Dict, __event_emitter__, run_id: str, idx: int):
            if not (__event_emitter__ and self.valves.CITATION_LINKS):
                return

            full_doc = r.get("content") or ""
            doc_for_emit = take_text(full_doc, self.valves.CITATION_DOC_MAX_CHARS)
            chunks = split_text_chunks(doc_for_emit, self.valves.CITATION_CHUNK_SIZE)

            base_title = (r.get("title") or "") or (r.get("url") or "Source")
            base_url = (r.get("url") or "").strip()

            for ci, chunk in enumerate(chunks, 1):
                if self.valves.UNIQUE_REFERENCE_NAMES:
                    src_name = f"{base_title} | {base_url} | {run_id}#{idx}-{ci}-{uuid4().hex[:6]}"
                else:
                    src_name = base_url or base_title

                payload = {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "title": base_title,
                                "date_accessed": datetime.now().isoformat(),
                            }
                        ],
                        "source": {
                            "name": src_name,
                            "url": base_url or "",
                            "type": r.get("source_type", "webpage"),
                        },
                    },
                }
                await __event_emitter__(payload)

                if self.valves.PERSIST_CITATIONS:
                    self.citations_history.append(payload)
                    if len(self.citations_history) > self.valves.PERSIST_CITATIONS_MAX:
                        self.citations_history = self.citations_history[
                            -self.valves.PERSIST_CITATIONS_MAX :
                        ]

        run_id = next_run_id("web-scrape-raw")
        if self.valves.PERSIST_CITATIONS and __event_emitter__:
            for old in self.citations_history:
                await __event_emitter__(old)

        def debug_log(message: str, error=None):
            if self.valves.DEBUG_MODE:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG {timestamp}] {message}")
                if error:
                    print(f"[DEBUG ERROR] {str(error)}")
                    print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")

        async def emit_status(
            description: str, done: bool, action: str, urls_list: List[str]
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "done": done,
                            "action": f"{action}:{run_id}",
                            "description": description,
                            "urls": urls_list,
                        },
                    }
                )

        try:
            debug_log(f"开始Raw网页读取，URL数量: {len(urls)}")
            await emit_status(
                f"🌐 正在Raw读取 {len(urls)} 个网页", False, "web_search", urls
            )

            async def process_url(url):
                jina_url = f"https://r.jina.ai/{url}"
                headers = {
                    "X-No-Cache": "true",
                    "X-With-Images-Summary": "true",
                    "X-With-Links-Summary": "true",
                    "Authorization": f"Bearer {self.valves.JINA_API_KEY}",
                }

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            jina_url, headers=headers, timeout=60
                        ) as response:
                            response.raise_for_status()
                            content = await response.text()

                    if not content or content.strip() == "":
                        return {
                            "content": "",
                            "title": url,
                            "url": url,
                            "error": "返回内容为空",
                            "status": "empty",
                        }

                    debug_log(f"成功读取URL {url}，内容长度: {len(content)}")

                    return {
                        "content": content,
                        "title": f"网页内容 - {url.split('/')[2] if '/' in url else url}",
                        "url": url,
                        "status": "success",
                    }
                except Exception as e:
                    error_message = f"读取网页 {url} 时出错: {str(e)}"
                    debug_log(f"处理URL失败: {url}", e)
                    return {
                        "content": "",
                        "title": url,
                        "url": url,
                        "error": error_message,
                        "status": "error",
                    }

            tasks = [process_url(url) for url in urls]
            results = await asyncio.gather(*tasks)

            successful_results = []
            error_results = []

            for result in results:
                if result.get("status") == "success" and result.get("content"):
                    successful_results.append(result)
                else:
                    error_results.append(result)

            for idx, r in enumerate(successful_results):
                await emit_citation_data(r, __event_emitter__, run_id, idx)

            await emit_status(
                f"🎉 Raw读取完成，成功处理 {len(successful_results)} 个",
                True,
                "web_search",
                urls,
            )

            if self.valves.RAW_OUTPUT_FORMAT.lower() == "json":
                results_data = []
                for r in successful_results:
                    result_item = {
                        "title": (r.get("title") or ""),
                        "url": r.get("url"),
                        "content": take_text(
                            r.get("content", ""), self.valves.RETURN_CONTENT_MAX_CHARS
                        ),
                    }
                    results_data.append(result_item)

                return json.dumps(
                    {"results": results_data, "errors": error_results},
                    ensure_ascii=False,
                    indent=2,
                )
            else:
                final_results = []
                for result in successful_results:
                    content = take_text(
                        result.get("content", ""), self.valves.RETURN_CONTENT_MAX_CHARS
                    )
                    final_results.append(
                        f"""URL: {result['url']}
标题: {result.get('title', '')}
内容: {content}
"""
                    )

                for result in error_results:
                    final_results.append(
                        f"""URL: {result['url']}
错误: {result.get('error', '未知错误')}
"""
                    )

                final_result = "\n".join(final_results)
                if not final_result.strip():
                    final_result = "所有网页读取均失败。"

                result_text = f"""Raw网页读取结果:
📊 总URL数: {len(urls)}
✅ 成功读取: {len(successful_results)}
❌ 失败读取: {len(error_results)}

原始网页内容:
{final_result}"""
                return result_text

        except Exception as e:
            debug_log("Raw网页读取失败", e)
            if self.valves.RAW_OUTPUT_FORMAT.lower() == "json":
                return json.dumps(
                    {
                        "error": str(e),
                        "results": [],
                        "errors": [{"url": url, "error": "处理失败"} for url in urls],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            else:
                return f"""❌ Raw网页读取出现错误: {str(e)}
请检查网络连接和API配置。"""

    # ======================== AI智能搜索 ========================
    async def search_ai_intelligent(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🤖 高级AI智能搜索工具"""

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            return text if (max_chars is None or max_chars <= 0) else text[:max_chars]

        def split_text_chunks(text: str, size: int) -> List[str]:
            if text is None:
                return [""]
            if size is None or size <= 0:
                return [text]
            return [text[i : i + size] for i in range(0, len(text), size)]

        async def emit_citation_data(r: Dict, __event_emitter__, run_id: str, idx: int):
            if not (__event_emitter__ and self.valves.CITATION_LINKS):
                return

            full_doc = r.get("content") or ""
            doc_for_emit = take_text(full_doc, self.valves.CITATION_DOC_MAX_CHARS)
            chunks = split_text_chunks(doc_for_emit, self.valves.CITATION_CHUNK_SIZE)

            base_title = (r.get("title") or "") or (r.get("url") or "Source")
            base_url = (r.get("url") or "").strip()

            for ci, chunk in enumerate(chunks, 1):
                if self.valves.UNIQUE_REFERENCE_NAMES:
                    src_name = f"{base_title} | {base_url} | {run_id}#{idx}-{ci}-{uuid4().hex[:6]}"
                else:
                    src_name = base_url or base_title

                payload = {
                    "type": "citation",
                    "data": {
                        "document": [chunk],
                        "metadata": [
                            {
                                "title": base_title,
                                "date_accessed": datetime.now().isoformat(),
                            }
                        ],
                        "source": {
                            "name": src_name,
                            "url": base_url or "",
                            "type": r.get("source_type", "webpage"),
                        },
                    },
                }
                await __event_emitter__(payload)

                if self.valves.PERSIST_CITATIONS:
                    self.citations_history.append(payload)
                    if len(self.citations_history) > self.valves.PERSIST_CITATIONS_MAX:
                        self.citations_history = self.citations_history[
                            -self.valves.PERSIST_CITATIONS_MAX :
                        ]

        run_id = next_run_id("ai-search")
        if self.valves.PERSIST_CITATIONS and __event_emitter__:
            for old in self.citations_history:
                await __event_emitter__(old)

        def debug_log(message: str, error=None):
            if self.valves.DEBUG_MODE:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG {timestamp}] {message}")
                if error:
                    print(f"[DEBUG ERROR] {str(error)}")
                    print(f"[DEBUG TRACEBACK] {traceback.format_exc()}")

        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": status,
                            "description": description,
                            "done": done,
                            "action": f"ai_search:{run_id}",
                        },
                    }
                )

        try:
            debug_log(f"开始AI智能搜索: {query}")
            await emit_status(f"🤖 正在进行高级AI智能搜索: {query}")

            headers = {
                "Authorization": f"Bearer {self.valves.BOCHA_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "query": query,
                "freshness": self.valves.FRESHNESS,
                "answer": True,
                "stream": False,
                "count": self.valves.AI_SEARCH_COUNT,
            }

            await emit_status("⏳ 连接AI搜索服务器...")
            resp = requests.post(
                self.valves.AI_SEARCH_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            source_context_list = []
            ai_answers = []
            follow_up_questions = []

            await emit_status("🧠 AI正在分析搜索结果...")

            if "messages" in data:
                for msg in data["messages"]:
                    msg_role = msg.get("role", "")
                    msg_type = msg.get("type", "")
                    content_type = msg.get("content_type", "")
                    content = msg.get("content", "")

                    if (
                        msg_role == "assistant"
                        and msg_type == "source"
                        and content_type == "webpage"
                    ):
                        try:
                            content_obj = json.loads(content)
                            if "value" in content_obj and isinstance(
                                content_obj["value"], list
                            ):
                                await emit_status(
                                    f"📄 处理 {len(content_obj['value'])} 个AI搜索结果..."
                                )

                                for i, item in enumerate(content_obj["value"]):
                                    search_content = item.get(
                                        "summary", ""
                                    ) or item.get("snippet", "")
                                    if not search_content:
                                        continue

                                    result_item = {
                                        "content": search_content,
                                        "title": item.get("name", ""),
                                        "url": item.get("url", ""),
                                        "site_name": item.get("siteName", ""),
                                        "date_published": item.get("datePublished", ""),
                                        "source_type": "高级AI智能搜索",
                                    }
                                    source_context_list.append(result_item)
                        except json.JSONDecodeError as e:
                            debug_log("解析AI搜索结果JSON失败", e)

                    elif (
                        msg_role == "assistant"
                        and msg_type == "answer"
                        and content_type == "text"
                    ):
                        ai_answers.append(f"🤖 {content}")
                        await emit_status(f"✨ AI生成了第 {len(ai_answers)} 个回答...")

                    elif (
                        msg_role == "assistant"
                        and msg_type == "follow_up"
                        and content_type == "text"
                    ):
                        follow_up_questions.append(f"💭 {content}")
                        await emit_status(
                            f"💡 AI建议了第 {len(follow_up_questions)} 个追问..."
                        )

            # 发送引用
            for idx, r in enumerate(source_context_list):
                await emit_citation_data(r, __event_emitter__, run_id, idx)

            results_data = []
            for r in source_context_list:
                result_item = {
                    "title": (r.get("title") or ""),
                    "url": r.get("url"),
                    "snippet": take_text(r.get("content", ""), 300),
                }
                if self.valves.RETURN_CONTENT_IN_RESULTS:
                    result_item["content"] = take_text(
                        r.get("content", ""), self.valves.RETURN_CONTENT_MAX_CHARS
                    )
                results_data.append(result_item)

            result = {
                "search_results": results_data,
                "ai_answers": ai_answers,
                "follow_up_questions": follow_up_questions,
                "summary": {
                    "total_results": len(source_context_list),
                    "ai_answers_count": len(ai_answers),
                    "follow_up_count": len(follow_up_questions),
                    "search_type": "🤖 高级AI智能搜索",
                    "timestamp": datetime.now().isoformat(),
                },
            }

            await emit_status(
                status="complete",
                description=f"🎉 AI智能搜索完成！{len(source_context_list)} 个结果，{len(ai_answers)} 个AI答案",
                done=True,
            )

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            debug_log("AI智能搜索失败", e)
            error_details = {
                "error": str(e),
                "type": "❌ AI智能搜索错误",
                "debug_info": (
                    traceback.format_exc() if self.valves.DEBUG_MODE else None
                ),
            }
            await emit_status(
                status="error", description=f"❌ AI搜索出错: {str(e)}", done=True
            )
            return json.dumps(error_details, ensure_ascii=False, indent=2)


# ======================== Function类 - 暴露工具函数 ========================
class Function:
    def __init__(self):
        self.tools = Tools()

    # Kimi AI基础搜索
    async def kimi_ai_search(
        self,
        search_query: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌙 Kimi AI基础搜索 - 通用搜索功能"""
        return await self.tools.kimi_ai_search(search_query, context, __event_emitter__)

    # 专业中文网页搜索
    async def search_chinese_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🇨🇳 专业中文网页搜索工具"""
        return await self.tools.search_chinese_web(query, __event_emitter__)

    # 专业英文网页搜索
    async def search_english_web(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 专业英文网页搜索工具"""
        return await self.tools.search_english_web(query, __event_emitter__)

    # 智能网页读取（支持LLM摘要）
    async def web_scrape(
        self,
        urls: List[str],
        user_request: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 智能网页读取工具"""
        return await self.tools.web_scrape(urls, user_request, __event_emitter__)

    # Raw网页读取（不做处理）
    async def web_scrape_raw(
        self,
        urls: List[str],
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 Raw网页读取工具"""
        return await self.tools.web_scrape_raw(urls, __event_emitter__)

    # 高级AI智能搜索
    async def search_ai_intelligent(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🤖 高级AI智能搜索工具"""
        return await self.tools.search_ai_intelligent(query, __event_emitter__)
