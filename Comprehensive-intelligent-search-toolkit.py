"""
title: 🔍 综合智能搜索工具集 - Kimi AI + Bocha + RAG优化 + LLM智能摘要 + 链接噪声治理 (完整修复版)
author: JiangNanGenius
Github: https://github.com/JiangNanGenius
description: 集成Kimi AI基础搜索、Bocha专业搜索、网页读取，支持LLM智能摘要提取、RAG向量化、语义重排序的智能搜索工具集，强化链接噪声治理和优雅回退，修复语法错误和分片重叠问题，实现并发LLM调用
required_open_webui_version: 0.4.0
requirements: openai>=1.0.0, requests, beautifulsoup4, numpy, aiohttp
version: 3.9.3
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
        SUMMARY_MIN_CHARS: int = Field(
            default=200, description="📏 单条摘要最小字符数（给LLM参考）"
        )
        SUMMARY_MAX_CHARS: int = Field(
            default=800, description="📏 单条摘要最大字符数（给LLM参考）"
        )
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
            default=5, description="🕊️ 分片并发摘要的最大并发数"
        )
        LLM_RETRIES: int = Field(default=2, description="🔁 LLM调用失败的重试次数")
        LLM_BACKOFF_BASE_SEC: float = Field(
            default=1.2, description="⏳ 重试退避基数（秒）"
        )
        LLM_REQUEST_TIMEOUT_SEC: float = Field(
            default=45.0, description="⏱️ 单次LLM调用超时时间（秒）"
        )

        # 分片保护策略
        PRESERVE_TABLES: bool = Field(
            default=False, description="📊 分片时整块保留表格"
        )
        FLATTEN_TABLES: bool = Field(
            default=True, description="📋 将Markdown表格转为条目列表，便于模型提炼"
        )
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
            default=3, description="🧭 map 阶段：每个分片要提取的摘要条数"
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

        # 评分权重配置 - 简化为只有RAG和rerank
        RERANK_WEIGHT: float = Field(default=0.6, description="⚖️ 重排序分数权重")
        RAG_WEIGHT: float = Field(default=0.4, description="⚖️ RAG相似度权重")

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

        # 内容返回控制 - 简化为只返回摘要
        RETURN_CONTENT_IN_RESULTS: bool = Field(
            default=False, description="📄 是否在结果JSON中携带content字段"
        )
        RETURN_CONTENT_MAX_CHARS: int = Field(
            default=-1, description="📏 返回content的最大字符数，<=0表示不截断"
        )
        CITATION_DOC_MAX_CHARS: int = Field(
            default=6400, description="📋 引用中文档最大字符数，<=0表示不截断"
        )
        CITATION_CHUNK_SIZE: int = Field(
            default=1600, description="🔗 引用分片大小，<=0表示不分片"
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

        # Kimi联网搜索配置
        KIMI_FORCE_SEARCH: bool = Field(
            default=True, description="🌙 强制Kimi进行联网搜索"
        )
        KIMI_SEARCH_MAX_RETRIES: int = Field(
            default=3, description="🔄 Kimi联网搜索最大重试次数"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.kimi_client = None
        self.segmenter_client = None
        self.embedding_cache = {}
        self.citation = False
        self.run_seq = 0
        self.citations_history = []

    # ======================== Kimi AI 搜索（修复版：强制联网） ========================
    async def kimi_ai_search(
        self,
        search_query: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌙 Kimi AI基础搜索（修复版：强制联网）"""

        # === 内嵌工具函数 ===
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

        def next_run_id(tool: str) -> str:
            self.run_seq += 1
            return f"{tool}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.run_seq}"

        def take_text(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            if len(text) <= max_chars or max_chars <= 0:
                return text
            cut = text[:max_chars]
            # 尝试在最近的句读符处截断
            p = max(
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
                cut.rfind("."),
                cut.rfind(";"),
            )
            if p >= max_chars * 0.6:  # 仅在较靠后才使用
                return cut[: p + 1] + " …"
            return cut + " …"

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

        def search_impl(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """
            根据Kimi文档，使用内置$web_search时，只需要原封不动返回arguments即可
            """
            debug_log(f"Kimi $web_search 参数: {arguments}")
            return arguments

        async def chat_with_tool_calls(messages: list) -> tuple:
            """
            使用工具调用方式与Kimi交互，强制进行联网搜索
            返回: (final_response, search_used, search_results)
            """
            client = get_kimi_client()

            # 使用支持更大上下文的模型
            model_to_use = "moonshot-v1-auto"
            if "kimi" in self.valves.KIMI_MODEL.lower():
                model_to_use = self.valves.KIMI_MODEL
            elif "moonshot" in self.valves.KIMI_MODEL.lower():
                model_to_use = self.valves.KIMI_MODEL
            else:
                # 默认使用auto模型
                model_to_use = "moonshot-v1-auto"

            debug_log(f"使用模型: {model_to_use}")

            finish_reason = None
            search_used = False
            search_results = []
            final_content = ""

            # 添加搜索工具声明
            tools = [
                {
                    "type": "builtin_function",
                    "function": {
                        "name": "$web_search",
                    },
                }
            ]

            max_iterations = 5  # 防止无限循环
            iteration = 0

            while (
                finish_reason is None or finish_reason == "tool_calls"
            ) and iteration < max_iterations:
                iteration += 1
                debug_log(f"Kimi工具调用迭代 {iteration}")

                try:
                    completion = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=model_to_use,
                        messages=messages,
                        temperature=self.valves.KIMI_TEMPERATURE,
                        tools=tools,
                        timeout=60,
                    )

                    choice = completion.choices[0]
                    finish_reason = choice.finish_reason

                    debug_log(f"Kimi响应finish_reason: {finish_reason}")

                    if finish_reason == "tool_calls":
                        search_used = True
                        await emit_status("🔍 Kimi正在进行联网搜索...")

                        # 添加assistant消息到上下文
                        messages.append(choice.message)

                        # 处理工具调用
                        for tool_call in choice.message.tool_calls:
                            tool_call_name = tool_call.function.name
                            tool_call_arguments = json.loads(
                                tool_call.function.arguments
                            )

                            debug_log(f"工具调用: {tool_call_name}")
                            debug_log(f"工具参数: {tool_call_arguments}")

                            if tool_call_name == "$web_search":
                                # 检查tokens消耗信息
                                if "usage" in tool_call_arguments:
                                    search_tokens = tool_call_arguments["usage"].get(
                                        "total_tokens", 0
                                    )
                                    debug_log(f"搜索内容tokens消耗: {search_tokens}")

                                tool_result = search_impl(tool_call_arguments)

                                # 记录搜索结果用于后续解析
                                search_results.append(
                                    {
                                        "arguments": tool_call_arguments,
                                        "result": tool_result,
                                    }
                                )
                            else:
                                tool_result = f"Error: 无法找到工具 '{tool_call_name}'"

                            # 添加工具结果到消息
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_call_name,
                                    "content": json.dumps(
                                        tool_result, ensure_ascii=False
                                    ),
                                }
                            )

                    elif finish_reason == "stop":
                        final_content = choice.message.content or ""
                        if completion.usage:
                            debug_log(
                                f"最终tokens消耗: prompt={completion.usage.prompt_tokens}, completion={completion.usage.completion_tokens}, total={completion.usage.total_tokens}"
                            )
                        break

                except Exception as e:
                    debug_log(f"Kimi工具调用异常: {e}")
                    raise e

            return final_content, search_used, search_results

        def parse_kimi_response(content: str, search_results: List[Dict]) -> tuple:
            """
            解析Kimi的响应内容，提取结构化信息
            返回: (parsed_results, sources)
            """
            debug_log(f"解析Kimi响应内容: {content[:300]}...")

            # 从搜索结果中提取URL信息
            urls_from_search = []
            for sr in search_results:
                args = sr.get("arguments", {})
                if "urls" in args:
                    urls_from_search.extend(args["urls"])
                elif "url" in args:
                    urls_from_search.append(args["url"])

            # 从内容中提取链接
            url_pattern = r'https?://[^\s\)\]\}，。；！？"\']*'
            urls_from_content = re.findall(url_pattern, content)

            # 合并所有URL
            all_urls = list(set(urls_from_search + urls_from_content))
            debug_log(f"提取到的URLs: {all_urls}")

            # 尝试按段落分割内容
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            parsed_results = []

            # 如果内容有明显的结构化信息
            if any(
                marker in content for marker in ["1.", "2.", "一、", "二、", "##", "**"]
            ):
                # 尝试解析结构化内容
                sections = re.split(r"\n(?=\d+\.|\w+、|##|\*\*)", content)
                for i, section in enumerate(sections):
                    section = section.strip()
                    if not section:
                        continue

                    # 提取标题（第一行）
                    lines = section.split("\n")
                    title = lines[0].strip("*#").strip()
                    content_text = (
                        "\n".join(lines[1:]).strip() if len(lines) > 1 else section
                    )

                    # 为每个段落分配URL
                    section_url = ""
                    if i < len(all_urls):
                        section_url = all_urls[i]
                    elif all_urls:
                        section_url = all_urls[0]  # 使用第一个URL作为默认

                    parsed_results.append(
                        {
                            "title": title or f"搜索结果 {i+1}",
                            "content": content_text or section,
                            "url": section_url,
                            "site_name": (
                                section_url.split("/")[2]
                                if section_url and "/" in section_url
                                else "Kimi搜索"
                            ),
                            "date_published": datetime.now().strftime("%Y-%m-%d"),
                            "source_type": "Kimi AI联网搜索",
                        }
                    )
            else:
                # 没有明显结构，将整个内容作为一个结果
                main_url = all_urls[0] if all_urls else ""
                parsed_results.append(
                    {
                        "title": f"关于 {search_query}",
                        "content": content,
                        "url": main_url,
                        "site_name": (
                            main_url.split("/")[2]
                            if main_url and "/" in main_url
                            else "Kimi搜索"
                        ),
                        "date_published": datetime.now().strftime("%Y-%m-%d"),
                        "source_type": "Kimi AI联网搜索",
                    }
                )

            debug_log(f"解析完成，得到 {len(parsed_results)} 个结果")
            return parsed_results, all_urls

        try:
            debug_log(f"开始Kimi AI搜索: {search_query}, 上下文: {context}")

            # 构建搜索提示
            if context:
                search_prompt = f"请搜索关于'{search_query}'的最新信息，特别是在'{context}'背景下的相关内容。请确保进行实时搜索以获取准确和最新的信息。"
                await emit_status(
                    f"🌙 开始Kimi AI联网搜索: {search_query} (背景: {context})"
                )
            else:
                search_prompt = f"请搜索关于'{search_query}'的最新信息。请确保进行实时联网搜索以获取准确和最新的信息，并提供详细的搜索结果。"
                await emit_status(f"🌙 开始Kimi AI联网搜索: {search_query}")

            # 强调联网搜索的系统提示
            system_prompt = """你是Kimi AI助手，具有强大的实时联网搜索能力。

重要指示：
1. 必须使用$web_search工具进行实时联网搜索
2. 不要仅依赖训练数据，必须获取最新信息
3. 搜索结果要包含具体的网站链接和来源
4. 按结构化方式组织搜索结果，包含标题、内容和来源链接

请严格遵循以上要求，确保进行真实的联网搜索。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": search_prompt},
            ]

            # 重试机制确保联网搜索
            search_success = False
            final_content = ""
            all_search_results = []

            for attempt in range(self.valves.KIMI_SEARCH_MAX_RETRIES):
                try:
                    debug_log(
                        f"Kimi搜索尝试 {attempt + 1}/{self.valves.KIMI_SEARCH_MAX_RETRIES}"
                    )

                    content, search_used, search_results = await chat_with_tool_calls(
                        messages.copy()
                    )

                    if search_used:
                        debug_log("✅ 确认Kimi进行了联网搜索")
                        search_success = True
                        final_content = content
                        all_search_results = search_results
                        break
                    else:
                        debug_log(
                            f"⚠️ 第{attempt + 1}次尝试：Kimi未进行联网搜索，准备重试"
                        )
                        if attempt < self.valves.KIMI_SEARCH_MAX_RETRIES - 1:
                            # 修改提示词，更强调联网搜索
                            messages = [
                                {
                                    "role": "system",
                                    "content": system_prompt
                                    + "\n\n特别强调：你必须使用$web_search工具进行联网搜索，不得仅使用已有知识回答。",
                                },
                                {
                                    "role": "user",
                                    "content": f"请立即使用联网搜索功能查找关于'{search_query}'的最新信息。这是第{attempt + 2}次请求，请务必进行实时搜索。",
                                },
                            ]
                            await asyncio.sleep(1)  # 短暂延迟

                except Exception as e:
                    debug_log(f"Kimi搜索尝试 {attempt + 1} 失败: {e}")
                    if attempt == self.valves.KIMI_SEARCH_MAX_RETRIES - 1:
                        raise e
                    await asyncio.sleep(2)

            if not search_success:
                # 如果所有重试都未进行联网搜索，返回警告
                debug_log("❌ 所有重试都未能触发Kimi联网搜索")
                await emit_status(
                    "⚠️ 未能触发联网搜索，返回基础回答", status="warning", done=True
                )

                error_result = {
                    "summary": {
                        "total_results": 0,
                        "total_sources": 0,
                        "search_query": search_query,
                        "context": context,
                        "search_type": "🌙 Kimi AI基础搜索",
                        "timestamp": datetime.now().isoformat(),
                        "status": "warning",
                        "message": "未能触发联网搜索功能",
                    },
                    "results": [],
                }
                return json.dumps(error_result, ensure_ascii=False, indent=2)

            await emit_status("✅ 联网搜索完成，正在处理结果...")

            # 解析搜索结果
            parsed_results, sources = parse_kimi_response(
                final_content, all_search_results
            )

            # 发送引用数据
            for idx, r in enumerate(parsed_results):
                await emit_citation_data(r, __event_emitter__, run_id, idx)

            # 构建返回数据
            results_data = []
            for r in parsed_results:
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
                    "total_results": len(parsed_results),
                    "total_sources": len(sources),
                    "search_query": search_query,
                    "context": context,
                    "search_type": "🌙 Kimi AI联网搜索",
                    "timestamp": datetime.now().isoformat(),
                    "search_verified": search_success,
                },
                "results": results_data,
            }

            await emit_status("🎉 Kimi AI联网搜索完成！", status="complete", done=True)
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            debug_log("Kimi AI搜索失败", e)
            await emit_status(
                f"❌ Kimi AI搜索失败: {str(e)}", status="error", done=True
            )

            error_result = {
                "summary": {
                    "total_results": 0,
                    "total_sources": 0,
                    "search_query": search_query,
                    "context": context,
                    "search_type": "🌙 Kimi AI联网搜索",
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(e),
                },
                "results": [],
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
            if len(text) <= max_chars or max_chars <= 0:
                return text
            cut = text[:max_chars]
            # 尝试在最近的句读符处截断
            p = max(
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
                cut.rfind("."),
                cut.rfind(";"),
            )
            if p >= max_chars * 0.6:  # 仅在较靠后才使用
                return cut[: p + 1] + " …"
            return cut + " …"

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

        # 意图拆解 + 多视角相似度
        def plan_aspects(user_request: str):
            """轻量规划器：根据查询意图拆解搜索视角"""
            buckets = []
            if re.search(r"意义|含义|象征|本质|哲学", user_request):
                buckets += [
                    "数学定义 与 单位元",
                    "文化/哲学象征",
                    "应用场景 与 归一化/计量",
                    "语言学/词源",
                ]
            else:
                buckets += ["核心定义", "性质/定理", "历史与符号", "应用与工程"]
            return buckets[:4]

        async def batch_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
            """批量向量化"""
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
                        v = await get_text_embedding(t)
                        fallback.append(v)
                    return fallback

                return vecs

            except Exception:
                out = []
                for t in texts:
                    v = await get_text_embedding(t)
                    out.append(v)
                return out

        async def enhance_results_with_rag(results: List[Dict]) -> List[Dict]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not results:
                debug_log("RAG未启用或结果为空")
                return results

            try:
                await emit_status(f"🧠 正在进行RAG向量化优化 ({len(results)} 个结果)")
                debug_log(f"开始RAG优化，查询: {query}, 结果数: {len(results)}")

                # 多视角相似度融合
                aspects = plan_aspects(query)
                all_texts = [query] + aspects
                all_vecs = await batch_embeddings(all_texts)

                query_vec = all_vecs[0] if all_vecs else None
                aspect_vecs = all_vecs[1:] if len(all_vecs) > 1 else []

                def fuse_similarity(doc_vec):
                    """融合查询和各方面子查询的相似度"""
                    sims = []
                    if query_vec is not None and doc_vec is not None:
                        sims.append(calculate_similarity(query_vec, doc_vec))
                    for av in aspect_vecs:
                        if av is not None and doc_vec is not None:
                            sims.append(calculate_similarity(av, doc_vec))
                    return max(sims) if sims else 0.0

                if not query_vec:
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
                        similarity = fuse_similarity(content_embedding)
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

                # 构造 documents 时建立映射
                documents = []
                doc_to_result_idx = []

                for i, result in enumerate(results):
                    content = (result.get("content") or "")[:4000]
                    if content:
                        documents.append(content)
                        doc_to_result_idx.append(i)

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

                    # 回填时使用映射
                    for rerank_item in rerank_results_data:
                        doc_idx = rerank_item.get("index", 0)
                        relevance_score = rerank_item.get("relevance_score", 0.0)

                        if 0 <= doc_idx < len(doc_to_result_idx):
                            orig_idx = doc_to_result_idx[doc_idx]
                            result = results[orig_idx].copy()
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

            # 评分归一化与顺序
            def zminmax(vals):
                if not vals:
                    return []
                lo, hi = min(vals), max(vals)
                if hi - lo < 1e-6:
                    return [0.5 for _ in vals]
                return [(v - lo) / (hi - lo) for v in vals]

            if source_context_list:
                rags = [float(x.get("rag_similarity", 0)) for x in source_context_list]
                rers = [float(x.get("rerank_score", 0)) for x in source_context_list]

                rags_n, rers_n = map(zminmax, (rags, rers))

                for i, s in enumerate(source_context_list):
                    s["final_score"] = (
                        self.valves.RAG_WEIGHT * rags_n[i]
                        + self.valves.RERANK_WEIGHT * rers_n[i]
                    )

                # 先排序取前N，再按阈值做轻过滤
                source_context_list.sort(
                    key=lambda x: x.get("final_score", 0), reverse=True
                )
                source_context_list = source_context_list[: self.valves.RERANK_TOP_N]

                if (
                    self.valves.ENABLE_RAG_ENHANCEMENT
                    and self.valves.EMIT_ONLY_RAG_PASS
                ):
                    thr = max(
                        0.03, float(self.valves.SIMILARITY_THRESHOLD) * 0.6
                    )  # 放宽
                    source_context_list = [
                        s
                        for s in source_context_list
                        if float(s.get("rag_similarity", 0)) >= thr
                        or s["final_score"] >= 0.35
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
                    "final_score": float(r.get("final_score") or 0.0),
                    "snippet": take_text(r.get("content", ""), 450),
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
            if len(text) <= max_chars or max_chars <= 0:
                return text
            cut = text[:max_chars]
            # 尝试在最近的句读符处截断
            p = max(
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
                cut.rfind("."),
                cut.rfind(";"),
            )
            if p >= max_chars * 0.6:  # 仅在较靠后才使用
                return cut[: p + 1] + " …"
            return cut + " …"

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

        # 意图拆解 + 多视角相似度
        def plan_aspects(user_request: str):
            """轻量规划器：根据查询意图拆解搜索视角"""
            buckets = []
            if re.search(
                r"meaning|significance|symbolism|essence|philosophy", user_request, re.I
            ):
                buckets += [
                    "mathematical definition",
                    "cultural/philosophical symbolism",
                    "application scenarios",
                    "linguistic/etymology",
                ]
            else:
                buckets += [
                    "core definition",
                    "properties/theorems",
                    "history and symbols",
                    "applications and engineering",
                ]
            return buckets[:4]

        async def batch_embeddings(texts: List[str]) -> List[Optional[List[float]]]:
            """批量向量化"""
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
                        v = await get_text_embedding(t)
                        fallback.append(v)
                    return fallback

                return vecs

            except Exception:
                out = []
                for t in texts:
                    v = await get_text_embedding(t)
                    out.append(v)
                return out

        async def enhance_results_with_rag(results: List[Dict]) -> List[Dict]:
            if not self.valves.ENABLE_RAG_ENHANCEMENT or not results:
                return results

            try:
                await emit_status(f"🧠 RAG优化 ({len(results)} 个结果)")

                # 多视角相似度融合
                aspects = plan_aspects(query)
                all_texts = [query] + aspects
                all_vecs = await batch_embeddings(all_texts)

                query_vec = all_vecs[0] if all_vecs else None
                aspect_vecs = all_vecs[1:] if len(all_vecs) > 1 else []

                def fuse_similarity(doc_vec):
                    """融合查询和各方面子查询的相似度"""
                    sims = []
                    if query_vec is not None and doc_vec is not None:
                        sims.append(calculate_similarity(query_vec, doc_vec))
                    for av in aspect_vecs:
                        if av is not None and doc_vec is not None:
                            sims.append(calculate_similarity(av, doc_vec))
                    return max(sims) if sims else 0.0

                if not query_vec:
                    return results

                enhanced_results = []
                for i, result in enumerate(results):
                    content = result.get("content", "")
                    if not content:
                        continue

                    content_embedding = await get_text_embedding(content)
                    if content_embedding:
                        similarity = fuse_similarity(content_embedding)
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

                # 构造 documents 时建立映射
                documents = []
                doc_to_result_idx = []

                for i, result in enumerate(results):
                    content = (result.get("content") or "")[:4000]
                    if content:
                        documents.append(content)
                        doc_to_result_idx.append(i)

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

                    # 回填时使用映射
                    for rerank_item in rerank_results_data:
                        doc_idx = rerank_item.get("index", 0)
                        relevance_score = rerank_item.get("relevance_score", 0.0)

                        if 0 <= doc_idx < len(doc_to_result_idx):
                            orig_idx = doc_to_result_idx[doc_idx]
                            result = results[orig_idx].copy()
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

            # 评分归一化与顺序
            def zminmax(vals):
                if not vals:
                    return []
                lo, hi = min(vals), max(vals)
                if hi - lo < 1e-6:
                    return [0.5 for _ in vals]
                return [(v - lo) / (hi - lo) for v in vals]

            if source_context_list:
                rags = [float(x.get("rag_similarity", 0)) for x in source_context_list]
                rers = [float(x.get("rerank_score", 0)) for x in source_context_list]

                rags_n, rers_n = map(zminmax, (rags, rers))

                for i, s in enumerate(source_context_list):
                    s["final_score"] = (
                        self.valves.RAG_WEIGHT * rags_n[i]
                        + self.valves.RERANK_WEIGHT * rers_n[i]
                    )

                # 先排序取前N，再按阈值做轻过滤
                source_context_list.sort(
                    key=lambda x: x.get("final_score", 0), reverse=True
                )
                source_context_list = source_context_list[: self.valves.RERANK_TOP_N]

                if (
                    self.valves.ENABLE_RAG_ENHANCEMENT
                    and self.valves.EMIT_ONLY_RAG_PASS
                ):
                    thr = max(
                        0.03, float(self.valves.SIMILARITY_THRESHOLD) * 0.6
                    )  # 放宽
                    source_context_list = [
                        s
                        for s in source_context_list
                        if float(s.get("rag_similarity", 0)) >= thr
                        or s["final_score"] >= 0.35
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
                    "final_score": float(r.get("final_score") or 0.0),
                    "snippet": take_text(r.get("content", ""), 450),
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

    # ======================== 智能网页读取功能（修复版） ========================
    async def web_scrape(
        self,
        urls: List[str],
        user_request: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 智能网页读取工具 (修复版)"""

        # === 表格扁平化工具函数 ===
        def _flatten_md_tables(text: str) -> str:
            """将Markdown表格转为条目列表"""
            lines = text.splitlines()
            out = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith("|") and "|" in line:
                    # 收集整块表格
                    tbl = [line]
                    i += 1
                    while i < len(lines) and (
                        lines[i].strip().startswith("|")
                        or re.match(r"^\s*[:\-\|\s]+$", lines[i])
                    ):
                        tbl.append(lines[i])
                        i += 1

                    # 解析表头和行
                    if len(tbl) >= 3:  # 至少要有表头、分割线、数据行
                        header = [h.strip() for h in tbl[0].strip("| ").split("|")]
                        for r in tbl[2:]:  # 跳过对齐行
                            cells = [c.strip() for c in r.strip("| ").split("|")]
                            if len(cells) == len(header):
                                # 转为要点行
                                kv = [
                                    f"{header[j]}：{cells[j]}"
                                    for j in range(len(header))
                                    if cells[j]
                                ]
                                if kv:
                                    out.append("• " + "；".join(kv))
                    else:
                        # 表格格式不完整，保持原样
                        out.extend(tbl)
                else:
                    out.append(line)
                    i += 1
            return "\n".join(out)

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

            if self.valves.PRESERVE_TABLES and not self.valves.FLATTEN_TABLES:
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

        # LLM调用硬超时
        async def llm_call(
            messages: list, temperature: float = None, max_tokens: int = 4000
        ) -> str:
            """调用LLM（修复版：重试+线程池+超时）"""
            client = get_segmenter_client()
            temp = (
                temperature
                if temperature is not None
                else self.valves.SUMMARY_TEMPERATURE
            )

            last_err = None
            for attempt in range(self.valves.LLM_RETRIES + 1):
                try:
                    # 添加硬超时保护
                    resp = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.chat.completions.create,
                            model=self.valves.SEGMENTER_MODEL,
                            messages=messages,
                            temperature=temp,
                            max_tokens=max_tokens,
                        ),
                        timeout=self.valves.LLM_REQUEST_TIMEOUT_SEC,
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
            if len(text) <= max_chars or max_chars <= 0:
                return text
            cut = text[:max_chars]
            # 尝试在最近的句读符处截断
            p = max(
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
                cut.rfind("."),
                cut.rfind(";"),
            )
            if p >= max_chars * 0.6:  # 仅在较靠后才使用
                return cut[: p + 1] + " …"
            return cut + " …"

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

        # 进度条管理器
        class ProgressManager:
            def __init__(self, total_steps: int):
                self.total_steps = total_steps
                self.current_step = 0

            async def update_step(self, description: str, __event_emitter__):
                self.current_step += 1
                percentage = int((self.current_step / self.total_steps) * 100)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "in_progress",
                                "description": f"[{percentage}%] {description}",
                                "done": False,
                                "progress": percentage,
                                "step": self.current_step,
                                "total_steps": self.total_steps,
                                "action": "web_scrape",
                            },
                        }
                    )

        # 修复版摘要提取函数
        async def extract_summaries_fixed(
            content: str,
            user_request: str,
            url: str,
            page_title: str,
            progress_mgr: ProgressManager,
        ) -> List[Dict]:
            """修复版摘要提取：解决9个分片只返回1个结果的问题"""

            def cleanup(text: str) -> str:
                t = re.sub(r"\n{4,}", "\n\n", text)
                t = re.sub(r"[ \t]{3,}", " ", t)
                t = re.sub(r"\[\d+\]", "", t)
                return t.strip()

            cleaned = cleanup(content)
            if not cleaned:
                return []

            # 表格扁平化
            if self.valves.FLATTEN_TABLES:
                cleaned = _flatten_md_tables(cleaned)

            chunks = smart_segment_text(cleaned)
            if not chunks:
                return []

            if len(chunks) > int(self.valves.MAX_TOTAL_CHUNKS):
                chunks = chunks[: int(self.valves.MAX_TOTAL_CHUNKS)]

            debug_log(f"分片完成：{len(chunks)} 片")

            await progress_mgr.update_step(
                f"📄 开始处理 {len(chunks)} 个分片", __event_emitter__
            )

            # 并发控制
            sem = asyncio.Semaphore(self.valves.LLM_MAX_CONCURRENCY)
            per_chunk = max(2, min(4, int(self.valves.MAP_SUMMARY_PER_CHUNK)))

            def _extract_json_array(text: str, debug_chunk_idx: int = -1) -> List[dict]:
                """增强的JSON数组提取，带调试信息"""
                if not text:
                    debug_log(f"分片{debug_chunk_idx} JSON提取：输入为空")
                    return []

                t = text.strip()
                debug_log(f"分片{debug_chunk_idx} LLM原始响应: {t[:200]}...")

                # 清理代码块标记
                if t.startswith("```"):
                    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.I | re.M).strip()

                # 尝试完整JSON解析
                try:
                    obj = json.loads(t)
                    result = obj if isinstance(obj, list) else []
                    debug_log(
                        f"分片{debug_chunk_idx} 完整JSON解析成功，得到{len(result)}个项目"
                    )
                    return result
                except Exception as e1:
                    debug_log(f"分片{debug_chunk_idx} 完整JSON解析失败: {e1}")

                # 尝试提取JSON数组部分
                s, e = t.find("["), t.rfind("]")
                if s != -1 and e != -1 and e > s:
                    try:
                        obj = json.loads(t[s : e + 1])
                        result = obj if isinstance(obj, list) else []
                        debug_log(
                            f"分片{debug_chunk_idx} 部分JSON解析成功，得到{len(result)}个项目"
                        )
                        return result
                    except Exception as e2:
                        debug_log(f"分片{debug_chunk_idx} 部分JSON解析也失败: {e2}")

                debug_log(f"分片{debug_chunk_idx} 所有JSON解析都失败，返回空列表")
                return []

            # 本地兜底填充器
            STOPWORDS = set(
                list("的一是在不了有和就也而及与或被于把等其并之之于以为")
            ) | {
                "the",
                "a",
                "an",
                "and",
                "or",
                "of",
                "to",
                "in",
                "on",
                "for",
                "as",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
            }

            def derive_key_points(text: str, topk=4):
                """朴素词频关键词"""
                tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)
                cnt = {}
                for t in tokens:
                    if t.lower() in STOPWORDS:
                        continue
                    cnt[t] = cnt.get(t, 0) + 1
                return [w for w, _ in sorted(cnt.items(), key=lambda x: -x[1])[:topk]]

            async def _extract_one_chunk(idx: int, c: dict):
                """单个分片的摘要提取 - 修复版"""
                # 更清晰的系统提示，强调输出格式
                sys_prompt = f"""你是专业信息提取专家。基于给定内容片段，围绕用户需求提取{per_chunk}条摘要。

**重要要求：**
1. 必须输出JSON数组格式：[{{"summary": "摘要内容", "relevance": 0.8}}]
2. 每条摘要控制在{self.valves.SUMMARY_MIN_CHARS}-{self.valves.SUMMARY_MAX_CHARS}个字符
3. 摘要要完整表达一个要点，语句完整通顺
4. relevance为0-1的相关度分数
5. 如果内容不相关或无法提取，返回[]

**严格按照JSON格式输出，不要任何额外说明。**"""

                user_prompt = f"""用户需求：{user_request}

分片内容：
{c['text'][:4000]}

请严格按JSON数组格式输出："""

                try:
                    async with sem:
                        resp = await llm_call(
                            [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=self.valves.SUMMARY_TEMPERATURE,
                            max_tokens=3000,
                        )

                    # 增强的JSON解析，带调试信息
                    arr = _extract_json_array(resp, idx)
                    out = []

                    for item_idx, item in enumerate(arr):
                        if not isinstance(item, dict):
                            debug_log(
                                f"分片{idx} 第{item_idx}项不是字典，跳过: {type(item)}"
                            )
                            continue

                        s = (item.get("summary") or "").strip()
                        if not s:
                            debug_log(f"分片{idx} 第{item_idx}项摘要为空，跳过")
                            continue

                        # 修复：更宽松的字数检查，避免过度过滤
                        if (
                            len(s) < int(self.valves.SUMMARY_MIN_CHARS) * 0.7
                        ):  # 允许30%的弹性
                            debug_log(
                                f"分片{idx} 第{item_idx}项过短({len(s)}字符)，跳过"
                            )
                            continue

                        # 温和的长度处理
                        if len(s) > int(self.valves.SUMMARY_MAX_CHARS):
                            cut_pos = int(self.valves.SUMMARY_MAX_CHARS)
                            sentence_end = max(
                                s.rfind("。", 0, cut_pos),
                                s.rfind("！", 0, cut_pos),
                                s.rfind("？", 0, cut_pos),
                                s.rfind(".", 0, cut_pos),
                            )
                            if sentence_end > cut_pos * 0.7:
                                s = s[: sentence_end + 1]
                                debug_log(
                                    f"分片{idx} 第{item_idx}项在句子边界截断为{len(s)}字符"
                                )
                            else:
                                s = s[:cut_pos] + "..."
                                debug_log(
                                    f"分片{idx} 第{item_idx}项强制截断为{len(s)}字符"
                                )

                        # 兜底填充
                        kp = derive_key_points(s)
                        out.append(
                            {
                                "content": s,
                                "title": f"{page_title} · 摘要",
                                "url": url,
                                "relevance": float(item.get("relevance", 0.7)),
                                "key_points": kp,
                                "extract_method": "fixed_concurrent",
                                "source_type": "LLM智能摘要",
                                "chunk_index": idx,
                            }
                        )

                    debug_log(f"分片{idx} 成功提取{len(out)}条摘要")
                    return out

                except Exception as e:
                    debug_log(f"分片 {idx+1} 摘要提取异常：{e}")
                    # 如果JSON解析完全失败，尝试基于原始响应创建摘要
                    try:
                        # 将LLM响应作为单条摘要处理
                        if (
                            resp
                            and len(resp.strip())
                            >= int(self.valves.SUMMARY_MIN_CHARS) * 0.5
                        ):
                            content = resp.strip()
                            if len(content) > int(self.valves.SUMMARY_MAX_CHARS):
                                cut_pos = int(self.valves.SUMMARY_MAX_CHARS)
                                sentence_end = max(
                                    content.rfind("。", 0, cut_pos),
                                    content.rfind(".", 0, cut_pos),
                                )
                                if sentence_end > cut_pos * 0.7:
                                    content = content[: sentence_end + 1]
                                else:
                                    content = content[:cut_pos] + "..."

                            fallback_item = {
                                "content": content,
                                "title": f"{page_title} · 摘要",
                                "url": url,
                                "relevance": 0.6,
                                "key_points": derive_key_points(content),
                                "extract_method": "fallback_from_response",
                                "source_type": "LLM响应回退",
                                "chunk_index": idx,
                            }
                            debug_log(f"分片{idx} 使用响应回退创建1条摘要")
                            return [fallback_item]
                    except:
                        pass

                    return []

            # 并发执行，追踪每个分片结果
            tasks = [_extract_one_chunk(idx, c) for idx, c in enumerate(chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_summaries = []
            successful_chunks = 0
            failed_chunks = 0

            for i, r in enumerate(results):
                if isinstance(r, list):
                    all_summaries.extend(r)
                    if r:  # 有结果
                        successful_chunks += 1
                        debug_log(f"分片{i}成功提取{len(r)}条摘要")
                    else:  # 空结果
                        failed_chunks += 1
                        debug_log(f"分片{i}提取结果为空")
                else:
                    failed_chunks += 1
                    debug_log(f"分片{i}出现异常: {r}")

            debug_log(
                f"并发摘要提取完成：成功{successful_chunks}个分片，失败{failed_chunks}个分片，总摘要{len(all_summaries)}条"
            )

            # 如果提取效果太差，启动强化回退
            if len(all_summaries) < max(2, len(chunks) * 0.3):  # 如果摘要数量太少
                debug_log(f"摘要提取效果不佳（{len(all_summaries)}条），启动强化回退")
                for i, chunk in enumerate(chunks[:5]):  # 最多处理5个分片
                    try:
                        # 直接将分片内容作为摘要，温和处理长度
                        fallback_content = chunk["text"]
                        if len(fallback_content) > int(self.valves.SUMMARY_MAX_CHARS):
                            cut_pos = int(self.valves.SUMMARY_MAX_CHARS)
                            sentence_end = max(
                                fallback_content.rfind("。", 0, cut_pos),
                                fallback_content.rfind(".", 0, cut_pos),
                            )
                            if sentence_end > cut_pos * 0.7:
                                fallback_content = fallback_content[: sentence_end + 1]
                            else:
                                fallback_content = fallback_content[:cut_pos] + "..."

                        fallback_summary = {
                            "content": fallback_content,
                            "title": f"{page_title} · 分片摘要{i+1}",
                            "url": url,
                            "relevance": 0.6,
                            "key_points": derive_key_points(fallback_content),
                            "extract_method": "enhanced_fallback",
                            "source_type": "强化回退摘要",
                            "chunk_index": i,
                        }
                        all_summaries.append(fallback_summary)
                    except Exception as e:
                        debug_log(f"强化回退分片{i}也失败: {e}")

            await progress_mgr.update_step(
                f"✅ 摘要提取完成，获得 {len(all_summaries)} 条摘要", __event_emitter__
            )

            debug_log(f"最终摘要提取完成：共 {len(all_summaries)} 条")
            return all_summaries

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
                    debug_log("批量向量化长度不匹配，回退单个处理")
                    fallback = []
                    for t in texts:
                        v = await get_single_embedding(t)
                        fallback.append(v)
                    return fallback

                return vecs

            except Exception as e:
                debug_log(f"批量向量化失败：{e}")
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

        # 意图拆解 + 多视角相似度
        def plan_aspects(user_request: str):
            """轻量规划器：根据查询意图拆解搜索视角"""
            buckets = []
            if re.search(
                r"意义|含义|象征|本质|哲学|meaning|significance", user_request, re.I
            ):
                buckets += [
                    "数学定义 与 单位元",
                    "文化/哲学象征",
                    "应用场景 与 归一化/计量",
                    "语言学/词源",
                ]
            else:
                buckets += ["核心定义", "性质/定理", "历史与符号", "应用与工程"]
            return buckets[:4]

        # 去重工具
        def dedup_by_embedding(items, vecs, thr=0.88):
            """基于embedding的去重"""
            kept, kept_vecs = [], []
            for it, v in zip(items, vecs):
                if v is None:
                    kept.append(it)
                    kept_vecs.append(v)
                    continue
                if any(
                    cos_similarity(v, kv) >= thr for kv in kept_vecs if kv is not None
                ):
                    continue
                kept.append(it)
                kept_vecs.append(v)
            return kept

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

        # 初始化进度管理器
        total_steps = 6  # 读取网页、摘要提取、RAG、重排序、评分、完成
        progress_mgr = ProgressManager(total_steps)

        try:
            debug_log(f"开始智能网页读取，URL数量: {len(urls)}")

            await progress_mgr.update_step(
                f"🚀 开始处理 {len(urls)} 个网页", __event_emitter__
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

            await progress_mgr.update_step(
                f"📖 成功读取 {len(successful_results)} 个网页", __event_emitter__
            )

            if not successful_results:
                return json.dumps(
                    {
                        "request": user_request,
                        "error": "所有网页读取都失败",
                        "summaries_count": 0,
                        "summaries": [],
                        "errors": error_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # 智能摘要提取流程
            if self.valves.ENABLE_SMART_SUMMARY:
                all_summaries = []

                for i, page in enumerate(successful_results):
                    content = page.get("content", "")
                    url = page.get("url", "")
                    title = page.get("title", "")

                    debug_log(f"为页面 {i+1}/{len(successful_results)} 提取摘要: {url}")

                    try:
                        summaries = await extract_summaries_fixed(
                            content=content,
                            user_request=user_request,
                            url=url,
                            page_title=title,
                            progress_mgr=progress_mgr,
                        )
                        debug_log(f"页面 {url} 提取到 {len(summaries)} 条摘要")
                        all_summaries.extend(summaries)
                    except Exception as e:
                        debug_log(f"页面 {url} 摘要提取失败: {e}")

                debug_log(f"所有页面摘要提取完成，总计 {len(all_summaries)} 条摘要")

                # RAG处理
                if self.valves.ENABLE_RAG_ENHANCEMENT and all_summaries:
                    await progress_mgr.update_step(
                        f"🎯 RAG向量化处理 {len(all_summaries)} 条摘要",
                        __event_emitter__,
                    )

                    # 多视角相似度融合
                    aspects = plan_aspects(user_request)
                    all_texts = (
                        [user_request] + aspects + [s["content"] for s in all_summaries]
                    )
                    all_vecs = await batch_embeddings(all_texts)

                    query_vec = all_vecs[0] if all_vecs else None
                    aspect_vecs = (
                        all_vecs[1 : len(aspects) + 1]
                        if len(all_vecs) > len(aspects)
                        else []
                    )
                    summary_vecs = (
                        all_vecs[len(aspects) + 1 :]
                        if len(all_vecs) > len(aspects) + 1
                        else []
                    )

                    def fuse_similarity(doc_vec):
                        """融合查询和各方面子查询的相似度"""
                        sims = []
                        if query_vec is not None and doc_vec is not None:
                            sims.append(cos_similarity(query_vec, doc_vec))
                        for av in aspect_vecs:
                            if av is not None and doc_vec is not None:
                                sims.append(cos_similarity(av, doc_vec))
                        return max(sims) if sims else 0.0

                    for i, summary in enumerate(all_summaries):
                        if i < len(summary_vecs) and summary_vecs[i] is not None:
                            similarity = fuse_similarity(summary_vecs[i])
                            summary["rag_similarity"] = similarity
                        else:
                            summary["rag_similarity"] = summary.get("relevance", 0.6)

                    # 去重
                    all_summaries = dedup_by_embedding(
                        all_summaries, summary_vecs, thr=0.88
                    )

                    all_summaries.sort(
                        key=lambda x: x.get("rag_similarity", 0), reverse=True
                    )

                # 语义重排序
                if (
                    self.valves.ENABLE_SEMANTIC_RERANK
                    and self.valves.BOCHA_API_KEY
                    and all_summaries
                ):
                    await progress_mgr.update_step(
                        f"🎯 语义重排序 {len(all_summaries)} 条摘要", __event_emitter__
                    )

                    try:
                        headers = {
                            "Authorization": f"Bearer {self.valves.BOCHA_API_KEY}",
                            "Content-Type": "application/json",
                        }

                        # 构造 documents 时建立映射
                        documents = []
                        doc_to_result_idx = []

                        for i, s in enumerate(all_summaries):
                            content = s["content"][:4000]
                            if content:
                                documents.append(content)
                                doc_to_result_idx.append(i)

                        if documents:
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

                                # 回填时使用映射
                                for rerank_item in rerank_results_data:
                                    doc_idx = rerank_item.get("index", 0)
                                    relevance_score = rerank_item.get(
                                        "relevance_score", 0.0
                                    )

                                    if 0 <= doc_idx < len(doc_to_result_idx):
                                        orig_idx = doc_to_result_idx[doc_idx]
                                        summary = all_summaries[orig_idx].copy()
                                        summary["rerank_score"] = relevance_score
                                        reranked_summaries.append(summary)

                                all_summaries = reranked_summaries
                                debug_log(
                                    f"重排序完成，保留 {len(all_summaries)} 条摘要"
                                )

                    except Exception as e:
                        debug_log(f"语义重排序失败: {e}")

                # 最终评分 - 只用RAG和rerank
                await progress_mgr.update_step(
                    "🏆 计算最终评分并筛选结果", __event_emitter__
                )

                def zminmax(vals):
                    if not vals:
                        return []
                    lo, hi = min(vals), max(vals)
                    if hi - lo < 1e-6:
                        return [0.5 for _ in vals]
                    return [(v - lo) / (hi - lo) for v in vals]

                if all_summaries:
                    rags = [float(x.get("rag_similarity", 0)) for x in all_summaries]
                    rers = [float(x.get("rerank_score", 0)) for x in all_summaries]

                    rags_n, rers_n = map(zminmax, (rags, rers))

                    for i, s in enumerate(all_summaries):
                        s["final_score"] = (
                            self.valves.RAG_WEIGHT * rags_n[i]
                            + self.valves.RERANK_WEIGHT * rers_n[i]
                        )

                    # 排序并筛选
                    all_summaries.sort(
                        key=lambda x: x.get("final_score", 0), reverse=True
                    )
                    final_summaries = all_summaries[: self.valves.RERANK_TOP_N]

                    # 阈值过滤
                    if (
                        self.valves.ENABLE_RAG_ENHANCEMENT
                        and self.valves.EMIT_ONLY_RAG_PASS
                    ):
                        thr = max(0.03, float(self.valves.SIMILARITY_THRESHOLD) * 0.6)
                        if any(is_wikipedia(r["url"]) for r in successful_results):
                            thr = max(0.03, thr * 0.4)
                            debug_log(f"检测到维基百科，放宽阈值到: {thr}")

                        final_summaries = [
                            s
                            for s in final_summaries
                            if float(s.get("rag_similarity", 0)) >= thr
                            or s["final_score"] >= 0.35
                        ]
                else:
                    final_summaries = []

                debug_log(f"最终保留 {len(final_summaries)} 条摘要")

                # 发送引用
                for idx, summary in enumerate(final_summaries):
                    await emit_citation_data(summary, __event_emitter__, run_id, idx)

                await progress_mgr.update_step("🎉 处理完成！", __event_emitter__)

                # 构建返回体 - 简化stats信息
                results_data = []
                for summary in final_summaries:
                    item = {
                        "title": summary.get("title") or "",
                        "url": summary.get("url"),
                        "rag_similarity": float(summary.get("rag_similarity", 0.0)),
                        "rerank_score": float(summary.get("rerank_score", 0.0)),
                        "final_score": float(summary.get("final_score", 0.0)),
                        "key_points": summary.get("key_points", []),
                        "snippet": summary.get(
                            "content", ""
                        ),  # 使用完整摘要作为snippet
                    }
                    results_data.append(item)

                return json.dumps(
                    {
                        "summaries_count": len(final_summaries),
                        "summaries": results_data,
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
                        "snippet": take_text(r.get("content", ""), 600),
                    }
                    results_data.append(result_item)

                return json.dumps(
                    {
                        "results_count": len(successful_results),
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
                    "error": str(e),
                    "summaries_count": 0,
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
            if len(text) <= max_chars or max_chars <= 0:
                return text
            cut = text[:max_chars]
            # 尝试在最近的句读符处截断
            p = max(
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
                cut.rfind("."),
                cut.rfind(";"),
            )
            if p >= max_chars * 0.6:  # 仅在较靠后才使用
                return cut[: p + 1] + " …"
            return cut + " …"

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
            if len(text) <= max_chars or max_chars <= 0:
                return text
            cut = text[:max_chars]
            # 尝试在最近的句读符处截断
            p = max(
                cut.rfind("。"),
                cut.rfind("！"),
                cut.rfind("？"),
                cut.rfind("."),
                cut.rfind(";"),
            )
            if p >= max_chars * 0.6:  # 仅在较靠后才使用
                return cut[: p + 1] + " …"
            return cut + " …"

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
                    "snippet": take_text(r.get("content", ""), 450),
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

    # Kimi AI基础搜索（修复版：强制联网）
    async def kimi_ai_search(
        self,
        search_query: str,
        context: str = "",
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌙 Kimi AI联网搜索 - 强制使用内置$web_search工具进行真实联网搜索"""
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

    # 智能网页读取（修复版）
    async def web_scrape(
        self,
        urls: List[str],
        user_request: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """🌐 智能网页读取工具（修复版）"""
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
