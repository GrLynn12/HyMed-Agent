"""千问/DashScope LLM 调用封装。

项目运行时只依赖两个能力：

* complete: 非流式生成，用于意图识别
* stream_chat: 流式生成，用于最终答案输出

DashScope 提供 OpenAI 兼容接口，因此这里使用 openai SDK，后续接入 tool call
或 LangGraph 时也能复用同一套 messages/tools 协议。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from medical_rag.core.config import settings

logger = logging.getLogger(__name__)


Message = Dict[str, Any]


class QwenClient:
    """基于 DashScope OpenAI 兼容接口的千问客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.QWEN_API_KEY
        self.base_url = base_url or settings.QWEN_BASE_URL
        self.model = model or settings.QWEN_MODEL
        if not self.api_key:
            raise RuntimeError(
                "未配置千问 API Key，请设置 QWEN_API_KEY 或 DASHSCOPE_API_KEY 环境变量。"
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str) -> str:
        """非流式生成文本，主要用于意图识别。"""
        completion = self.chat([{"role": "user", "content": prompt}])
        message = completion.choices[0].message
        content = message.content or ""
        logger.debug("千问非流式响应: %s", content)
        return content

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        """非流式 chat completion，可选 OpenAI 兼容 tool calling。"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.QWEN_TEMPERATURE,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def stream_chat(self, messages: List[Message]) -> Iterable[str]:
        """流式生成文本片段。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.QWEN_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            if content:
                yield content


def get_llm_client() -> QwenClient:
    """构建默认千问客户端。单独函数便于 Streamlit cache_resource 缓存。"""
    return QwenClient()
