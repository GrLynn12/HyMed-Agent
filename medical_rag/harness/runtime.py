"""Agent Harness 运行时。

Harness 不负责医学推理，只负责执行边界：预算、超时、重复调用、工具参数
校验和结构化 trace。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List

from medical_rag.core.config import settings
from medical_rag.retrieval.tools import ToolResult


ALLOWED_TOOLS = {"medical_graph_search", "medical_vector_search"}


@dataclass(frozen=True)
class HarnessConfig:
    """一次 Agent run 的执行限制。"""

    max_tool_calls: int = settings.AGENT_MAX_TOOL_CALLS
    max_rewrites: int = settings.AGENT_MAX_REWRITES
    timeout_seconds: float = settings.AGENT_TIMEOUT_SECONDS
    max_same_tool_calls: int = settings.AGENT_MAX_SAME_TOOL_CALLS
    max_top_k: int = settings.AGENT_MAX_TOP_K


@dataclass(frozen=True)
class ToolCallRequest:
    """经过解析、准备交给 Harness 的工具调用。"""

    name: str
    arguments: Dict[str, Any]
    reason: str = ""


@dataclass
class AgentHarness:
    """管理单次 ReAct run 的执行状态。"""

    config: HarnessConfig = field(default_factory=HarnessConfig)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.monotonic)
    tool_call_count: int = 0
    rewrite_count: int = 0
    call_signatures: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def timed_out(self) -> bool:
        return self.elapsed_seconds() >= self.config.timeout_seconds

    def add_trace(self, event: str, **details: Any) -> None:
        self.trace.append(
            {
                "node": event,
                "run_id": self.run_id,
                "elapsed_ms": round(self.elapsed_seconds() * 1000, 2),
                **details,
            }
        )

    def consume_rewrite(self) -> bool:
        if self.rewrite_count >= self.config.max_rewrites:
            self.add_trace(
                "harness_guardrail",
                allowed=False,
                reason="rewrite_budget_exceeded",
            )
            return False
        self.rewrite_count += 1
        return True

    def _normalize_request(self, request: ToolCallRequest) -> ToolCallRequest:
        name = request.name.strip()
        arguments = dict(request.arguments or {})
        query = str(arguments.get("query", "")).strip()
        arguments["query"] = query
        if name == "medical_vector_search":
            try:
                top_k = int(arguments.get("top_k") or settings.VECTOR_TOP_K)
            except (TypeError, ValueError):
                top_k = settings.VECTOR_TOP_K
            arguments["top_k"] = min(max(top_k, 1), self.config.max_top_k)
        else:
            arguments.pop("top_k", None)
        return ToolCallRequest(name=name, arguments=arguments, reason=request.reason)

    def validate_tool_call(
        self,
        request: ToolCallRequest,
    ) -> tuple[bool, str, ToolCallRequest]:
        normalized = self._normalize_request(request)
        if self.timed_out():
            return False, "timeout", normalized
        if self.tool_call_count >= self.config.max_tool_calls:
            return False, "tool_budget_exceeded", normalized
        if normalized.name not in ALLOWED_TOOLS:
            return False, "tool_not_allowed", normalized
        if not normalized.arguments.get("query"):
            return False, "empty_query", normalized

        signature = json.dumps(
            {"name": normalized.name, "arguments": normalized.arguments},
            ensure_ascii=False,
            sort_keys=True,
        )
        repeated = self.call_signatures.count(signature)
        if repeated >= self.config.max_same_tool_calls:
            return False, "duplicate_tool_call", normalized
        return True, "allowed", normalized

    def execute_tool(
        self,
        request: ToolCallRequest,
        executor: Callable[[str, dict], ToolResult],
    ) -> tuple[ToolResult | None, str]:
        allowed, reason, normalized = self.validate_tool_call(request)
        self.add_trace(
            "tool_guardrail",
            allowed=allowed,
            reason=reason,
            tool_name=normalized.name,
            arguments=normalized.arguments,
        )
        if not allowed:
            return None, reason

        signature = json.dumps(
            {"name": normalized.name, "arguments": normalized.arguments},
            ensure_ascii=False,
            sort_keys=True,
        )
        self.call_signatures.append(signature)
        self.tool_call_count += 1
        try:
            result = executor(normalized.name, normalized.arguments)
        except Exception as exc:
            self.add_trace(
                "tool_error",
                tool_name=normalized.name,
                error=type(exc).__name__,
                message=str(exc)[:300],
            )
            return None, "tool_error"

        self.add_trace(
            "tool_observation",
            tool_name=result.name,
            arguments=normalized.arguments,
            result_preview=result.content[:1000],
            debug=result.debug or {},
        )
        return result, "completed"

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tool_call_count": self.tool_call_count,
            "rewrite_count": self.rewrite_count,
            "elapsed_ms": round(self.elapsed_seconds() * 1000, 2),
            "config": asdict(self.config),
        }
