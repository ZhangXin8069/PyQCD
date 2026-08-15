from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


def _load_dotenv(dotenv_path: str | Path = ".env") -> None:
    p = Path(dotenv_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


class LQCDLLMClient:
    """OpenAI-compatible client with .env auto-loading.

    Env vars:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (default https://api.gpugeek.com/v1)
    - OPENAI_MODEL (default Vendor2/GPT-5.2)
    - OPENAI_MAX_RETRIES (default 3)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        dotenv_path: str | Path = ".env",
        extra_body: dict[str, Any] | None = None,
    ):
        _load_dotenv(dotenv_path)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.gpugeek.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "Vendor2/GPT-5.2")
        if max_retries is None:
            max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self._client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=max_retries)
            if self.api_key else None
        )
        # extra_body for provider-specific params (e.g. DeepSeek thinking mode)
        self.extra_body = extra_body or self._parse_extra_body()

    def _parse_extra_body(self) -> dict[str, Any]:
        raw = os.getenv("OPENAI_EXTRA_BODY", "")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_content(msg: Any) -> str:
        """Extract message content, handling DeepSeek reasoning mode.

        DeepSeek with thinking enabled puts content in `reasoning_content`
        and may leave `content` as None. Fall back to reasoning_content,
        then to empty string.
        """
        if hasattr(msg, "content") and msg.content:
            return msg.content
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            return msg.reasoning_content
        return ""

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        if not self._client:
            return ""
        req: dict[str, Any] = {
            "model": model or self.model,
            "stream": stream,
            "messages": messages,
        }
        if temperature is not None:
            req["temperature"] = temperature
        if max_tokens is not None:
            req["max_tokens"] = max_tokens

        if self.extra_body:
            req["extra_body"] = self.extra_body

        if stream:
            try:
                stream_resp = self._client.chat.completions.create(**req)
            except Exception:
                return ""
            chunks: list[str] = []
            for chunk in stream_resp:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                # DeepSeek reasoning models: content may be None,
                # actual response in reasoning_content
                if hasattr(delta, "content") and delta.content:
                    chunks.append(delta.content)
                elif hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    chunks.append(delta.reasoning_content)
            return "".join(chunks)

        try:
            resp = self._client.chat.completions.create(**req)
        except Exception:
            return ""

        if not isinstance(resp, str) and not getattr(resp, "choices", None):
            return ""
        # Defend against custom API endpoints that return a raw string
        # instead of a parsed ChatCompletion object.
        if isinstance(resp, str):
            return resp
        return self._extract_content(resp.choices[0].message)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        text = self.chat(
            messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        ).strip()
        if not text:
            return {}

        text = self._strip_code_fence(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            extracted = self._extract_json_obj(text)
            if extracted is not None:
                return extracted
            return {"raw_text": text}

    def _strip_code_fence(self, text: str) -> str:
        if not text.startswith("```"):
            return text
        out = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        out = re.sub(r"\s*```$", "", out)
        return out.strip()

    def _extract_json_obj(self, text: str) -> dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        snippet = text[start : end + 1]
        try:
            payload = json.loads(snippet)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    def chat_json_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_handler: Any,
        max_turns: int = 8,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run multi-turn chat with tool-calling until final JSON is produced.

        Returns:
        - final_payload: parsed json object or {"raw_text": "..."}
        - tool_trace: sequence of executed tool calls and results
        """
        if not self._client:
            return {}, []

        conversation: list[dict[str, Any]] = list(messages)
        tool_trace: list[dict[str, Any]] = []

        for _ in range(max_turns):
            req: dict[str, Any] = {
                "model": model or self.model,
                "messages": conversation,
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
            }
            if temperature is not None:
                req["temperature"] = temperature
            if max_tokens is not None:
                req["max_tokens"] = max_tokens
            if self.extra_body:
                req["extra_body"] = self.extra_body

            try:
                resp = self._client.chat.completions.create(**req)
            except Exception:
                return {}, tool_trace
            if isinstance(resp, str):
                return {"raw_text": resp}, tool_trace
            msg = resp.choices[0].message if getattr(resp, "choices", None) else None
            if msg is None:
                return {}, tool_trace

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": self._extract_content(msg),
            }
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                assistant_msg["reasoning_content"] = msg.reasoning_content
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                assistant_msg["tool_calls"] = []
                for tc in tool_calls:
                    fn_name = tc.function.name if tc.function else ""
                    fn_args_str = tc.function.arguments if tc.function else "{}"
                    assistant_msg["tool_calls"].append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args_str,
                            },
                        }
                    )
            conversation.append(assistant_msg)

            if not tool_calls:
                text = self._extract_content(msg).strip()
                if not text:
                    return {}, tool_trace
                text = self._strip_code_fence(text)
                try:
                    out = json.loads(text)
                    if isinstance(out, dict):
                        return out, tool_trace
                except json.JSONDecodeError:
                    extracted = self._extract_json_obj(text)
                    if extracted is not None:
                        return extracted, tool_trace
                    return {"raw_text": text}, tool_trace

            for tc in tool_calls:
                fn_name = tc.function.name if tc.function else ""
                fn_args_str = tc.function.arguments if tc.function else "{}"
                try:
                    fn_args = json.loads(fn_args_str) if fn_args_str else {}
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                except json.JSONDecodeError:
                    fn_args = {}

                try:
                    result = tool_handler(fn_name, fn_args)
                except Exception as e:
                    result = {"error": str(e), "tool": fn_name}

                tool_trace.append(
                    {
                        "tool_name": fn_name,
                        "tool_args": fn_args,
                        "tool_result": result,
                    }
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return {"error": "max_turns_exceeded", "raw_text": ""}, tool_trace
