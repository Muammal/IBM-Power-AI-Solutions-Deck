"""llm_client.py: one tiny client for the OpenAI-compatible chat API that llama-server
(and vLLM, Ollama, watsonx.ai on-prem gateways) expose. Only depends on httpx.

    from llm_client import LLM
    llm = LLM("http://127.0.0.1:8080/v1")
    text = llm.chat([{"role": "user", "content": "مرحبا"}])
    data = llm.chat(messages, json_schema=MY_SCHEMA)          # -> dict, validated by the server grammar
    msg  = llm.chat(messages, tools=TOOLS, raw=True)          # -> full assistant message (tool_calls)

Granite 4.2 is a reasoning model: thinking is ON by default in its chat template. We turn it OFF
per request (chat_template_kwargs.enable_thinking=False) unless thinking=True is passed, because
on CPU every hidden reasoning token costs time. llama-server must run with --jinja for this.
"""
import json
import os
import re

import httpx

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLM:
    def __init__(self, base_url=None, model=None, timeout=600.0, api_key=None):
        self.base_url = (base_url or os.getenv("LLM_URL", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "local")  # llama-server ignores the name
        headers = {"Authorization": f"Bearer {api_key or os.getenv('LLM_API_KEY', 'none')}"}
        self.http = httpx.Client(timeout=timeout, headers=headers)

    @staticmethod
    def strip_thinking(text):
        """Remove <think>...</think> reasoning so users and logs see only the answer."""
        return _THINK.sub("", text or "").strip()

    def health(self):
        """True when the server answers. llama-server: GET /health on the root URL."""
        root = self.base_url.rsplit("/v1", 1)[0]
        try:
            return self.http.get(f"{root}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, messages, *, temperature=0.2, max_tokens=1024, json_schema=None, tools=None,
             thinking=False, raw=False, **extra):
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": bool(thinking)},
                **extra}
        if json_schema:
            # Constrained decoding: the server turns the schema into a grammar, so the output parses.
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": json_schema}}
        if tools:
            body["tools"] = tools
        r = self.http.post(f"{self.base_url}/chat/completions", json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        if raw:
            return msg
        text = _THINK.sub("", msg.get("content") or "").strip()
        return json.loads(text) if json_schema else text
