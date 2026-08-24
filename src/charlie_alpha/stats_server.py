from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .config import load_config
from .stats_agent import StatsAgent, classify_stats_route


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _user_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    )


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("messages must contain a user message")


def _file_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("charlie_files must be a list of local paths")
    paths: list[Path] = []
    for item in value:
        if isinstance(item, str):
            paths.append(Path(item))
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            paths.append(Path(item["path"]))
        else:
            raise ValueError("each charlie_files item must be a path string or {path: ...}")
    return paths


def serve(config_path: str, host: str, port: int, adapter_path: str | None = None) -> None:
    config = load_config(config_path)
    agent = StatsAgent(config, adapter_path=adapter_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"Charlie alpha stats API: {format % args}")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json(
                    self,
                    200,
                    {
                        "status": "ok",
                        "model": "Charlie-Alpha-4B",
                        "profile": "DGP-Regret statistical adapter",
                        "loaded_models": 1,
                        "routes": ["auto", "base", "stats"],
                        "loopback_default": True,
                    },
                )
            elif self.path == "/v1/models":
                _json(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "Charlie-Alpha-4B",
                                "object": "model",
                                "owned_by": "f0909172434",
                            }
                        ],
                    },
                )
            else:
                _json(self, 404, {"error": {"message": "Not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                _json(self, 404, {"error": {"message": "Not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                messages = request.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
                if request.get("stream"):
                    raise ValueError("streaming is not supported by this local server")
                files = _file_paths(request.get("charlie_files"))
                route = classify_stats_route(
                    _user_text(messages),
                    has_files=bool(files),
                    override=str(
                        request.get("charlie_route")
                        or self.headers.get("X-Charlie-Route")
                        or "auto"
                    ),
                )
                max_tokens = min(max(int(request.get("max_tokens", 1024)), 1), 2048)
                temperature = min(max(float(request.get("temperature", 0.0)), 0.0), 2.0)
                top_p = min(max(float(request.get("top_p", 1.0)), 0.0), 1.0)
                tools_enabled = request.get("charlie_tools", True) is not False
                if route == "stats" and files and tools_enabled:
                    latest_question = _latest_user_text(messages)
                    result = agent.analyze(
                        data_paths=files,
                        question=latest_question,
                        language=str(request.get("charlie_language", "auto")),
                        conversation=messages[:-1],
                    )
                    answer = result["answer"]
                    plan = result["analysis_plan"]
                    tool_calls = int(result["tool_calls"])
                    isolation = result["isolation"]
                else:
                    answer = agent.answer_without_tools(
                        messages,
                        route=route,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    plan = None
                    tool_calls = 0
                    isolation = {
                        "sandboxed": False,
                        "reason": "no statistical tool was invoked",
                        "data_local_only": True,
                    }
                prompt = agent.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                _json(
                    self,
                    200,
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "Charlie-Alpha-4B",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(agent.tokenizer.encode(prompt)),
                            "completion_tokens": len(agent.tokenizer.encode(answer)),
                        },
                        "charlie_route": {"selected": route},
                        "charlie_tool_calls": tool_calls,
                        "charlie_isolation": isolation,
                        "charlie_analysis_plan": plan,
                    },
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                _json(self, 400, {"error": {"message": str(error)}})
            except Exception as error:  # pragma: no cover - defensive API boundary
                _json(self, 500, {"error": {"message": f"request failed: {error}"}})

    print(f"Charlie alpha stats API listening on http://{host}:{port}")
    server = HTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Charlie alpha local statistical API")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--adapter-path")
    arguments = parser.parse_args()
    serve(arguments.config, arguments.host, arguments.port, arguments.adapter_path)


if __name__ == "__main__":
    main()
