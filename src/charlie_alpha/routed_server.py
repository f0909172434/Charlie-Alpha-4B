from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .config import load_config
from .routed_inference import generate_routed, load_routed_model


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve(config_path: str, host: str, port: int) -> None:
    config = load_config(config_path)
    model, tokenizer, router = load_routed_model(config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"Charlie alpha API: {format % args}")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json(
                    self,
                    200,
                    {
                        "status": "ok",
                        "model": "Charlie-Alpha-4B",
                        "architecture": "FORGE dynamic sparse LoRA",
                        "loaded_models": 1,
                        "lora_modules": router.module_count,
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
                route = str(
                    request.get("charlie_route")
                    or self.headers.get("X-Charlie-Route")
                    or "auto"
                )
                max_tokens = min(max(int(request.get("max_tokens", 1024)), 1), 2048)
                temperature = min(max(float(request.get("temperature", 0.2)), 0.0), 2.0)
                top_p = min(max(float(request.get("top_p", 0.8)), 0.0), 1.0)
                answer, decision = generate_routed(
                    model,
                    tokenizer,
                    router,
                    messages,
                    route=route,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
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
                            "prompt_tokens": len(tokenizer.encode(prompt)),
                            "completion_tokens": len(tokenizer.encode(answer)),
                        },
                        "charlie_route": {
                            "selected": decision.route,
                            "reason": decision.reason,
                        },
                    },
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                _json(self, 400, {"error": {"message": str(error)}})
            except Exception as error:  # pragma: no cover - defensive API boundary
                _json(self, 500, {"error": {"message": f"generation failed: {error}"}})

    print(f"Charlie alpha routed API listening on http://{host}:{port}")
    server = HTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Charlie alpha routed local API")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
