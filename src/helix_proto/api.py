from __future__ import annotations

import json
import mimetypes
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from helix_proto.agent import AgentRunner
from helix_proto.hf import GPT2StreamingEngine
from helix_proto.text import decode_tokens, encode_text, render_messages_prompt
from helix_proto.tools import build_runtime_tool_registry
from helix_proto.workspace import (
    list_model_workspaces,
    model_session_dir,
    prepare_model_workspace,
    resolve_export_dir,
    resolve_model_info,
    resolve_tokenizer_dir,
    slugify,
    workspace_root,
)


def _session_id(prefix: str = "session") -> str:
    return f"{slugify(prefix)}-{int(time.time())}"


def _web_root() -> Path:
    return Path(__file__).resolve().parents[2] / "web"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = dict(value)
        last_logits = cleaned.pop("last_logits", None)
        if isinstance(last_logits, np.ndarray):
            cleaned["last_logits_shape"] = list(last_logits.shape)
        return {key: _json_ready(item) for key, item in cleaned.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cors_origins() -> list[str]:
    raw = os.environ.get("HELIX_CORS_ORIGINS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _allowed_origin(request_origin: str | None) -> str | None:
    if not request_origin:
        return None
    configured = _cors_origins()
    if not configured:
        return request_origin
    if "*" in configured:
        return "*"
    if request_origin in configured:
        return request_origin
    return None


class HelixRuntime:
    def __init__(self, *, root: str | Path | None = None) -> None:
        self.root = workspace_root(root)
        self._engines: dict[tuple[str, str], GPT2StreamingEngine] = {}
        self._tools = None

    def list_models(self) -> list[dict[str, Any]]:
        return list_model_workspaces(self.root)

    def model_info(self, alias: str) -> dict[str, Any]:
        return resolve_model_info(alias, self.root)

    def prepare_model(
        self,
        *,
        model_ref: str,
        alias: str | None = None,
        block_rows: int = 256,
        trust_remote_code: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        return prepare_model_workspace(
            model_ref=model_ref,
            alias=alias,
            root=self.root,
            block_rows=block_rows,
            trust_remote_code=trust_remote_code,
            force=force,
        )

    def _engine(self, alias: str, *, cache_mode: str) -> GPT2StreamingEngine:
        key = (slugify(alias), cache_mode)
        engine = self._engines.get(key)
        if engine is None:
            export_dir = resolve_export_dir(alias, self.root)
            engine = GPT2StreamingEngine(export_dir, cache_mode=cache_mode)
            self._engines[key] = engine
        return engine

    def _tokenizer_dir(self, alias: str) -> Path:
        return resolve_tokenizer_dir(alias, self.root)

    def generate(
        self,
        *,
        alias: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        engine = self._engine(alias, cache_mode=cache_mode)
        result = engine.generate_advanced(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            seed=seed,
        )
        if session_id is not None:
            session_dir = model_session_dir(alias, session_id, self.root)
            engine.save_session(
                session_dir,
                generated_ids=result["generated_ids"],
                last_logits=result.get("last_logits"),
            )
            result["session_id"] = slugify(session_id)
            result["session_dir"] = str(session_dir)
        return result

    def generate_text(
        self,
        *,
        alias: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        session_id: str | None = None,
        add_special_tokens: bool = False,
        skip_special_tokens: bool = True,
    ) -> dict[str, Any]:
        if messages:
            prompt_text = render_messages_prompt(messages)
        elif prompt is not None:
            prompt_text = prompt
        else:
            raise ValueError("prompt or messages is required")

        tokenizer_dir = self._tokenizer_dir(alias)
        prompt_ids = encode_text(
            tokenizer_dir,
            prompt_text,
            add_special_tokens=add_special_tokens,
        )
        if not prompt_ids:
            raise ValueError("tokenizer produced no prompt tokens")
        result = self.generate(
            alias=alias,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            cache_mode=cache_mode,
            session_id=session_id,
        )
        new_ids = result["new_ids"]
        result["prompt_text"] = prompt_text
        result["prompt_ids"] = prompt_ids
        result["completion_text"] = decode_tokens(
            tokenizer_dir,
            new_ids,
            skip_special_tokens=skip_special_tokens,
        )
        result["generated_text"] = decode_tokens(
            tokenizer_dir,
            result["generated_ids"],
            skip_special_tokens=skip_special_tokens,
        )
        if messages:
            result["messages"] = messages
        return result

    def stream_text(
        self,
        *,
        alias: str,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        session_id: str | None = None,
        add_special_tokens: bool = False,
        skip_special_tokens: bool = True,
    ):
        if messages:
            prompt_text = render_messages_prompt(messages)
        elif prompt is not None:
            prompt_text = prompt
        else:
            raise ValueError("prompt or messages is required")

        tokenizer_dir = self._tokenizer_dir(alias)
        prompt_ids = encode_text(
            tokenizer_dir,
            prompt_text,
            add_special_tokens=add_special_tokens,
        )
        if not prompt_ids:
            raise ValueError("tokenizer produced no prompt tokens")

        engine = self._engine(alias, cache_mode=cache_mode)
        prompt_len = len(prompt_ids)
        last_logits = None
        current_generated = list(prompt_ids)

        yield {
            "event": "start",
            "alias": alias,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "session_id": slugify(session_id) if session_id else None,
        }

        for event in engine.stream_generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            seed=seed,
        ):
            last_logits = event["last_logits"]
            if event["phase"] != "generated":
                continue
            generated_ids = event["generated_ids"]
            current_generated = list(generated_ids)
            new_ids = generated_ids[prompt_len:]
            token_id = int(event["token_id"])
            token_text = decode_tokens(
                tokenizer_dir,
                [token_id],
                skip_special_tokens=skip_special_tokens,
            )
            yield {
                "event": "token",
                "token_id": token_id,
                "token_text": token_text,
                "generated_ids": generated_ids,
                "new_ids": new_ids,
                "completion_text": decode_tokens(
                    tokenizer_dir,
                    new_ids,
                    skip_special_tokens=skip_special_tokens,
                ),
                "generated_text": decode_tokens(
                    tokenizer_dir,
                    generated_ids,
                    skip_special_tokens=skip_special_tokens,
                ),
            }

        final_generated = current_generated
        final_new_ids = final_generated[prompt_len:]
        if session_id is not None:
            session_dir = model_session_dir(alias, session_id, self.root)
            engine.save_session(
                session_dir,
                generated_ids=final_generated,
                last_logits=last_logits,
            )
        yield {
            "event": "done",
            "alias": alias,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "generated_ids": final_generated,
            "new_ids": final_new_ids,
            "completion_text": decode_tokens(
                tokenizer_dir,
                final_new_ids,
                skip_special_tokens=skip_special_tokens,
            ),
            "generated_text": decode_tokens(
                tokenizer_dir,
                final_generated,
                skip_special_tokens=skip_special_tokens,
            ),
            "session_id": slugify(session_id) if session_id else None,
        }

    def resume(
        self,
        *,
        alias: str,
        session_id: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        save_session: bool = True,
    ) -> dict[str, Any]:
        session_dir = model_session_dir(alias, session_id, self.root)
        engine = self._engine(alias, cache_mode=cache_mode)
        result = engine.resume_advanced(
            session_dir,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            top_p=top_p if do_sample else 1.0,
            seed=seed,
        )
        if save_session:
            engine.save_session(
                session_dir,
                generated_ids=result["generated_ids"],
                last_logits=result.get("last_logits"),
            )
        result["session_id"] = slugify(session_id)
        result["session_dir"] = str(session_dir)
        return result

    def resume_text(
        self,
        *,
        alias: str,
        session_id: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        cache_mode: str = "session",
        save_session: bool = True,
        skip_special_tokens: bool = True,
    ) -> dict[str, Any]:
        tokenizer_dir = self._tokenizer_dir(alias)
        result = self.resume(
            alias=alias,
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            cache_mode=cache_mode,
            save_session=save_session,
        )
        result["completion_text"] = decode_tokens(
            tokenizer_dir,
            result["new_ids"],
            skip_special_tokens=skip_special_tokens,
        )
        result["generated_text"] = decode_tokens(
            tokenizer_dir,
            result["generated_ids"],
            skip_special_tokens=skip_special_tokens,
        )
        return result

    def tool_manifest(self) -> list[dict[str, Any]]:
        if self._tools is None:
            self._tools = build_runtime_tool_registry(self)
        return self._tools.manifest()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._tools is None:
            self._tools = build_runtime_tool_registry(self)
        return self._tools.call(name, arguments)

    def agent_runner(self) -> AgentRunner:
        return AgentRunner(self, root=self.root)


class _HelixHandler(BaseHTTPRequestHandler):
    runtime: HelixRuntime

    def _apply_cors_headers(self) -> None:
        allowed_origin = _allowed_origin(self.headers.get("Origin"))
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_json_ready(payload), indent=2).encode("utf-8")
        self.send_response(status)
        self._apply_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body: bytes, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self._apply_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._apply_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_sse_event(self, name: str, payload: Any) -> None:
        body = f"event: {name}\ndata: {json.dumps(_json_ready(payload), ensure_ascii=True)}\n\n".encode(
            "utf-8"
        )
        self.wfile.write(body)
        self.wfile.flush()

    def _send_error(self, message: str, *, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message}, status=status)

    def _serve_static(self, relative_path: str) -> None:
        path = (_web_root() / relative_path).resolve()
        if not str(path).startswith(str(_web_root().resolve())) or not path.exists():
            self._send_error("route not found", status=HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(path.name)
        self._send_text(
            path.read_bytes(),
            content_type=content_type or "application/octet-stream",
        )

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._apply_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._serve_static("index.html")
                return
            if path == "/app":
                self._serve_static("app.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path.removeprefix("/static/"))
                return
            if path == "/health":
                self._send_json({"status": "ok", "workspace_root": str(self.runtime.root)})
                return
            if path == "/models":
                self._send_json({"models": self.runtime.list_models()})
                return
            if path == "/tools":
                self._send_json({"tools": self.runtime.tool_manifest()})
                return
            if path == "/agent/knowledge/search":
                params = parse_qs(urlparse(self.path).query)
                agent_name = params.get("agent_name", ["default-agent"])[0]
                goal = params.get("query", [""])[0]
                top_k = int(params.get("top_k", ["5"])[0])
                self._send_json(self.runtime.agent_runner().search_knowledge(agent_name, goal, top_k=top_k))
                return
            if path.startswith("/models/"):
                alias = path.removeprefix("/models/")
                self._send_json(self.runtime.model_info(alias))
                return
            self._send_error("route not found", status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_error(str(exc), status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_error(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json()
        try:
            if path == "/prepare":
                self._send_json(
                    self.runtime.prepare_model(
                        model_ref=body["model_ref"],
                        alias=body.get("alias"),
                        block_rows=int(body.get("block_rows", 256)),
                        trust_remote_code=bool(body.get("trust_remote_code", False)),
                        force=bool(body.get("force", False)),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/generate":
                save_session = bool(body.get("save_session", False))
                session_id = body.get("session_id")
                if save_session and not session_id:
                    session_id = _session_id(body["alias"])
                self._send_json(
                    self.runtime.generate(
                        alias=body["alias"],
                        prompt_ids=[int(item) for item in body["prompt_ids"]],
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        session_id=session_id,
                    )
                )
                return
            if path == "/chat":
                save_session = bool(body.get("save_session", False))
                session_id = body.get("session_id")
                if save_session and not session_id:
                    session_id = _session_id(body["alias"])
                self._send_json(
                    self.runtime.generate_text(
                        alias=body["alias"],
                        prompt=body.get("prompt"),
                        messages=body.get("messages"),
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        session_id=session_id,
                        add_special_tokens=bool(body.get("add_special_tokens", False)),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    )
                )
                return
            if path == "/chat/stream":
                save_session = bool(body.get("save_session", False))
                session_id = body.get("session_id")
                if save_session and not session_id:
                    session_id = _session_id(body["alias"])
                self._send_sse_headers()
                try:
                    for item in self.runtime.stream_text(
                        alias=body["alias"],
                        prompt=body.get("prompt"),
                        messages=body.get("messages"),
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        session_id=session_id,
                        add_special_tokens=bool(body.get("add_special_tokens", False)),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    ):
                        event_name = str(item.get("event", "message"))
                        self._write_sse_event(event_name, item)
                except Exception as exc:  # noqa: BLE001
                    self._write_sse_event("error", {"error": str(exc)})
                self.close_connection = True
                return
            if path == "/resume":
                self._send_json(
                    self.runtime.resume(
                        alias=body["alias"],
                        session_id=body["session_id"],
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        save_session=bool(body.get("save_session", True)),
                    )
                )
                return
            if path == "/chat/resume":
                self._send_json(
                    self.runtime.resume_text(
                        alias=body["alias"],
                        session_id=body["session_id"],
                        max_new_tokens=int(body.get("max_new_tokens", 1)),
                        do_sample=bool(body.get("do_sample", False)),
                        temperature=float(body.get("temperature", 1.0)),
                        top_k=int(body.get("top_k", 0)),
                        top_p=float(body.get("top_p", 1.0)),
                        seed=None if body.get("seed") is None else int(body["seed"]),
                        cache_mode=str(body.get("cache_mode", "session")),
                        save_session=bool(body.get("save_session", True)),
                        skip_special_tokens=bool(body.get("skip_special_tokens", True)),
                    )
                )
                return
            if path == "/agent/knowledge/add-text":
                self._send_json(
                    self.runtime.agent_runner().add_knowledge_text(
                        body.get("agent_name", "default-agent"),
                        body["text"],
                        source=body.get("source", "inline-text"),
                        metadata=body.get("metadata"),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/agent/knowledge/add-file":
                self._send_json(
                    self.runtime.agent_runner().add_knowledge_file(
                        body.get("agent_name", "default-agent"),
                        body["file_path"],
                        metadata=body.get("metadata"),
                    ),
                    status=HTTPStatus.CREATED,
                )
                return
            if path == "/agent/memory/search":
                self._send_json(
                    self.runtime.agent_runner().search_memory(
                        body.get("agent_name", "default-agent"),
                        body["query"],
                        top_k=int(body.get("top_k", 5)),
                    )
                )
                return
            if path == "/agent/run":
                self._send_json(
                    self.runtime.agent_runner().run(
                        goal=body["goal"],
                        agent_name=body.get("agent_name", "default-agent"),
                        default_model_alias=body.get("default_model_alias"),
                        local_planner_alias=body.get("local_planner_alias"),
                        remote_model_ref=body.get("remote_model_ref"),
                        prefer_remote=bool(body.get("prefer_remote", False)),
                        trust_remote_code=bool(body.get("trust_remote_code", False)),
                        max_steps=int(body.get("max_steps", 4)),
                        generation_max_new_tokens=int(body.get("generation_max_new_tokens", 128)),
                    )
                )
                return
            if path == "/agent/run/stream":
                self._send_sse_headers()
                try:
                    for item in self.runtime.agent_runner().run_stream(
                        goal=body["goal"],
                        agent_name=body.get("agent_name", "default-agent"),
                        default_model_alias=body.get("default_model_alias"),
                        local_planner_alias=body.get("local_planner_alias"),
                        remote_model_ref=body.get("remote_model_ref"),
                        prefer_remote=bool(body.get("prefer_remote", False)),
                        trust_remote_code=bool(body.get("trust_remote_code", False)),
                        max_steps=int(body.get("max_steps", 4)),
                        generation_max_new_tokens=int(body.get("generation_max_new_tokens", 128)),
                    ):
                        event_name = str(item.get("event", "message"))
                        self._write_sse_event(event_name, item)
                except Exception as exc:  # noqa: BLE001
                    self._write_sse_event("error", {"error": str(exc)})
                self.close_connection = True
                return
            if path.startswith("/tools/"):
                tool_name = path.removeprefix("/tools/")
                self._send_json(self.runtime.call_tool(tool_name, body))
                return
            self._send_error("route not found", status=HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_error(f"missing field: {exc.args[0]}")
        except FileNotFoundError as exc:
            self._send_error(str(exc), status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_error(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def create_api_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    root: str | Path | None = None,
) -> ThreadingHTTPServer:
    runtime = HelixRuntime(root=root)
    handler = type("HelixHandler", (_HelixHandler,), {"runtime": runtime})
    return ThreadingHTTPServer((host, port), handler)


def serve_api(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    root: str | Path | None = None,
) -> None:
    server = create_api_server(host=host, port=port, root=root)
    try:
        server.serve_forever()
    finally:
        server.server_close()
