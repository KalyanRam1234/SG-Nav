from __future__ import annotations

import difflib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


@dataclass
class LLMConfig:
    backend: str = "ollama"  # 'ollama' (python pkg) or 'http'
    # Match README / SG_Nav defaults
    model: str = "llama3.2-vision:latest"
    temperature: float = 0.2
    # HTTP endpoint used for backend='http'. For backend='ollama', we also use this
    # to infer the host for the python client (scheme://host:port).
    url: str = "http://localhost:11434/api/chat"
    # Prevent indefinite hangs (ollama python client defaults to no timeout)
    timeout_s: float = 600.0
    # Cap output length to avoid very long generations (tune per qa_count)
    num_predict: Optional[int] = 1536
    # Keep model loaded between calls (e.g. '10m', 600). Optional.
    keep_alive: Optional[Union[float, str]] = None
    # Stream response chunks; useful for progress indication.
    stream: bool = False
    # If true, print progress dots to stderr while streaming.
    show_progress: bool = False


def _ollama_host_from_url(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}"
    except Exception:
        return None
    return None


def _ollama_installed_models(client: Any) -> List[str]:
    try:
        resp = client.list()

        models: Any = getattr(resp, "models", None)
        if models is None and isinstance(resp, dict):
            models = resp.get("models")
        models = models or []

        out: List[str] = []
        for m in models:
            name: Any = getattr(m, "model", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if isinstance(name, str) and name:
                out.append(name)
        return out
    except Exception:
        return []


def _resolve_model_name(requested: str, available: List[str]) -> str:
    if not available:
        return requested

    if requested in available:
        return requested

    if ":" not in requested:
        latest = f"{requested}:latest"
        if latest in available:
            return latest

    return requested


def _model_not_found_help(requested: str, available: List[str]) -> str:
    lines = [f"Ollama model '{requested}' not found."]

    if available:
        close = [m for m in available if requested.split(":")[0] in m]
        if not close:
            close = difflib.get_close_matches(requested, available, n=5, cutoff=0.2)
        if close:
            lines.append("Available models that look relevant: " + ", ".join(close))
        lines.append("(See full list with: ollama list)")

    lines.append(f"To download it: ollama pull {requested}")
    lines.append("Or pass an installed model via --llm_model (e.g. llama3.2-vision:latest)")
    return "\n".join(lines)


def chat(
    prompt: str,
    cfg: Optional[LLMConfig] = None,
    *,
    json_mode: bool = False,
    images: Optional[List[str]] = None,
) -> str:
    cfg = cfg or LLMConfig()

    options: Dict[str, Any] = {"temperature": float(cfg.temperature)}
    if cfg.num_predict is not None:
        options["num_predict"] = int(cfg.num_predict)

    if cfg.backend == "ollama":
        try:
            import ollama  # type: ignore
        except Exception:
            # Fall back to HTTP if the python pkg isn't available
            return chat(
                prompt,
                LLMConfig(**{**cfg.__dict__, "backend": "http"}),
                json_mode=json_mode,
                images=images,
            )

        host = _ollama_host_from_url(str(cfg.url))
        # Critical: enforce a timeout (ollama python client defaults to timeout=None)
        client = ollama.Client(host=host, timeout=float(cfg.timeout_s))

        available = _ollama_installed_models(client)
        model = _resolve_model_name(str(cfg.model), available)
        if available and model not in available:
            raise RuntimeError(_model_not_found_help(str(cfg.model), available))

        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images

        stream = bool(getattr(cfg, "stream", False) or getattr(cfg, "show_progress", False))

        try:
            if stream:
                chunks: List[str] = []
                for part in client.chat(
                    model=model,
                    messages=[msg],
                    options=options,
                    format="json" if json_mode else None,
                    keep_alive=cfg.keep_alive,
                    stream=True,
                ):
                    msgp: Any = getattr(part, "message", None)
                    if msgp is None and isinstance(part, dict):
                        msgp = part.get("message")
                    delta: Any = getattr(msgp, "content", None) if msgp is not None else None
                    if delta is None and isinstance(msgp, dict):
                        delta = msgp.get("content")

                    if isinstance(delta, str) and delta:
                        chunks.append(delta)
                    if getattr(cfg, "show_progress", False):
                        sys.stderr.write(".")
                        sys.stderr.flush()
                if getattr(cfg, "show_progress", False):
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                return "".join(chunks)

            resp = client.chat(
                model=model,
                messages=[msg],
                options=options,
                format="json" if json_mode else None,
                keep_alive=cfg.keep_alive,
                stream=False,
            )
        except Exception as e:
            # If it's a 404, provide a helpful message with installed models
            status = getattr(e, "status_code", None)
            if status == 404:
                raise RuntimeError(_model_not_found_help(str(cfg.model), available)) from e
            raise

        msgp: Any = getattr(resp, "message", None)
        if msgp is None and isinstance(resp, dict):
            msgp = resp.get("message")
        content: Any = getattr(msgp, "content", None) if msgp is not None else None
        if content is None and isinstance(msgp, dict):
            content = msgp.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected Ollama response shape (python backend)")
        return content

    if cfg.backend != "http":
        raise ValueError(f"Unknown LLM backend: {cfg.backend}")

    # Ollama HTTP API (chat)
    msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images

    stream = bool(getattr(cfg, "stream", False) or getattr(cfg, "show_progress", False))

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [msg],
        "stream": bool(stream),
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"
    if cfg.keep_alive is not None:
        payload["keep_alive"] = cfg.keep_alive

    req = urllib.request.Request(
        cfg.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        stream = bool(getattr(cfg, "stream", False) or getattr(cfg, "show_progress", False))
        if stream:
            chunks: List[str] = []
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
                for raw_line in r:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    part = json.loads(line)
                    if isinstance(part, dict) and part.get("error"):
                        raise RuntimeError(str(part.get("error")))
                    msgp = part.get("message") if isinstance(part, dict) else None
                    if isinstance(msgp, dict):
                        delta = msgp.get("content")
                        if isinstance(delta, str) and delta:
                            chunks.append(delta)
                    if getattr(cfg, "show_progress", False):
                        sys.stderr.write(".")
                        sys.stderr.flush()
                    if isinstance(part, dict) and part.get("done") is True:
                        break
            if getattr(cfg, "show_progress", False):
                sys.stderr.write("\n")
                sys.stderr.flush()
            content = "".join(chunks)
            if not isinstance(content, str):
                raise RuntimeError("Unexpected streamed LLM response")
            return content

        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        body = e.read().decode("utf-8", errors="ignore")
        if e.code == 404 and "model" in body.lower() and "not found" in body.lower():
            raise RuntimeError(
                f"Ollama model '{cfg.model}' not found (HTTP backend). Try: ollama pull {cfg.model}"
            ) from e
        raise

    # Ollama chat returns: { message: { content: ... } }
    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected LLM response shape: {list(data.keys())}")
    return content
