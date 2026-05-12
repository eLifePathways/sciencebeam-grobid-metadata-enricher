from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .telemetry import get_tracer

DEFAULT_POOL_PATH = Path(os.getenv("AOAI_POOL_PATH", "aoai_pool.json"))
DEFAULT_GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "60"))
DEFAULT_PDFALTO_BIN = Path(os.getenv("PDFALTO_BIN", "pdfalto"))
# Upstream PDF parser: "grobid" (default, lfoppiano/grobid:0.9.0-crf compatible)
# or "sciencebeam" (eLifePathways/sciencebeam-parser). Both expose
# /api/processFulltextDocument and /api/isalive on the same port; the
# difference is which form fields they honour and whether they require an
# explicit Accept header to negotiate the response media type.
PARSER_GROBID = "grobid"
PARSER_SCIENCEBEAM = "sciencebeam"
SUPPORTED_PARSERS = (PARSER_GROBID, PARSER_SCIENCEBEAM)
DEFAULT_PARSER = os.getenv("PARSER", PARSER_GROBID)


class LLMCallError(RuntimeError):
    pass


def _read_error_body(error: urllib.error.HTTPError, limit: int = 500) -> str:
    try:
        raw = error.read().decode("utf-8", errors="replace").strip()
    except Exception:  # pylint: disable=broad-except
        return "<no body>"
    return raw[:limit] if raw else "<empty body>"


# Per-parser default URLs. Picked so `--parser sciencebeam` works on its
# own without also having to pass `--grobid-url http://localhost:8071/api`,
# which is the port the compose `sciencebeam-parser` service publishes.
DEFAULT_PARSER_URLS = {
    PARSER_GROBID: "http://localhost:8070/api",
    PARSER_SCIENCEBEAM: "http://localhost:8071/api",
}


def resolve_parser_url(parser: str = DEFAULT_PARSER, override: Optional[str] = None) -> str:
    """Return the URL for the given parser. Precedence: explicit override
    argument > GROBID_URL env var > per-parser default. The GROBID_URL env
    name is parser-agnostic (kept that way for compose.yml + CI back-compat)
    so it overrides regardless of the parser choice."""
    if override:
        return override
    env_url = os.getenv("GROBID_URL")
    if env_url:
        return env_url
    return DEFAULT_PARSER_URLS.get(parser, DEFAULT_PARSER_URLS[PARSER_GROBID])


# Resolved at import time for back-compat with callers that read
# DEFAULT_GROBID_URL directly (api.py, PipelineSettings default field).
DEFAULT_GROBID_URL = resolve_parser_url(DEFAULT_PARSER)
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL")
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass(frozen=True)
class AoaiBackend:
    backend_id: str
    endpoint: str
    deployment: str
    api_key: str
    api_version: str
    model: Optional[str] = None


class AoaiPool:
    _ROUTING_ROUND_ROBIN = "round_robin"
    _ROUTING_STABLE = "stable"

    def __init__(self, pool_path: Path, routing: Optional[str] = None) -> None:
        with pool_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.backends = [
            AoaiBackend(
                backend_id=entry["id"],
                endpoint=entry["endpoint"],
                deployment=entry["deployment"],
                api_key=entry["apiKey"],
                api_version=entry["apiVersion"],
                model=entry.get("model"),
            )
            for entry in raw
        ]
        if not self.backends:
            raise RuntimeError(f"No backends found in {pool_path}")
        self.routing = routing or os.getenv("AOAI_POOL_ROUTING", self._ROUTING_ROUND_ROBIN)
        if self.routing not in {self._ROUTING_ROUND_ROBIN, self._ROUTING_STABLE}:
            raise ValueError(
                f"Unsupported AOAI pool routing {self.routing!r}; "
                f"expected {self._ROUTING_ROUND_ROBIN!r} or {self._ROUTING_STABLE!r}"
            )
        self._index = 0
        self._lock = threading.Lock()

    def next_backend(self) -> AoaiBackend:
        with self._lock:
            backend = self.backends[self._index % len(self.backends)]
            self._index += 1
        return backend

    def backend_for_request(
        self,
        messages: List[Dict[str, str]],
        step_name: str = "",
        attempt: int = 0,
    ) -> AoaiBackend:
        if self.routing == self._ROUTING_ROUND_ROBIN:
            return self.next_backend()
        payload = {
            "messages": messages,
            "step_name": step_name or "",
        }
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        index = (int.from_bytes(digest, "big") + max(0, int(attempt))) % len(self.backends)
        return self.backends[index]

    def chat_with_usage(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
        step_name: str = "",
    ) -> Tuple[str, Dict[str, int]]:
        with get_tracer().start_as_current_span(step_name or "llm") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("input.value", json.dumps(messages, ensure_ascii=False))
            last_error: Optional[Exception] = None
            for attempt in range(max_attempts):
                backend = self.backend_for_request(messages, step_name=step_name, attempt=attempt)
                url = (
                    backend.endpoint.rstrip("/")
                    + f"/openai/deployments/{backend.deployment}/chat/completions"
                    + f"?api-version={backend.api_version}"
                )
                payload = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "api-key": backend.api_key,
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    content = _extract_chat_content(body)
                    usage = _extract_usage(body)
                    span.set_attribute("llm.model_name", backend.deployment)
                    span.set_attribute("output.value", content)
                    span.set_attribute("llm.token_count.prompt", usage["prompt_tokens"])
                    span.set_attribute("llm.token_count.completion", usage["completion_tokens"])
                    span.set_attribute("llm.token_count.total", usage["total_tokens"])
                    return content, usage
                except urllib.error.HTTPError as error:
                    last_error = error
                    if error.code in {429, 500, 502, 503, 504}:
                        time.sleep(2**attempt)
                        continue
                    raise LLMCallError(
                        f"AOAI request failed with HTTP {error.code} "
                        f"(step={step_name or 'llm'}, deployment={backend.deployment}, "
                        f"endpoint={backend.endpoint}): {_read_error_body(error)}"
                    ) from error
                except Exception as error:  # pylint: disable=broad-except
                    last_error = error
                    time.sleep(2**attempt)
            raise LLMCallError(f"AOAI request failed after {max_attempts} attempts: {last_error}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
        step_name: str = "",
    ) -> str:
        content, _ = self.chat_with_usage(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            step_name=step_name,
        )
        return content


class OpenAIClient:
    def __init__(self, api_key: str, model: str, base_url: str = DEFAULT_OPENAI_BASE_URL) -> None:
        if not api_key or not model:
            raise ValueError("OpenAI client requires api_key and model.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat_with_usage(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
        step_name: str = "",
    ) -> Tuple[str, Dict[str, int]]:
        with get_tracer().start_as_current_span(step_name or "llm") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.model_name", self.model)
            span.set_attribute("input.value", json.dumps(messages, ensure_ascii=False))
            last_error: Optional[Exception] = None
            for attempt in range(max_attempts):
                url = f"{self.base_url}/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    content = _extract_chat_content(body)
                    usage = _extract_usage(body)
                    span.set_attribute("output.value", content)
                    span.set_attribute("llm.token_count.prompt", usage["prompt_tokens"])
                    span.set_attribute("llm.token_count.completion", usage["completion_tokens"])
                    span.set_attribute("llm.token_count.total", usage["total_tokens"])
                    return content, usage
                except urllib.error.HTTPError as error:
                    last_error = error
                    if error.code in {429, 500, 502, 503, 504}:
                        time.sleep(2**attempt)
                        continue
                    raise LLMCallError(
                        f"OpenAI request failed with HTTP {error.code} "
                        f"(step={step_name or 'llm'}): {_read_error_body(error)}"
                    ) from error
                except Exception as error:  # pylint: disable=broad-except
                    last_error = error
                    time.sleep(2**attempt)
            raise LLMCallError(f"OpenAI request failed after {max_attempts} attempts: {last_error}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
        step_name: str = "",
    ) -> str:
        content, _ = self.chat_with_usage(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            step_name=step_name,
        )
        return content


def _extract_chat_content(payload: Dict[str, Any]) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _extract_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


_GROBID_CONNECT_TIMEOUT = 10.0
_GROBID_RETRY_INTERVAL = 5.0


def run_grobid(
    pdf_path: Path,
    tei_path: Path,
    grobid_url: str = DEFAULT_GROBID_URL,
    timeout: int = DEFAULT_GROBID_TIMEOUT,
    consolidate_header: int = 0,
    consolidate_citations: int = 0,
    parser: str = DEFAULT_PARSER,
) -> None:
    """POST a PDF to the configured parser's /processFulltextDocument and write the response TEI to tei_path.

    parser=sciencebeam drops the GROBID-only form fields (teiCoordinates,
    consolidate*) which sciencebeam-parser ignores. No explicit Accept
    header: lfoppiano/grobid returns 406 for application/tei+xml,
    sciencebeam-parser maps the httpx default */* to TEI. A 2xx body
    without a <TEI marker raises here so a malformed parser response
    cannot become a silent empty-TEI parse downstream.
    """
    if parser not in SUPPORTED_PARSERS:
        raise ValueError(
            f"Unsupported parser {parser!r}; expected one of {SUPPORTED_PARSERS}"
        )
    ensure_parent(tei_path)
    if tei_path.exists() and tei_path.stat().st_size > 0:
        return
    url = f"{grobid_url}/processFulltextDocument"
    if parser == PARSER_GROBID:
        form_data = {
            "teiCoordinates": "1",
            "consolidateHeader": str(int(consolidate_header)),
            "consolidateCitations": str(int(consolidate_citations)),
        }
    else:
        form_data = {}
    deadline = time.monotonic() + timeout
    while True:
        try:
            with pdf_path.open("rb") as pdf_file:
                response = httpx.post(
                    url,
                    files={"input": (pdf_path.name, pdf_file, "application/pdf")},
                    data=form_data,
                    timeout=httpx.Timeout(
                        connect=_GROBID_CONNECT_TIMEOUT,
                        read=float(timeout),
                        write=30.0,
                        pool=5.0,
                    ),
                )
            response.raise_for_status()
            body = response.text
            # Surface the silent-failure case where the upstream returns 200
            # but the body is not a TEI document (e.g. an HTML error page or
            # an empty body). Without this, downstream extract_tei_fields
            # parses an empty document and the pipeline reports zero metrics
            # without ever raising.
            if "<TEI" not in body and "<tei" not in body:
                snippet = body[:200].replace("\n", " ")
                raise RuntimeError(
                    f"{parser} returned non-TEI response from {url} "
                    f"(content-type={response.headers.get('content-type', '?')}): {snippet!r}"
                )
            tei_path.write_text(body, encoding="utf-8")
            return
        except httpx.ConnectError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"could not connect to {parser} at {grobid_url} after {timeout}s"
                ) from exc
            time.sleep(min(_GROBID_RETRY_INTERVAL, remaining))
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"{parser} request timed out after {timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"{parser} returned HTTP {exc.response.status_code}") from exc


def run_pdfalto(
    pdf_path: Path,
    alto_path: Path,
    pdfalto_bin: Path = DEFAULT_PDFALTO_BIN,
    start_page: int = 1,
    end_page: int = 1,
) -> None:
    ensure_parent(alto_path)
    if alto_path.exists() and alto_path.stat().st_size > 0:
        return
    command = [
        str(pdfalto_bin),
        "-skipGraphs",
        "-f",
        str(start_page),
        "-l",
        str(end_page),
        str(pdf_path),
        str(alto_path),
    ]
    subprocess.run(command, check=True)
    if not alto_path.exists() or alto_path.stat().st_size == 0:
        raise RuntimeError(f"pdfalto did not produce output for {pdf_path}")
