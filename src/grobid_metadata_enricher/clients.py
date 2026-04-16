from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_POOL_PATH = Path(os.getenv("AOAI_POOL_PATH", "aoai_pool.json"))
DEFAULT_GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api")
DEFAULT_GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "60"))
DEFAULT_PDFALTO_BIN = Path(os.getenv("PDFALTO_BIN", "pdfalto"))
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


class AoaiPool:
    def __init__(self, pool_path: Path) -> None:
        with pool_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.backends = [
            AoaiBackend(
                backend_id=entry["id"],
                endpoint=entry["endpoint"],
                deployment=entry["deployment"],
                api_key=entry["apiKey"],
                api_version=entry["apiVersion"],
            )
            for entry in raw
        ]
        if not self.backends:
            raise RuntimeError(f"No backends found in {pool_path}")
        self._index = 0
        self._lock = threading.Lock()

    def next_backend(self) -> AoaiBackend:
        with self._lock:
            backend = self.backends[self._index % len(self.backends)]
            self._index += 1
        return backend

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            backend = self.next_backend()
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
                return _extract_chat_content(body)
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code in {429, 500, 502, 503, 504}:
                    time.sleep(2**attempt)
                    continue
                raise
            except Exception as error:
                last_error = error
                time.sleep(2**attempt)
        raise RuntimeError(f"AOAI request failed after {max_attempts} attempts: {last_error}")


class OpenAIClient:
    def __init__(self, api_key: str, model: str, base_url: str = DEFAULT_OPENAI_BASE_URL) -> None:
        if not api_key or not model:
            raise ValueError("OpenAI client requires api_key and model.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
    ) -> str:
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
                return _extract_chat_content(body)
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code in {429, 500, 502, 503, 504}:
                    time.sleep(2**attempt)
                    continue
                raise
            except Exception as error:
                last_error = error
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI request failed after {max_attempts} attempts: {last_error}")


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
) -> None:
    ensure_parent(tei_path)
    if tei_path.exists() and tei_path.stat().st_size > 0:
        return
    url = f"{grobid_url}/processFulltextDocument"
    deadline = time.monotonic() + timeout
    while True:
        try:
            with pdf_path.open("rb") as pdf_file:
                response = httpx.post(
                    url,
                    files={"input": (pdf_path.name, pdf_file, "application/pdf")},
                    data={
                        "teiCoordinates": "1",
                        "consolidateHeader": str(int(consolidate_header)),
                        "consolidateCitations": str(int(consolidate_citations)),
                    },
                    timeout=httpx.Timeout(
                        connect=_GROBID_CONNECT_TIMEOUT,
                        read=float(timeout),
                        write=30.0,
                        pool=5.0,
                    ),
                )
            response.raise_for_status()
            tei_path.write_text(response.text, encoding="utf-8")
            return
        except httpx.ConnectError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"could not connect to GROBID at {grobid_url} after {timeout}s"
                ) from exc
            time.sleep(min(_GROBID_RETRY_INTERVAL, remaining))
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"GROBID request timed out after {timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"GROBID returned HTTP {exc.response.status_code}") from exc


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
