from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from grobid_metadata_enricher.clients import _extract_usage


class TestExtractUsage:
    def test_full_usage_with_details(self) -> None:
        payload = {
            "choices": [{"message": {"content": "x"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 40, "audio_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 10, "audio_tokens": 0},
            },
        }
        usage = _extract_usage(payload)
        assert usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 40,
            "reasoning_tokens": 10,
        }

    def test_missing_usage_returns_zeros(self) -> None:
        # Proxies / older api versions can strip the usage block. Aggregation must still work.
        usage = _extract_usage({"choices": [{"message": {"content": "x"}}]})
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }

    def test_missing_total_tokens_derived_from_prompt_plus_completion(self) -> None:
        payload = {"usage": {"prompt_tokens": 30, "completion_tokens": 70}}
        assert _extract_usage(payload)["total_tokens"] == 100

    def test_missing_details_blocks_default_to_zero(self) -> None:
        payload = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        usage = _extract_usage(payload)
        assert usage["cached_tokens"] == 0
        assert usage["reasoning_tokens"] == 0

    def test_null_details_fields_treated_as_zero(self) -> None:
        # None values inside details should not crash the int() cast.
        payload = {
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": None},
                "completion_tokens_details": {"reasoning_tokens": None},
            }
        }
        usage = _extract_usage(payload)
        assert usage["cached_tokens"] == 0
        assert usage["reasoning_tokens"] == 0


class _FakePool:
    """AoaiPool-shaped stand-in; records each call and returns a scripted (content, usage) tuple."""

    def __init__(self, script: Dict[str, Tuple[str, Dict[str, int]]]) -> None:
        self._script = script
        self.seen: List[Tuple[int, float, int]] = []
        self._lock = threading.Lock()

    def chat_with_usage(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        step_name: str = "",
    ) -> Tuple[str, Dict[str, int]]:
        # Key off the first system-prompt content word so tests can script per-stage returns.
        del step_name  # captured by the closure caller; fake pool only inspects messages.
        tag = messages[0]["content"].split()[0]
        with self._lock:
            self.seen.append((len(messages), temperature, max_tokens))
        return self._script[tag]


class TestUsageRecorderAndSummarise:
    def test_basic_aggregation_single_thread(self) -> None:
        from benchmarks.predict import UsageRecorder, summarise_tokens

        recorder = UsageRecorder()
        recorder.add("HEADER_METADATA", {
            "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
            "cached_tokens": 0, "reasoning_tokens": 0,
        }, latency_ms=12.3)
        recorder.add("CONTENT_HEAD", {
            "prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700,
            "cached_tokens": 40, "reasoning_tokens": 0,
        }, latency_ms=45.7)

        summary = summarise_tokens(recorder)
        assert summary["total"]["prompt_tokens"] == 600
        assert summary["total"]["completion_tokens"] == 250
        assert summary["total"]["total_tokens"] == 850
        assert summary["total"]["cached_tokens"] == 40
        assert summary["total"]["n_calls"] == 2
        assert summary["total"]["latency_ms_sum"] == pytest.approx(58.0, abs=0.5)

        assert summary["by_stage"]["HEADER_METADATA"]["prompt_tokens"] == 100
        assert summary["by_stage"]["CONTENT_HEAD"]["prompt_tokens"] == 500

        # HEADER_METADATA maps to "header"; CONTENT_HEAD maps to "content"
        assert summary["by_metric_group"]["header"]["prompt_tokens"] == 100
        assert summary["by_metric_group"]["content"]["prompt_tokens"] == 500
        assert summary["by_metric_group"]["header"]["n_calls"] == 1
        assert summary["by_metric_group"]["content"]["n_calls"] == 1

        assert len(summary["calls"]) == 2

    def test_unknown_stage_buckets_into_other_group(self) -> None:
        from benchmarks.predict import UsageRecorder, summarise_tokens

        recorder = UsageRecorder()
        recorder.add("MADE_UP_STAGE", {
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
            "cached_tokens": 0, "reasoning_tokens": 0,
        }, latency_ms=0.5)

        summary = summarise_tokens(recorder)
        assert "other" in summary["by_metric_group"]
        assert summary["by_metric_group"]["other"]["prompt_tokens"] == 1

    def test_thread_safety_concurrent_appends(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from benchmarks.predict import UsageRecorder, summarise_tokens

        recorder = UsageRecorder()
        usage = {
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
            "cached_tokens": 0, "reasoning_tokens": 0,
        }
        stages = [
            "HEADER_METADATA", "TEI_METADATA", "TEI_VALIDATED",
            "ABSTRACT_SELECT", "OCR_CLEANUP", "ABSTRACT_FROM_OCR",
            "KEYWORD_TRANSLATE", "CONTENT_HEAD", "CONTENT_REFERENCES", "CONTENT_TABLES_FIGURES",
        ]

        # 10 stages x 200 concurrent calls = 2000 writes across 40 threads.
        n_per_stage = 200
        with ThreadPoolExecutor(max_workers=40) as ex:
            futures = []
            for stage in stages:
                for _ in range(n_per_stage):
                    futures.append(ex.submit(recorder.add, stage, usage, 1.0))
            for f in futures:
                f.result()

        summary = summarise_tokens(recorder)
        assert summary["total"]["n_calls"] == n_per_stage * len(stages)
        assert summary["total"]["prompt_tokens"] == n_per_stage * len(stages)
        for stage in stages:
            assert summary["by_stage"][stage]["n_calls"] == n_per_stage

    def test_make_chat_records_usage_and_returns_content(self) -> None:
        from benchmarks.predict import UsageRecorder, make_chat

        pool = _FakePool({
            "HEADER_METADATA_SYSTEM": (
                "content-header",
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                 "cached_tokens": 0, "reasoning_tokens": 0},
            ),
            "CONTENT_HEAD_SYSTEM": (
                "content-body",
                {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                 "cached_tokens": 20, "reasoning_tokens": 0},
            ),
        })
        semaphore = threading.Semaphore(4)
        recorder = UsageRecorder()
        chat = make_chat(
            pool,   # type: ignore[arg-type]
            semaphore,
            recorder,
            default_temperature=0.0,
            default_max_tokens=800,
        )

        out = chat(
            [{"role": "system", "content": "HEADER_METADATA_SYSTEM body"}],
            step_name="HEADER_METADATA",
        )
        assert out == "content-header"
        out = chat(
            [{"role": "system", "content": "CONTENT_HEAD_SYSTEM body"}],
            temperature=0.0,
            max_tokens=2000,
            step_name="CONTENT_HEAD",
        )
        assert out == "content-body"

        summary = __import__("benchmarks.predict", fromlist=["summarise_tokens"]).summarise_tokens(recorder)
        assert summary["total"]["prompt_tokens"] == 110
        assert summary["total"]["completion_tokens"] == 45
        assert summary["total"]["cached_tokens"] == 20
        assert summary["by_stage"]["CONTENT_HEAD"]["prompt_tokens"] == 100


class TestChatClientBackwardCompat:
    """.chat() must keep working for api.py / cli.py / existing tests that don't pass `stage`."""

    def test_aoai_chat_returns_string_and_ignores_usage(self, tmp_path: Any) -> None:
        from grobid_metadata_enricher.clients import AoaiPool

        pool_path = tmp_path / "pool.json"
        pool_path.write_text(
            '[{"id":"x","endpoint":"https://x/","deployment":"d","apiKey":"k","apiVersion":"v1"}]',
            encoding="utf-8",
        )
        pool = AoaiPool(pool_path)

        class _FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def read(self) -> bytes:
                return self._body

        body = (
            b'{"choices":[{"message":{"content":"hi"}}],'
            b'"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}'
        )
        with patch("grobid_metadata_enricher.clients.urllib.request.urlopen", return_value=_FakeResponse(body)):
            # No `stage` kwarg — historical call shape.
            assert pool.chat([{"role": "system", "content": "hi"}]) == "hi"
            # Also verify chat_with_usage returns the usage.
            content, usage = pool.chat_with_usage([{"role": "system", "content": "hi"}])
            assert content == "hi"
            assert usage["prompt_tokens"] == 3
            assert usage["completion_tokens"] == 1
            assert usage["total_tokens"] == 4
