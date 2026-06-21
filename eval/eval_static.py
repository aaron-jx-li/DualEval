#!/usr/bin/env python3
"""
Unified static evaluation pipeline (math + misc + coding) for the current model roster.

Key design choices:
1. Model specs (ModelSpec, MODEL_SPECS, MODEL_LOOKUP) and client utilities are
   defined inline here — no external model_api_smoke_test dependency.
2. Sample static benchmarks rather than running full benchmarks by default.
3. Use a flexible LLM judge for correctness on math/misc; execution harness for coding.
4. Domain is selected via --domain {math, misc, coding}; prompts and grading are
   dispatched from per-domain dicts/functions.

Usage examples:
    python eval/eval_static.py --domain math --config demo/config_static.yaml
    python eval/eval_static.py --domain misc --config demo/config_static.yaml
    python eval/eval_static.py --domain coding --config demo/config_static_coding.yaml
    python eval/eval_static.py --domain math --sample-file results/static_samples/run1/sampled_items.jsonl
    python eval/eval_static.py --domain math --sample-file results/static_samples/run1/sampled_items.jsonl --models gpt-5.4 claude-opus-4-6
    python eval/eval_static.py --domain math --sample-file results/static_samples/run1/sampled_items.jsonl --judge-model gpt-4.1-mini
    python eval/eval_static.py --domain math --config demo/config_static.yaml --use-litellm
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anthropic
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from openai import BadRequestError, OpenAI
from tqdm import tqdm
import yaml


# ---------------------------------------------------------------------------
# Model specs (inlined from model_api_smoke_test.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    label: str
    family: str
    provider: str
    model_id: str
    api_env: str
    litellm_model_id: str | None = None
    reasoning_effort: str | None = None  # OpenAI reasoning_effort top-level param
    effort: str | None = None            # Anthropic thinking.budget_tokens trigger


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("gpt-5.4", "openai", "openai", "gpt-5.4", "OPENAI_API_KEY", "openai/gpt-5.4"),
    ModelSpec("gpt-5-mini", "openai", "openai", "gpt-5-mini", "OPENAI_API_KEY", "openai/gpt-5-mini"),
    ModelSpec("gpt-5.1-mini", "openai", "openai", "gpt-5.1-mini", "OPENAI_API_KEY", "openai/gpt-5.1-mini"),
    ModelSpec("gpt-5.4-mini", "openai", "openai", "gpt-5.4-mini", "OPENAI_API_KEY"),
    ModelSpec("gpt-5.5", "openai", "openai", "gpt-5.5", "OPENAI_API_KEY"),
    ModelSpec("gpt-5.5-high", "openai", "openai", "gpt-5.5", "OPENAI_API_KEY", reasoning_effort="high"),
    ModelSpec("gpt-4.1", "openai", "openai", "gpt-4.1", "OPENAI_API_KEY", "openai/gpt-4.1"),
    ModelSpec(
        "gpt-4.1-mini",
        "openai",
        "openai",
        "gpt-4.1-mini",
        "OPENAI_API_KEY",
        "openai/gpt-4.1-mini",
    ),
    ModelSpec(
        "claude-opus-4-6",
        "anthropic",
        "anthropic",
        "claude-opus-4-6",
        "ANTHROPIC_API_KEY",
        "claude-opus-4-6",
    ),
    ModelSpec(
        "claude-opus-4-7",
        "anthropic",
        "anthropic",
        "claude-opus-4-7",
        "ANTHROPIC_API_KEY",
    ),
    ModelSpec(
        "claude-opus-4-7-thinking",
        "anthropic",
        "anthropic",
        "claude-opus-4-7",
        "ANTHROPIC_API_KEY",
        effort="max",
    ),
    ModelSpec(
        "claude-sonnet-4-6",
        "anthropic",
        "anthropic",
        "claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-4-6",
    ),
    ModelSpec(
        "claude-haiku-4-5",
        "anthropic",
        "anthropic",
        "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY",
        "vertex_ai/claude-haiku-4-5@20251001",
    ),
    ModelSpec(
        "gemini-3.1-pro",
        "google",
        "google",
        "gemini-3.1-pro",
        "GOOGLE_API_KEY",
        "gemini/gemini-3.1-pro",
    ),
    ModelSpec(
        "gemini-2.5-pro",
        "google",
        "google",
        "gemini-2.5-pro",
        "GOOGLE_API_KEY",
        "gemini/gemini-2.5-pro",
    ),
    ModelSpec(
        "gemini-2.5-flash",
        "google",
        "google",
        "gemini-2.5-flash",
        "GOOGLE_API_KEY",
        "gemini/gemini-2.5-flash",
    ),
    ModelSpec("grok-4", "xai", "openrouter", "x-ai/grok-4.20-beta", "OPENROUTER_API_KEY", None),
    ModelSpec(
        "deepseek-v3.2",
        "deepseek",
        "openrouter",
        "deepseek/deepseek-v3.2",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "deepseek-r1",
        "deepseek",
        "openrouter",
        "deepseek/deepseek-r1-0528",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "mistral-large-3",
        "mistral",
        "openrouter",
        "mistralai/mistral-large-2512",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "qwen3-max-thinking",
        "qwen",
        "openrouter",
        "qwen/qwen3-max-thinking",
        "OPENROUTER_API_KEY",
        None,
    ),
    ModelSpec(
        "llama-4-maverick-instruct",
        "llama",
        "openrouter",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "OPENROUTER_API_KEY",
        None,
    ),
]

MODEL_LOOKUP: dict[str, ModelSpec] = {spec.label: spec for spec in MODEL_SPECS}


def get_env_value(*names: str) -> str | None:
    """Return the first non-empty env value that is not a literal shell expression."""
    for name in names:
        value = os.environ.get(name)
        if value and "${" not in value:
            return value
    return None


def extract_chat_content(content: Any) -> str:
    """Best-effort extraction from OpenAI-compatible message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def normalize_base_url(base_url: str | None, provider: str) -> str | None:
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    if provider == "anthropic" and normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def should_use_litellm_for_model(spec: ModelSpec, *, use_litellm: bool) -> bool:
    return use_litellm and spec.litellm_model_id is not None


def get_routed_model_id(spec: ModelSpec, *, use_litellm: bool) -> str:
    if use_litellm:
        if not spec.litellm_model_id:
            raise ValueError(f"LiteLLM mapping is not configured for model '{spec.label}'.")
        return spec.litellm_model_id
    return spec.model_id


# ---------------------------------------------------------------------------
# Per-domain prompt constants
# ---------------------------------------------------------------------------

ANSWER_INSTRUCTIONS: dict[str, str] = {
    "math": (
        "Solve the math problem carefully. "
        "End your response with a final line of the form 'Final answer: <answer>'."
    ),
    "misc": (
        "Answer the question concisely and correctly. "
        "End your response with a final line of the form 'Final answer: <answer>'."
    ),
}

JUDGE_SYSTEM_INSTRUCTIONS: dict[str, str] = {
    "math": (
        "You are a careful mathematics grader. "
        "Determine whether a model response is mathematically correct. "
        "Mark an answer correct when the final answer is mathematically equivalent to the ground truth, "
        "even if it uses a different but equivalent form. "
        "Treat equivalent forms as correct, including fractions vs decimals (for example 11/2 and 5.5), "
        "unsimplified vs simplified expressions, equivalent algebraic forms, and equivalent interval/set notation "
        "when they represent the same solution. "
        "For multiple-choice questions, accept either the correct option letter or the correct option content. "
        "Do not reward style, verbosity, or formatting. Focus on mathematical correctness."
    ),
    "misc": (
        "You are a careful grader for short factual and technical answers. "
        "Decide whether the model response is correct with respect to the reference answer. "
        "Treat equivalent phrasings, common aliases, rounding-equivalent numbers, and equivalent units as correct. "
        "For multiple acceptable formulations of the same fact, mark correct. "
        "Do not reward style or verbosity. Focus on factual correctness."
    ),
}

JUDGE_PROMPT_TEMPLATES: dict[str, str] = {
    "math": """\
Question:
{question}

Ground-truth answer:
{gold_answer}

Model response:
{model_answer}

Decide whether the model response is mathematically correct.

Important grading rules:
- Count mathematically equivalent answers as correct.
- Examples of equivalent answers include 11/2 and 5.5, 0.5 and 1/2, or algebraically equivalent expressions.
- For multiple-choice questions, accept either the correct letter choice or the correct option content.
- Minor notation differences or harmless formatting differences should not make a correct answer wrong.
- If the response contains reasoning plus a final answer, judge based on whether the final mathematical conclusion is correct and supported well enough.

Return ONLY a JSON object:
{{"correct": true or false, "reason": "brief explanation"}}
""",
    "misc": """\
Question:
{question}

Reference answer:
{gold_answer}

Model response:
{model_answer}

Decide whether the model response is factually correct (including acceptable paraphrases and equivalents).

Return ONLY a JSON object:
{{"correct": true or false, "reason": "brief explanation"}}
""",
}


# ---------------------------------------------------------------------------
# Dataset specs (math domain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str
    default_pilot_samples: int
    default_paper_samples: int


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec("gsm8k", "openai/gsm8k", "main", "test", "gsm8k", 30, 50),
    DatasetSpec(
        "mmlu-abstract",
        "brucewlee1/mmlu-abstract-algebra",
        None,
        "test",
        "mmlu",
        10,
        20,
    ),
    DatasetSpec(
        "mmlu-college",
        "brucewlee1/mmlu-college-mathematics",
        None,
        "test",
        "mmlu",
        10,
        20,
    ),
    DatasetSpec("math-algebra", "EleutherAI/hendrycks_math", "algebra", "test", "math", 10, 35),
    DatasetSpec(
        "math-counting",
        "EleutherAI/hendrycks_math",
        "counting_and_probability",
        "test",
        "math",
        10,
        50,
    ),
    DatasetSpec("math-geometry", "EleutherAI/hendrycks_math", "geometry", "test", "math", 10, 50),
    DatasetSpec("math-number", "EleutherAI/hendrycks_math", "number_theory", "test", "math", 10, 50),
    DatasetSpec(
        "math-intermediate",
        "EleutherAI/hendrycks_math",
        "intermediate_algebra",
        "test",
        "math",
        10,
        55,
    ),
    DatasetSpec(
        "math-prealgebra",
        "EleutherAI/hendrycks_math",
        "prealgebra",
        "test",
        "math",
        10,
        15,
    ),
    DatasetSpec(
        "math-precalculus",
        "EleutherAI/hendrycks_math",
        "precalculus",
        "test",
        "math",
        10,
        15,
    ),
    DatasetSpec("aime-2025", "test-time-compute/aime_2025", None, "test", "aime", 5, 30),
    DatasetSpec("aime-2026", "MathArena/aime_2026", None, "train", "aime", 5, 30),
    DatasetSpec("olympiad-math", "math-ai/olympiadbench", None, "test", "olympiad", 10, 80),
    # HLE (Humanity's Last Exam) — text-only math subset
    DatasetSpec("hle-math", "cais/hle", None, "test", "hle", 10, 60),
]

DATASET_LOOKUP = {spec.name: spec for spec in DATASET_SPECS}


# ---------------------------------------------------------------------------
# Coding domain: dataset specs, prompt builders, graders
# ---------------------------------------------------------------------------

CODE_INSTRUCTION = (
    "Write a correct Python 3 solution. "
    "Return only executable Python code in a single ```python``` block, with no explanation."
)

EXECUTION_PREAMBLE = """\
from typing import *
"""


@dataclass(frozen=True)
class CodingDatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str
    default_pilot_samples: int
    default_paper_samples: int
    source_filename: str | None = None


CODING_DATASET_SPECS: list[CodingDatasetSpec] = [
    CodingDatasetSpec(
        "humaneval-plus",
        "evalplus/humanevalplus",
        None,
        "test",
        "humaneval",
        30,
        80,
    ),
    CodingDatasetSpec(
        "mbpp-plus-sanitized",
        "evalplus/mbppplus",
        None,
        "test",
        "mbpp",
        30,
        140,
    ),
    CodingDatasetSpec(
        "livecodebench-v6",
        "livecodebench/code_generation_lite",
        None,
        "test",
        "livecodebench",
        40,
        160,
        source_filename="test6.jsonl",
    ),
]

CODING_DATASET_LOOKUP: dict[str, CodingDatasetSpec] = {spec.name: spec for spec in CODING_DATASET_SPECS}


def _decode_livecodebench_private_tests(raw_value: str) -> list[dict[str, Any]]:
    decoded = zlib.decompress(base64.b64decode(raw_value))
    parsed = pickle.loads(decoded)
    if isinstance(parsed, bytes):
        parsed = parsed.decode("utf-8")
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return parsed


def _load_livecodebench_public_tests(item: dict[str, Any]) -> list[dict[str, Any]]:
    raw_cases = item.get("public_test_cases")
    if not raw_cases:
        return []
    if isinstance(raw_cases, str):
        return json.loads(raw_cases)
    if isinstance(raw_cases, list):
        return raw_cases
    return []


def _infer_livecodebench_test_type(item: dict[str, Any]) -> str:
    public_tests = _load_livecodebench_public_tests(item)
    if public_tests:
        return str(public_tests[0].get("testtype", "stdin"))
    return "functional" if item.get("starter_code") else "stdin"


def _build_livecodebench_interface_instruction(item: dict[str, Any]) -> str:
    test_type = _infer_livecodebench_test_type(item)
    starter_code = item.get("starter_code", "").strip()
    if test_type == "functional":
        if starter_code:
            return (
                "Implement the solution exactly within the provided starter code. "
                "Keep the same class name, function name, and signature."
            )
        return "Write the required Python function so it can be called directly by the tests."
    return (
        "Write a complete Python program that reads from standard input and writes to standard output. "
        "Do not print any extra text."
    )


def _format_livecodebench_public_examples(item: dict[str, Any]) -> str | None:
    public_tests = _load_livecodebench_public_tests(item)
    if not public_tests:
        return None
    test_type = _infer_livecodebench_test_type(item)
    lines: list[str] = ["Public examples:"]
    for idx, case in enumerate(public_tests, start=1):
        lines.append("")
        lines.append(f"Example {idx}:")
        input_label = "Input" if test_type == "stdin" else "Arguments"
        lines.append(f"{input_label}:")
        lines.append("```text")
        lines.append(str(case.get("input", "")).rstrip())
        lines.append("```")
        lines.append("Output:")
        lines.append("```text")
        lines.append(str(case.get("output", "")).rstrip())
        lines.append("```")
    return "\n".join(lines)


def _build_coding_raw_question(dataset_spec: CodingDatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "humaneval":
        return (
            "Complete the following Python function.\n\n"
            f"```python\n{item['prompt'].rstrip()}\n```"
        )
    if dataset_spec.kind == "mbpp":
        signature_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(.*?\)\s*:", item.get("code", ""), flags=re.DOTALL)
        required_name = signature_match.group(1) if signature_match else None
        sections = ["Write a Python function for this task:", "", item["prompt"].strip()]
        if required_name:
            sections.extend(["", f"Your function must be named `{required_name}`."])
        return "\n".join(sections)
    if dataset_spec.kind == "livecodebench":
        starter_code = item.get("starter_code", "").rstrip()
        interface_instruction = _build_livecodebench_interface_instruction(item)
        public_examples = _format_livecodebench_public_examples(item)
        sections = [
            f"Title: {item.get('question_title', '').strip()}",
            "",
            item.get("question_content", "").strip(),
            "",
            interface_instruction,
        ]
        if starter_code:
            sections.extend(["", "Starter code:", f"```python\n{starter_code}\n```"])
        if public_examples:
            sections.extend(["", public_examples])
        return "\n".join(part for part in sections if part is not None)
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def _build_coding_eval_prompt(dataset_spec: CodingDatasetSpec, item: dict) -> str:
    return f"{_build_coding_raw_question(dataset_spec, item)}\n\n{CODE_INSTRUCTION}"


def _build_coding_gold_answer(dataset_spec: CodingDatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "humaneval":
        return f"{item['prompt'].rstrip()}\n{item['canonical_solution'].rstrip()}"
    if dataset_spec.kind == "mbpp":
        return item["code"]
    if dataset_spec.kind == "livecodebench":
        private_tests = _decode_livecodebench_private_tests(item["private_test_cases"])
        return json.dumps(
            {
                "public_tests": json.loads(item["public_test_cases"]),
                "private_test_count": len(private_tests),
            },
            ensure_ascii=False,
        )
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def _get_coding_item_metadata(dataset_spec: CodingDatasetSpec, item: dict) -> dict[str, Any]:
    if dataset_spec.kind == "humaneval":
        return {"level": "function", "subject": "python_function_synthesis"}
    if dataset_spec.kind == "mbpp":
        return {"level": "basic", "subject": "python_programming"}
    if dataset_spec.kind == "livecodebench":
        return {
            "level": item.get("difficulty") or "unknown",
            "subject": item.get("platform") or "competitive_programming",
        }
    raise ValueError(f"Unsupported dataset kind: {dataset_spec.kind}")


def extract_code_from_response(response_text: str) -> str:
    text = response_text.strip()
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return code_blocks[0].strip()
    return text


def _normalize_text_output(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple) and len(actual) == len(expected):
        return all(_values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_values_equal(actual[k], expected[k]) for k in actual)
    return actual == expected


def _run_python_script(script_text: str, *, stdin_text: str = "", timeout_s: int = 10) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="static_coding_eval_") as tmpdir:
        script_path = Path(tmpdir) / "runner.py"
        script_path.write_text(script_text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script_path)],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=tmpdir,
        )


def _grade_humaneval(item: dict[str, Any], code: str) -> dict[str, Any]:
    raw = item["raw_item"]
    script = (
        f"{EXECUTION_PREAMBLE}\n"
        f"{code}\n\n"
        f"{raw['test']}\n\n"
        f"check({raw['entry_point']})\n"
        "print('__EVAL_PASS__')\n"
    )
    completed = _run_python_script(script)
    passed = completed.returncode == 0 and "__EVAL_PASS__" in completed.stdout
    details = completed.stderr or completed.stdout
    return {"passed": passed, "details": details[:4000], "grader_raw": (completed.stdout + completed.stderr)[:8000]}


def _grade_mbpp(item: dict[str, Any], code: str) -> dict[str, Any]:
    raw = item["raw_item"]
    script = f"{EXECUTION_PREAMBLE}\n{code}\n\n{raw['test']}\nprint('__EVAL_PASS__')\n"
    completed = _run_python_script(script)
    passed = completed.returncode == 0 and "__EVAL_PASS__" in completed.stdout
    details = completed.stderr or completed.stdout
    return {"passed": passed, "details": details[:4000], "grader_raw": (completed.stdout + completed.stderr)[:8000]}


def _infer_livecodebench_callable(raw_item: dict[str, Any], code: str) -> tuple[str, str]:
    starter_code = raw_item.get("starter_code", "")
    class_match = re.search(r"class\s+Solution\s*:\s*(?:\n[ \t]+.*)*?\n[ \t]+def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    if class_match:
        return "solution_method", class_match.group(1)
    func_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", starter_code)
    if func_match:
        return "function", func_match.group(1)
    code_func_match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    if code_func_match:
        return "function", code_func_match.group(1)
    raise ValueError("Could not infer callable name for LiveCodeBench task.")


def _build_livecodebench_functional_harness(code: str, raw_item: dict[str, Any], tests: list[dict[str, Any]]) -> str:
    target_kind, target_name = _infer_livecodebench_callable(raw_item, code)
    return f"""\
import inspect
import json
import math

{EXECUTION_PREAMBLE}
{code}

TESTS = {json.dumps(tests, ensure_ascii=False)}
TARGET_KIND = {target_kind!r}
TARGET_NAME = {target_name!r}

def parse_value(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        import ast
        return ast.literal_eval(text)

def values_equal(actual, expected):
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, tuple) and isinstance(expected, tuple) and len(actual) == len(expected):
        return all(values_equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(values_equal(actual[k], expected[k]) for k in actual)
    return actual == expected

if TARGET_KIND == "solution_method":
    callable_obj = getattr(Solution(), TARGET_NAME)
else:
    callable_obj = globals()[TARGET_NAME]

signature = inspect.signature(callable_obj)
param_count = len(signature.parameters)

for idx, case in enumerate(TESTS):
    raw_input = parse_value(case["input"])
    expected = parse_value(case["output"])
    if param_count == 0:
        actual = callable_obj()
    elif param_count == 1:
        actual = callable_obj(raw_input)
    elif isinstance(raw_input, dict):
        actual = callable_obj(**raw_input)
    elif isinstance(raw_input, (list, tuple)):
        actual = callable_obj(*raw_input)
    else:
        raise AssertionError(f"Case {{idx}} input shape incompatible with signature: {{raw_input!r}}")
    if not values_equal(actual, expected):
        raise AssertionError(
            f"Case {{idx}} failed: expected={{expected!r}} actual={{actual!r}} input={{raw_input!r}}"
        )

print("__EVAL_PASS__")
"""


def _grade_livecodebench(item: dict[str, Any], code: str) -> dict[str, Any]:
    raw = item["raw_item"]
    public_tests = json.loads(raw["public_test_cases"])
    private_tests = _decode_livecodebench_private_tests(raw["private_test_cases"])
    tests = public_tests + private_tests
    test_type = tests[0].get("testtype", "stdin") if tests else "stdin"

    if test_type == "functional":
        script = _build_livecodebench_functional_harness(code, raw, tests)
        completed = _run_python_script(script)
        passed = completed.returncode == 0 and "__EVAL_PASS__" in completed.stdout
        details = completed.stderr or completed.stdout
        return {"passed": passed, "details": details[:4000], "grader_raw": (completed.stdout + completed.stderr)[:8000]}

    last_stdout = ""
    for idx, case in enumerate(tests):
        completed = _run_python_script(f"{EXECUTION_PREAMBLE}\n{code}", stdin_text=case.get("input", ""))
        actual = _normalize_text_output(completed.stdout)
        expected = _normalize_text_output(case.get("output", ""))
        last_stdout = completed.stdout + completed.stderr
        if completed.returncode != 0:
            return {
                "passed": False,
                "details": f"stdin case {idx} exited with code {completed.returncode}",
                "grader_raw": last_stdout[:8000],
            }
        if actual != expected:
            return {
                "passed": False,
                "details": f"stdin case {idx} mismatch: expected={expected!r} actual={actual!r}",
                "grader_raw": last_stdout[:8000],
            }
    return {"passed": True, "details": "", "grader_raw": last_stdout[:8000]}


def _grade_code_response(item: dict[str, Any], response_text: str) -> dict[str, Any]:
    code = extract_code_from_response(response_text)
    if not code.strip():
        return {
            "passed": False,
            "details": "No executable code was extracted from the model response.",
            "grader_raw": response_text[:8000],
            "extracted_code": code,
        }
    try:
        if item["dataset_kind"] == "humaneval":
            graded = _grade_humaneval(item, code)
        elif item["dataset_kind"] == "mbpp":
            graded = _grade_mbpp(item, code)
        elif item["dataset_kind"] == "livecodebench":
            graded = _grade_livecodebench(item, code)
        else:
            graded = {
                "passed": False,
                "details": f"Unsupported dataset kind: {item['dataset_kind']}",
                "grader_raw": "",
            }
    except subprocess.TimeoutExpired as exc:
        graded = {
            "passed": False,
            "details": f"Execution timed out after {exc.timeout}s.",
            "grader_raw": "",
        }
    except Exception as exc:
        graded = {
            "passed": False,
            "details": f"Execution harness error: {exc}",
            "grader_raw": "",
        }
    graded["extracted_code"] = code
    return graded


def _load_coding_raw_rows(spec: CodingDatasetSpec) -> list[dict[str, Any]]:
    if spec.kind == "livecodebench":
        if not spec.source_filename:
            raise ValueError(f"Dataset '{spec.name}' requires a source filename.")
        path = hf_hub_download(
            repo_id=spec.hf_path,
            filename=spec.source_filename,
            repo_type="dataset",
        )
        rows: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_path)[spec.split]
    else:
        dataset = load_dataset(spec.hf_path, spec.hf_config)[spec.split]
    return [dict(row) for row in dataset]


def _hydrate_coding_sampled_item(item: dict[str, Any]) -> dict[str, Any]:
    dataset_name = str(item["dataset"])
    spec = CODING_DATASET_LOOKUP[dataset_name]
    raw_item = item["raw_item"]
    metadata = _get_coding_item_metadata(spec, raw_item)
    return {
        "dataset": dataset_name,
        "dataset_kind": item.get("dataset_kind", spec.kind),
        "sample_index": item["sample_index"],
        "prompt": item["prompt"],
        "question": item.get("question", _build_coding_raw_question(spec, raw_item)),
        "gold_answer": item.get("gold_answer", _build_coding_gold_answer(spec, raw_item)),
        "level": item.get("level", metadata["level"]),
        "subject": item.get("subject", metadata["subject"]),
        "raw_item": raw_item,
    }


def _write_coding_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "dataset",
                "num_items",
                "num_scored",
                "num_correct",
                "accuracy",
                "exec_graded",
                "generation_errors",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def _summarize_coding_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["dataset"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_label, dataset_name), group in sorted(grouped.items()):
        num_items = len(group)
        num_correct = sum(1 for row in group if row["correct"] is True)
        exec_count = sum(1 for row in group if row["grading_method"] == "exec_tests")
        error_count = sum(1 for row in group if row["status"] != "ok")
        summary_rows.append(
            {
                "model_label": model_label,
                "dataset": dataset_name,
                "num_items": num_items,
                "num_scored": num_items,
                "num_correct": num_correct,
                "accuracy": (num_correct / num_items) if num_items else None,
                "exec_graded": exec_count,
                "generation_errors": error_count,
            }
        )
    return summary_rows


def _save_coding_checkpoint(
    *,
    output_dir: Path,
    results: list[dict[str, Any]],
    eval_order_lookup: dict[str, int],
    completed_evals: int,
    total_evals: int,
) -> None:
    ordered = order_results(results, eval_order_lookup)
    write_jsonl(output_dir / "responses.jsonl", ordered)
    summary_rows = _summarize_coding_results(ordered)
    _write_coding_summary_csv(output_dir / "summary.csv", summary_rows)
    checkpoint = {
        "completed_evals": completed_evals,
        "total_evals": total_evals,
        "saved_at": datetime.now().isoformat(),
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def _print_coding_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nAccuracy summary")
    print("-" * 80)
    for row in summary_rows:
        accuracy = row["accuracy"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        print(
            f"{row['model_label']:24} {row['dataset']:22} "
            f"acc={accuracy_text:>8} "
            f"scored={row['num_scored']:4}/{row['num_items']:4} "
            f"exec={row['exec_graded']:4} errors={row['generation_errors']:3}"
        )
    print("-" * 80)


def _evaluate_coding_item(
    item: dict[str, Any],
    *,
    selected_model_specs: list[Any],
    clients: dict[str, Any],
    completed_keys: set[str],
    generation_max_tokens: int | None,
    generation_timeout: int,
    model_max_tokens: dict[str, int | None],
    model_generation_timeouts: dict[str, int | None] | None,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    item = _hydrate_coding_sampled_item(item)
    item_results: list[tuple[str, dict[str, Any]]] = []

    for model_spec in selected_model_specs:
        eval_key = make_eval_key(item, model_spec.label)
        if eval_key in completed_keys:
            continue

        record: dict[str, Any] = {
            "dataset": item["dataset"],
            "dataset_kind": item["dataset_kind"],
            "sample_index": item["sample_index"],
            "model_label": model_spec.label,
            "model_provider": model_spec.provider,
            "model_id": model_spec.model_id,
            "level": item["level"],
            "subject": item["subject"],
            "question": item["question"],
            "prompt": item["prompt"],
            "status": "ok",
            "correct": None,
            "grading_method": "exec_tests",
            "response_text": None,
            "extracted_code": None,
            "latency_s": None,
        }

        effective_max_tokens = model_max_tokens.get(model_spec.label, generation_max_tokens) or generation_max_tokens
        per_model_timeout = (model_generation_timeouts or {}).get(model_spec.label)
        effective_timeout = per_model_timeout if per_model_timeout is not None else generation_timeout
        started = time.time()
        try:
            response_text = call_model(
                model_spec,
                clients,
                item["prompt"],
                generation_max_tokens=effective_max_tokens,
                generation_timeout=effective_timeout,
                use_litellm=use_litellm,
                litellm_models=litellm_models,
            )
            record["response_text"] = response_text
            record["latency_s"] = round(time.time() - started, 2)
        except Exception:
            record["status"] = "generation_error"
            record["correct"] = False
            record["grading_method"] = "generation_error"
            record["latency_s"] = round(time.time() - started, 2)
            item_results.append((eval_key, record))
            continue

        graded = _grade_code_response(item, record["response_text"] or "")
        record["correct"] = graded["passed"]
        record["extracted_code"] = graded.get("extracted_code")
        item_results.append((eval_key, record))

    return item_results


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def call_openai_compatible(
    client: OpenAI,
    model_id: str,
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
    generation_timeout: int | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    request_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if not model_id.startswith("gpt-5"):
        request_kwargs["temperature"] = temperature
    if max_tokens is not None:
        if model_id.startswith("gpt-5"):
            request_kwargs["max_completion_tokens"] = max_tokens
        else:
            request_kwargs["max_tokens"] = max_tokens
    try:
        resp = client.chat.completions.create(
            **request_kwargs,
            **({"timeout": generation_timeout} if generation_timeout is not None else {}),
        )
    except BadRequestError as exc:
        message = str(exc).lower()
        timeout_kw = {"timeout": generation_timeout} if generation_timeout is not None else {}
        if (
            "max_tokens" in request_kwargs
            and "unsupported" in message
            and "max_tokens" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("max_tokens", None)
            retry_kwargs["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**retry_kwargs, **timeout_kw)
        elif (
            "max_completion_tokens" in request_kwargs
            and "unsupported" in message
            and "max_completion_tokens" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("max_completion_tokens", None)
            retry_kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**retry_kwargs, **timeout_kw)
        elif (
            "temperature" in request_kwargs
            and "unsupported" in message
            and "temperature" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**retry_kwargs, **timeout_kw)
        else:
            raise

    return extract_chat_content(resp.choices[0].message.content)


def call_openai_responses(
    client: OpenAI,
    model_id: str,
    *,
    user_prompt: str,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    generation_timeout: int | None = None,
) -> str:
    request_kwargs: dict[str, Any] = {"model": model_id, "input": user_prompt}
    if max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = max_output_tokens
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    if generation_timeout is not None:
        request_kwargs["timeout"] = generation_timeout
    resp = client.responses.create(**request_kwargs)
    return resp.output_text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current 16-model roster on static benchmarks (math or misc).",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["math", "misc", "coding"],
        help="Evaluation domain: 'math' or 'misc' uses LLM judge; 'coding' uses execution harness.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional YAML config file. Evaluation settings are read from "
            "config[domain]['evaluation']."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=[spec.label for spec in MODEL_SPECS],
        help="Models to evaluate. Defaults to all models in MODEL_SPECS.",
    )
    parser.add_argument(
        "--sample-file",
        default=None,
        help="Path to a previously generated sampled_items.jsonl file.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI judge model used for correctness grading.",
    )
    parser.add_argument(
        "--generation-timeout",
        type=int,
        default=None,
        help="Provider API timeout seconds for generation requests.",
    )
    parser.add_argument(
        "--generation-max-tokens",
        type=int,
        default=None,
        dest="generation_max_tokens",
        help="Maximum tokens for generation (default from config: 32768).",
    )
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=None,
        help="Number of retries for empty or failed generation attempts (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results. Defaults to results/static_eval/<timestamp>.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Periodically save partial responses and summary every N evaluations (default: 50).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing partial results in the output directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of question items to evaluate in parallel (default: 4). Use 1 for serial execution.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="(coding only) Optional limit on the number of sampled items to evaluate.",
    )
    parser.add_argument(
        "--use-litellm",
        action="store_true",
        help="Route all model and judge calls through a single OpenAI-compatible LiteLLM endpoint.",
    )
    parser.add_argument(
        "--litellm-models",
        nargs="+",
        default=None,
        choices=[spec.label for spec in MODEL_SPECS],
        help=(
            "Explicit subset of model labels to route through LiteLLM. "
            "When set, this overrides the default all-model behavior."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config / environment helpers
# ---------------------------------------------------------------------------

def build_output_dir(user_output_dir: str | None) -> Path:
    if user_output_dir:
        return Path(user_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "static_eval" / timestamp


def resolve_env_path() -> Path:
    here = Path(__file__).resolve()
    local_env = here.with_name(".env")
    if local_env.exists():
        return local_env
    return here.parent.parent / ".env"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _expand_env(raw)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    # The coding domain config uses a flat top-level "evaluation:" section
    # (matching the original eval_static_coding.py convention), while math/misc
    # use config[domain]["evaluation"].
    if args.domain == "coding":
        section = config.get("evaluation", {})
        sampling_section = config.get("sampling", {})
    else:
        section = config.get(args.domain, {}).get("evaluation", {})
        sampling_section = config.get(args.domain, {}).get("sampling", {})

    if args.models is None:
        args.models = section.get("models", [spec.label for spec in MODEL_SPECS])
    if args.sample_file is None:
        args.sample_file = section.get("sample_file")
        if args.sample_file is None:
            sampling_output_dir = sampling_section.get("output_dir")
            if sampling_output_dir:
                args.sample_file = str(Path(sampling_output_dir) / "sampled_items.jsonl")
    if args.judge_model is None:
        args.judge_model = section.get("judge_model", "gpt-4.1-mini")
    if args.generation_timeout is None:
        default_gen_timeout = 120 if args.domain != "coding" else 120
        args.generation_timeout = int(section.get("generation_timeout", default_gen_timeout))
    if args.generation_max_tokens is None:
        default_max_tokens = 32768 if args.domain != "coding" else 16384
        args.generation_max_tokens = int(section.get("generation_max_tokens", default_max_tokens))
    if args.generation_retries is None:
        args.generation_retries = int(section.get("generation_retries", 2))
    args.model_max_tokens = {
        key: (int(value) if value is not None else None)
        for key, value in section.get("model_max_tokens", {}).items()
    }
    args.model_generation_timeouts = {
        key: (int(value) if value is not None else None)
        for key, value in section.get("model_generation_timeouts", {}).items()
    }
    if args.output_dir is None:
        args.output_dir = section.get("output_dir")
    if args.save_every == 50:
        args.save_every = int(section.get("save_every", 50))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    if args.max_workers == 4:
        args.max_workers = int(section.get("max_workers", 4))
    if args.max_items is None and section.get("max_items") is not None:
        args.max_items = int(section.get("max_items"))
    if not args.use_litellm:
        args.use_litellm = bool(section.get("use_litellm", False))
    if args.litellm_models is None and section.get("litellm_models") is not None:
        args.litellm_models = list(section.get("litellm_models", []))

    if not args.sample_file:
        raise SystemExit("Error: --sample-file is required, either via CLI or config file.")
    return args


# ---------------------------------------------------------------------------
# Sampling helpers (math domain only; misc sampling lives in sample_static.py)
# ---------------------------------------------------------------------------

def build_sample_plan(selected_datasets: list[str], profile: str, uniform_n: int | None) -> dict[str, int]:
    plan: dict[str, int] = {}
    for name in selected_datasets:
        spec = DATASET_LOOKUP[name]
        if uniform_n is not None:
            plan[name] = uniform_n
        elif profile == "paper":
            plan[name] = spec.default_paper_samples
        else:
            plan[name] = spec.default_pilot_samples
    return plan


# ---------------------------------------------------------------------------
# Client builders
# ---------------------------------------------------------------------------

def build_clients(selected_models: list[str], generation_timeout: int) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    providers = {MODEL_LOOKUP[name].provider for name in selected_models}

    if "openai" in providers or True:
        key = get_env_value("OPENAI_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("OPENAI_BASE_URL"), "openai")
            clients["openai"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)

    if "anthropic" in providers:
        key = get_env_value("ANTHROPIC_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("ANTHROPIC_BASE_URL"), "anthropic")
            clients["anthropic"] = anthropic.Anthropic(
                api_key=key,
                base_url=base_url,
                timeout=generation_timeout,
            )

    if "google" in providers:
        key = get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if key:
            base_url = normalize_base_url(get_env_value("GEMINI_BASE_URL"), "google")
            if base_url:
                clients["google"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)

    if "openrouter" in providers:
        key = get_env_value("OPENROUTER_API_KEY")
        if key:
            clients["openrouter"] = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                timeout=generation_timeout,
            )

    if "openai" not in clients:
        sys.exit("OPENAI_API_KEY is required for the judge model.")
    clients["_generation_timeout"] = generation_timeout
    return clients


def build_clients_with_mode(
    selected_models: list[str],
    generation_timeout: int,
    *,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> dict[str, Any]:
    clients = build_clients(selected_models, generation_timeout)
    if use_litellm or litellm_models:
        key = get_env_value("LITELLM_API_KEY", "OPENAI_API_KEY")
        base_url = normalize_base_url(
            get_env_value("LITELLM_BASE_URL", "OPENAI_BASE_URL"),
            "openai",
        )
        if not key or not base_url:
            sys.exit(
                "When --use-litellm is enabled, set LITELLM_API_KEY (or OPENAI_API_KEY) "
                "and LITELLM_BASE_URL (or OPENAI_BASE_URL)."
            )
        clients["litellm"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)
    return clients


def _should_use_litellm(spec, *, use_litellm: bool, litellm_models: set[str] | None) -> bool:
    if litellm_models is not None:
        return spec.label in litellm_models and spec.litellm_model_id is not None
    return should_use_litellm_for_model(spec, use_litellm=use_litellm)


# ---------------------------------------------------------------------------
# Dataset row loading (math domain)
# ---------------------------------------------------------------------------

def load_raw_rows(spec: DatasetSpec) -> list[dict]:
    if spec.hf_config is None:
        dataset = load_dataset(spec.hf_path)[spec.split]
    else:
        dataset = load_dataset(spec.hf_path, spec.hf_config)[spec.split]

    rows = [dict(row) for row in dataset]
    if spec.kind == "olympiad":
        rows = [
            row
            for row in rows
            if row.get("subject") == "Math"
            and row.get("language") == "English"
            and row.get("modality") == "Text-only"
            and not row.get("is_multiple_answer", False)
        ]
    if spec.kind == "hle":
        # Filter to math questions and sanitize each row.
        # HF materialises all Image features as PngImageFile objects (including
        # fields we don't know about), so we recursively strip any value that
        # isn't a JSON primitive rather than trying to name specific keys.
        rows = [
            _sanitize_for_json(row)
            for row in rows
            if "math" in str(row.get("category", "")).lower()
        ]
    return rows


def sample_rows(rows: list[dict], *, n: int, seed: int, stratify_field: str | None = None) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    if not stratify_field:
        return rng.sample(rows, n)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(stratify_field, "unknown"))].append(row)

    buckets = list(grouped.values())
    if not buckets:
        return rng.sample(rows, n)

    per_bucket = max(1, n // len(buckets))
    sampled: list[dict] = []
    leftovers: list[dict] = []
    for bucket in buckets:
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        take = min(per_bucket, len(shuffled))
        sampled.extend(shuffled[:take])
        leftovers.extend(shuffled[take:])

    if len(sampled) < n:
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: n - len(sampled)])
    elif len(sampled) > n:
        rng.shuffle(sampled)
        sampled = sampled[:n]
    return sampled


def format_mmlu_question(item: dict) -> str:
    letters = ["A", "B", "C", "D", "E", "F"]
    options = item.get("options", [])
    option_lines = [f"({letters[idx]}) {option}" for idx, option in enumerate(options)]
    return f"{item['centerpiece']}\n\nOptions:\n" + "\n".join(option_lines)


def build_eval_prompt(dataset_spec: DatasetSpec, item: dict) -> str:
    return f"{build_raw_question(dataset_spec, item)}\n\n{ANSWER_INSTRUCTIONS['math']}"


def build_raw_question(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "gsm8k":
        return item["question"]
    if dataset_spec.kind == "mmlu":
        return format_mmlu_question(item)
    if dataset_spec.kind == "aime":
        return item.get("question") or item.get("problem")
    if dataset_spec.kind == "olympiad":
        return item["question"]
    if dataset_spec.kind == "hle":
        return item["question"]
    return item["problem"]


def build_gold_answer(dataset_spec: DatasetSpec, item: dict) -> str:
    if dataset_spec.kind == "gsm8k":
        return item["answer"]
    if dataset_spec.kind == "mmlu":
        letter = item["correct_options"][0]
        literal = item["correct_options_literal"][0]
        return f"Option {letter}: {literal}"
    if dataset_spec.kind == "aime":
        return str(item["answer"])
    if dataset_spec.kind == "olympiad":
        final_answer = item.get("final_answer")
        if isinstance(final_answer, list):
            return " ; ".join(str(part) for part in final_answer)
        return str(final_answer)
    if dataset_spec.kind == "hle":
        return str(item["answer"])
    return item["solution"]


def get_item_metadata(dataset_spec: DatasetSpec, item: dict) -> dict[str, Any]:
    if dataset_spec.kind == "gsm8k":
        return {"level": None, "subject": "word_problems"}
    if dataset_spec.kind == "mmlu":
        return {"level": None, "subject": dataset_spec.name}
    if dataset_spec.kind == "aime":
        metadata = item.get("metadata") or {}
        problem_type = metadata.get("problem_type")
        if isinstance(problem_type, list) and problem_type:
            subject = problem_type[0]
        else:
            subject = "AIME"
        return {"level": "Competition", "subject": subject}
    if dataset_spec.kind == "olympiad":
        return {
            "level": item.get("difficulty", "Competition"),
            "subject": item.get("subfield") or item.get("subject") or "Olympiad",
        }
    if dataset_spec.kind == "hle":
        return {"level": "Expert", "subject": item.get("category", "Mathematics")}
    return {"level": item.get("level"), "subject": item.get("type")}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_correctness(
    judge_client: OpenAI,
    judge_model: str,
    *,
    domain: str,
    question: str,
    gold_answer: str,
    model_answer: str,
    use_litellm: bool = False,
) -> dict[str, Any]:
    """Grade a single response using the judge model for the given domain."""
    system_instructions = JUDGE_SYSTEM_INSTRUCTIONS[domain]
    prompt_template = JUDGE_PROMPT_TEMPLATES[domain]
    prompt = prompt_template.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
    )
    if use_litellm:
        request_kwargs: dict[str, Any] = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = judge_client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            message = str(exc).lower()
            if "response_format" in request_kwargs and "unsupported" in message and "response_format" in message:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("response_format", None)
                resp = judge_client.chat.completions.create(**retry_kwargs)
            else:
                raise
        raw_text = extract_chat_content(resp.choices[0].message.content).strip()
    else:
        resp = judge_client.responses.create(
            model=judge_model,
            instructions=system_instructions,
            input=prompt,
            max_output_tokens=512,
            store=False,
        )
        raw_text = resp.output_text.strip()
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        correct = bool(parsed.get("correct", False))
        reason = str(parsed.get("reason", ""))
    except json.JSONDecodeError:
        lower = cleaned.lower()
        correct = '"correct": true' in lower or '"correct":true' in lower
        reason = cleaned[:500]
    return {
        "correct": correct,
        "method": "llm_judge",
        "judge_raw": raw_text,
        "judge_reason": reason,
    }


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------

def call_model(
    spec,
    clients: dict[str, Any],
    prompt: str,
    *,
    generation_max_tokens: int | None,
    generation_timeout: int | None = None,
    use_litellm: bool = False,
    litellm_models: set[str] | None = None,
) -> str:
    # Gemini 2.5/3.1 are thinking models that can consume a large output budget
    # before they emit the final answer, so match the arena eval headroom here.
    google_max_tokens = generation_max_tokens
    if spec.provider == "google" and any(prefix in spec.model_id for prefix in ("2.5", "3.1")):
        google_max_tokens = 65536
    if _should_use_litellm(spec, use_litellm=use_litellm, litellm_models=litellm_models):
        return call_openai_compatible(
            clients["litellm"],
            get_routed_model_id(spec, use_litellm=True),
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
            generation_timeout=generation_timeout,
        )
    if spec.provider == "openai":
        return call_openai_responses(
            clients["openai"],
            spec.model_id,
            user_prompt=prompt,
            max_output_tokens=generation_max_tokens,
            reasoning_effort=spec.reasoning_effort,
            generation_timeout=generation_timeout,
        )
    if spec.provider == "google":
        # Google's API sends HTTP keepalives while thinking, which resets the
        # httpx read timeout. Enforce a hard wall-clock timeout via a thread.
        _generation_timeout = generation_timeout or clients.get("_generation_timeout")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(
                call_openai_compatible,
                clients["google"],
                spec.model_id,
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=google_max_tokens,
            )
            try:
                return _fut.result(timeout=_generation_timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Request timed out after {_generation_timeout}s.")
    if spec.provider == "openrouter":
        return call_openai_compatible(
            clients["openrouter"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
            generation_timeout=generation_timeout,
        )
    if spec.provider == "anthropic":
        anthropic_kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": generation_max_tokens or 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
        if spec.effort:
            anthropic_kwargs["extra_body"] = {"output_config": {"effort": spec.effort}}
        if generation_timeout is not None:
            anthropic_kwargs["timeout"] = generation_timeout
        resp = clients["anthropic"].messages.create(**anthropic_kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
    raise ValueError(f"Unsupported provider: {spec.provider}")


# ---------------------------------------------------------------------------
# JSON / JSONL utilities
# ---------------------------------------------------------------------------

def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace any non-JSON-serializable value (e.g. PIL Images) with None."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def make_eval_key(item_or_record: dict[str, Any], model_label: str | None = None) -> str:
    label = model_label if model_label is not None else str(item_or_record["model_label"])
    return f"{item_or_record['dataset']}::{item_or_record['sample_index']}::{label}"


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "dataset",
                "num_items",
                "num_scored",
                "num_correct",
                "accuracy",
                "judge_graded",
                "generation_errors",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def build_eval_order_lookup(
    sampled_items: list[dict[str, Any]],
    selected_model_specs: list[Any],
) -> dict[str, int]:
    order_lookup: dict[str, int] = {}
    order = 0
    for item in sampled_items:
        for model_spec in selected_model_specs:
            order_lookup[make_eval_key(item, model_spec.label)] = order
            order += 1
    return order_lookup


def order_results(
    rows: list[dict[str, Any]],
    eval_order_lookup: dict[str, int],
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            eval_order_lookup.get(make_eval_key(row), float("inf")),
            row["dataset"],
            row["sample_index"],
            row["model_label"],
        ),
    )


def save_checkpoint(
    *,
    output_dir: Path,
    results: list[dict[str, Any]],
    eval_order_lookup: dict[str, int],
    completed_evals: int,
    total_evals: int,
) -> None:
    ordered = order_results(results, eval_order_lookup)
    write_jsonl(output_dir / "responses.jsonl", ordered)
    summary_rows = summarize_results(ordered)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    checkpoint = {
        "completed_evals": completed_evals,
        "total_evals": total_evals,
        "saved_at": datetime.now().isoformat(),
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["dataset"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_label, dataset_name), group in sorted(grouped.items()):
        all_binary = [1 if row.get("correct") is True else 0 for row in group]
        judge_count = sum(1 for row in group if row["grading_method"] == "llm_judge")
        error_count = sum(1 for row in group if row["status"] != "ok")
        summary_rows.append(
            {
                "model_label": model_label,
                "dataset": dataset_name,
                "num_items": len(group),
                "num_scored": len(group),
                "num_correct": sum(all_binary),
                "accuracy": mean(all_binary) if all_binary else None,
                "judge_graded": judge_count,
                "generation_errors": error_count,
            }
        )
    return summary_rows


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def evaluate_item(
    item: dict[str, Any],
    *,
    domain: str,
    selected_model_specs: list[Any],
    clients: dict[str, Any],
    judge_model: str,
    completed_keys: set[str],
    generation_max_tokens: int | None,
    generation_timeout: int,
    generation_retries: int,
    model_max_tokens: dict[str, int | None] | None,
    model_generation_timeouts: dict[str, int | None] | None,
    use_litellm: bool,
    litellm_models: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    item_results: list[tuple[str, dict[str, Any]]] = []

    for model_spec in selected_model_specs:
        eval_key = make_eval_key(item, model_spec.label)
        if eval_key in completed_keys:
            continue

        record: dict[str, Any] = {
            "dataset": item["dataset"],
            "dataset_kind": item["dataset_kind"],
            "sample_index": item["sample_index"],
            "model_label": model_spec.label,
            "model_provider": model_spec.provider,
            "model_id": model_spec.model_id,
            "level": item.get("level"),
            "subject": item.get("subject"),
            "question": item["question"],
            "prompt": item["prompt"],
            "gold_answer": item["gold_answer"],
            "status": "ok",
            "correct": None,
            "grading_method": "llm_judge",
            "response_text": None,
            "judge_reason": None,
            "judge_raw": None,
            "latency_s": None,
            "generation_attempts": 0,
        }

        started = time.time()
        effective_max_tokens = (model_max_tokens or {}).get(model_spec.label, generation_max_tokens) or generation_max_tokens
        per_model_timeout = (model_generation_timeouts or {}).get(model_spec.label)
        effective_timeout = per_model_timeout if per_model_timeout is not None else generation_timeout
        response_text: str | None = None
        last_generation_error: str | None = None
        for attempt in range(generation_retries + 1):
            record["generation_attempts"] = attempt + 1
            try:
                candidate = call_model(
                    model_spec,
                    clients,
                    item["prompt"],
                    generation_max_tokens=effective_max_tokens,
                    generation_timeout=effective_timeout,
                    use_litellm=use_litellm,
                    litellm_models=litellm_models,
                )
                if candidate.strip():
                    response_text = candidate
                    break
                last_generation_error = "Model returned an empty response after extraction."
            except Exception as exc:
                last_generation_error = str(exc)
            if attempt < generation_retries:
                time.sleep(min(2 ** attempt, 4))

        record["latency_s"] = round(time.time() - started, 2)
        if response_text is None:
            record["status"] = "generation_error"
            record["correct"] = False
            record["grading_method"] = "generation_error"
            record["judge_reason"] = last_generation_error or "Unknown generation failure."
            item_results.append((eval_key, record))
            continue
        record["response_text"] = response_text

        judge_spec = MODEL_LOOKUP.get(judge_model)
        judge_uses_litellm = (
            judge_spec is not None
            and _should_use_litellm(
                judge_spec,
                use_litellm=use_litellm,
                litellm_models=litellm_models,
            )
        )
        effective_judge_model = (
            get_routed_model_id(judge_spec, use_litellm=True)
            if judge_uses_litellm and judge_spec is not None
            else judge_model
        )
        judged = judge_correctness(
            clients["litellm"] if judge_uses_litellm else clients["openai"],
            effective_judge_model,
            domain=domain,
            question=item["question"],
            gold_answer=item["gold_answer"],
            model_answer=record["response_text"] or "",
            use_litellm=judge_uses_litellm,
        )
        record["correct"] = judged["correct"]
        record["grading_method"] = judged["method"]
        record["judge_reason"] = judged["judge_reason"]
        record["judge_raw"] = judged["judge_raw"]
        item_results.append((eval_key, record))

    return item_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nAccuracy summary")
    print("-" * 80)
    for row in summary_rows:
        accuracy = row["accuracy"]
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
        print(
            f"{row['model_label']:24} {row['dataset']:32} "
            f"acc={accuracy_text:>8} "
            f"scored={row['num_scored']:4}/{row['num_items']:4} "
            f"judge={row['judge_graded']:4} errors={row['generation_errors']:3}"
        )
    print("-" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    if args.domain == "coding":
        _main_coding(args)
    else:
        _main_judge(args)


def _main_judge(args: argparse.Namespace) -> None:
    """Entry point for math/misc domains (LLM judge grading)."""
    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    litellm_models = set(args.litellm_models) if args.litellm_models else None
    selected_labels = {spec.label for spec in selected_model_specs}
    if litellm_models is not None:
        unselected = sorted(litellm_models - selected_labels)
        if unselected:
            sys.exit(
                "LiteLLM subset includes models not in --models/config: "
                + ", ".join(unselected)
            )
        unsupported_subset = sorted(
            label for label in litellm_models if MODEL_LOOKUP[label].litellm_model_id is None
        )
        if unsupported_subset:
            sys.exit(
                "LiteLLM is not configured for these requested subset models: "
                + ", ".join(unsupported_subset)
            )
    elif args.use_litellm:
        unsupported = [spec.label for spec in selected_model_specs if spec.litellm_model_id is None]
        if unsupported:
            sys.exit(
                "LiteLLM is not configured for these models: "
                + ", ".join(unsupported)
                + ". Remove them from the config/models list or add mappings in MODEL_SPECS."
            )
    clients = build_clients_with_mode(
        args.models,
        args.generation_timeout,
        use_litellm=args.use_litellm,
        litellm_models=litellm_models,
    )
    sample_file = Path(args.sample_file)
    sampled_items = load_jsonl(sample_file)
    dataset_counts: dict[str, int] = defaultdict(int)
    for item in sampled_items:
        dataset_counts[str(item["dataset"])] += 1

    run_config = {
        "domain": args.domain,
        "models": args.models,
        "sample_file": str(sample_file),
        "datasets": sorted(dataset_counts.keys()),
        "sample_plan": dict(dataset_counts),
        "judge_model": args.judge_model,
        "generation_timeout": args.generation_timeout,
        "generation_max_tokens": args.generation_max_tokens,
        "generation_retries": args.generation_retries,
        "model_max_tokens": args.model_max_tokens,
        "model_generation_timeouts": args.model_generation_timeouts,
        "use_litellm": args.use_litellm,
        "litellm_models": sorted(litellm_models) if litellm_models is not None else None,
        "max_workers": args.max_workers,
        "output_dir": str(output_dir),
        "resume": args.resume,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    responses_path = output_dir / "responses.jsonl"
    results: list[dict[str, Any]] = load_jsonl(responses_path) if args.resume else []
    completed_keys = {make_eval_key(record) for record in results}
    eval_order_lookup = build_eval_order_lookup(sampled_items, selected_model_specs)
    total_evals = len(sampled_items) * len(selected_model_specs)
    completed_evals = len(completed_keys)
    progress = tqdm(
        total=total_evals,
        initial=completed_evals,
        desc=f"Evaluating static {args.domain}",
        unit="eval",
    )

    pending_items = [
        item
        for item in sampled_items
        if any(make_eval_key(item, model_spec.label) not in completed_keys for model_spec in selected_model_specs)
    ]
    completed_keys_snapshot = set(completed_keys)

    def record_result(eval_key: str, record: dict[str, Any]) -> None:
        nonlocal completed_evals
        results.append(record)
        completed_keys.add(eval_key)
        progress.set_postfix_str(
            f"{record['model_label']} | {record['dataset']} #{record['sample_index'] + 1}/{dataset_counts[record['dataset']]}",
            refresh=False,
        )
        progress.update(1)
        completed_evals += 1
        if args.save_every > 0 and completed_evals % args.save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                results=results,
                eval_order_lookup=eval_order_lookup,
                completed_evals=completed_evals,
                total_evals=total_evals,
            )

    if args.max_workers <= 1 or len(pending_items) <= 1:
        for item in pending_items:
            item_results = evaluate_item(
                item,
                domain=args.domain,
                selected_model_specs=selected_model_specs,
                clients=clients,
                judge_model=args.judge_model,
                completed_keys=completed_keys_snapshot,
                generation_max_tokens=args.generation_max_tokens,
                generation_timeout=args.generation_timeout,
                generation_retries=args.generation_retries,
                model_max_tokens=args.model_max_tokens,
                model_generation_timeouts=args.model_generation_timeouts,
                use_litellm=args.use_litellm,
                litellm_models=litellm_models,
            )
            for eval_key, record in item_results:
                record_result(eval_key, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_item,
                    item,
                    domain=args.domain,
                    selected_model_specs=selected_model_specs,
                    clients=clients,
                    judge_model=args.judge_model,
                    completed_keys=completed_keys_snapshot,
                    generation_max_tokens=args.generation_max_tokens,
                    generation_timeout=args.generation_timeout,
                    generation_retries=args.generation_retries,
                    model_max_tokens=args.model_max_tokens,
                    model_generation_timeouts=args.model_generation_timeouts,
                    use_litellm=args.use_litellm,
                    litellm_models=litellm_models,
                )
                for item in pending_items
            ]
            for future in concurrent.futures.as_completed(futures):
                item_results = future.result()
                for eval_key, record in item_results:
                    record_result(eval_key, record)

    progress.close()
    ordered_results = order_results(results, eval_order_lookup)
    summary_rows = summarize_results(ordered_results)
    save_checkpoint(
        output_dir=output_dir,
        results=results,
        eval_order_lookup=eval_order_lookup,
        completed_evals=completed_evals,
        total_evals=total_evals,
    )

    print_summary(summary_rows)
    print(f"\nSaved results to {output_dir}")


def _main_coding(args: argparse.Namespace) -> None:
    """Entry point for the coding domain (execution harness grading)."""
    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    litellm_models = set(args.litellm_models) if args.litellm_models else None
    clients = build_clients_with_mode(
        args.models,
        args.generation_timeout,
        use_litellm=args.use_litellm,
        litellm_models=litellm_models,
    )

    sample_file = Path(args.sample_file)
    sampled_items = load_jsonl(sample_file)
    if getattr(args, "max_items", None) is not None:
        if args.max_items < 0:
            raise SystemExit("Error: --max-items must be non-negative.")
        sampled_items = sampled_items[: args.max_items]

    dataset_counts: dict[str, int] = defaultdict(int)
    for item in sampled_items:
        dataset_counts[str(item["dataset"])] += 1

    run_config = {
        "domain": "coding",
        "models": args.models,
        "sample_file": str(sample_file),
        "datasets": sorted(dataset_counts.keys()),
        "sample_plan": dict(dataset_counts),
        "generation_timeout": args.generation_timeout,
        "generation_max_tokens": args.generation_max_tokens,
        "model_max_tokens": args.model_max_tokens,
        "model_generation_timeouts": args.model_generation_timeouts,
        "use_litellm": args.use_litellm,
        "litellm_models": sorted(litellm_models) if litellm_models is not None else None,
        "max_workers": args.max_workers,
        "max_items": getattr(args, "max_items", None),
        "output_dir": str(output_dir),
        "resume": args.resume,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    responses_path = output_dir / "responses.jsonl"
    results: list[dict[str, Any]] = load_jsonl(responses_path) if args.resume else []
    completed_keys = {make_eval_key(record) for record in results}
    eval_order_lookup = build_eval_order_lookup(sampled_items, selected_model_specs)
    total_evals = len(sampled_items) * len(selected_model_specs)
    completed_evals = len(completed_keys)
    progress = tqdm(
        total=total_evals,
        initial=completed_evals,
        desc="Evaluating static coding",
        unit="eval",
    )

    pending_items = [
        item
        for item in sampled_items
        if any(make_eval_key(item, model_spec.label) not in completed_keys for model_spec in selected_model_specs)
    ]
    completed_keys_snapshot = set(completed_keys)

    def record_result(eval_key: str, record: dict[str, Any]) -> None:
        nonlocal completed_evals
        results.append(record)
        completed_keys.add(eval_key)
        progress.set_postfix_str(
            f"{record['model_label']} | {record['dataset']} #{record['sample_index'] + 1}/{dataset_counts[record['dataset']]}",
            refresh=False,
        )
        progress.update(1)
        completed_evals += 1
        if args.save_every > 0 and completed_evals % args.save_every == 0:
            _save_coding_checkpoint(
                output_dir=output_dir,
                results=results,
                eval_order_lookup=eval_order_lookup,
                completed_evals=completed_evals,
                total_evals=total_evals,
            )

    if args.max_workers <= 1 or len(pending_items) <= 1:
        for item in pending_items:
            item_results = _evaluate_coding_item(
                item,
                selected_model_specs=selected_model_specs,
                clients=clients,
                completed_keys=completed_keys_snapshot,
                generation_max_tokens=args.generation_max_tokens,
                generation_timeout=args.generation_timeout,
                model_max_tokens=args.model_max_tokens,
                model_generation_timeouts=args.model_generation_timeouts,
                use_litellm=args.use_litellm,
                litellm_models=litellm_models,
            )
            for eval_key, record in item_results:
                record_result(eval_key, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_coding_item,
                    item,
                    selected_model_specs=selected_model_specs,
                    clients=clients,
                    completed_keys=completed_keys_snapshot,
                    generation_max_tokens=args.generation_max_tokens,
                    generation_timeout=args.generation_timeout,
                    model_max_tokens=args.model_max_tokens,
                    model_generation_timeouts=args.model_generation_timeouts,
                    use_litellm=args.use_litellm,
                    litellm_models=litellm_models,
                )
                for item in pending_items
            ]
            for future in concurrent.futures.as_completed(futures):
                item_results = future.result()
                for eval_key, record in item_results:
                    record_result(eval_key, record)

    progress.close()
    ordered_results = order_results(results, eval_order_lookup)
    summary_rows = _summarize_coding_results(ordered_results)
    _save_coding_checkpoint(
        output_dir=output_dir,
        results=results,
        eval_order_lookup=eval_order_lookup,
        completed_evals=completed_evals,
        total_evals=total_evals,
    )
    _print_coding_summary(summary_rows)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
