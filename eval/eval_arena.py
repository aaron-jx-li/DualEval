#!/usr/bin/env python3
"""
Unified multi-domain Arena evaluation — response generation only.

For each Arena question in the selected domain, this script generates one
answer from each selected model and writes the results to responses.jsonl
with reward=null. Reward scoring is done separately via
reward/public_reward.py or your own reward model.

Example:
  python eval/eval_arena.py --domain math    --config demo/config_arena.yaml
  python eval/eval_arena.py --domain coding  --config demo/config_arena.yaml
  python eval/eval_arena.py --domain misc    --config demo/config_arena.yaml
  python eval/eval_arena.py --domain generic --config demo/config_arena.yaml
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from openai import BadRequestError, OpenAI
from tqdm import tqdm
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_static import (  # noqa: E402
    MODEL_LOOKUP,
    MODEL_SPECS,
    extract_chat_content,
    get_routed_model_id,
    get_env_value,
    load_env_file,
    normalize_base_url,
    should_use_litellm_for_model,
)


DEFAULT_MODELS = [spec.label for spec in MODEL_SPECS]

DOMAIN_ANSWER_INSTRUCTIONS = {
    "math": (
        "Solve the math problem carefully. "
        "End your response with a final line of the form 'Final answer: <answer>'."
    ),
    "coding": (
        "Provide a complete, correct solution. "
        "Include any necessary code, explanations, or steps as appropriate."
    ),
    "misc": (
        "Provide a complete, helpful response. "
        "Include explanations, steps, or code only when useful for the prompt."
    ),
    "generic": (
        "Provide a complete, helpful response. "
        "Include explanations, steps, or code only when useful for the prompt."
    ),
}

DOMAIN_ITEM_PREFIXES = {
    "math": "arena_math_",
    "coding": "arena_coding_",
    "misc": "arena_misc_",
    "generic": "arena_generic_",
}

DOMAIN_TQDM_DESCS = {
    "math": "Evaluating Arena math",
    "coding": "Evaluating Arena coding",
    "misc": "Evaluating Arena misc",
    "generic": "Evaluating Arena generic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model responses for Arena questions across multiple domains.",
    )
    parser.add_argument(
        "--domain",
        choices=("math", "coding", "misc", "generic"),
        required=True,
        help="Evaluation domain.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Settings are read from the '<domain>.evaluation' section.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Arena input file. Supports JSON array or JSONL.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=DEFAULT_MODELS,
        help="Models to evaluate. Defaults to all models in MODEL_SPECS.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of Arena questions to evaluate in parallel.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Write checkpoint every N completed evals.",
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
        "--use-litellm",
        action="store_true",
        help="Route supported model calls through LiteLLM and fall back for the rest.",
    )
    parser.add_argument(
        "--litellm-models",
        nargs="+",
        default=None,
        choices=DEFAULT_MODELS,
        help=(
            "Explicit subset of model labels to route through LiteLLM. "
            "When set, this overrides the default Claude-only routing."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing responses.jsonl in the output directory.",
    )
    parser.add_argument(
        "--retry-statuses",
        nargs="*",
        default=None,
        choices=("generation_error",),
        help=(
            "When resuming, retry prior rows with these statuses instead of "
            "treating them as completed. Defaults to config or ['generation_error']."
        ),
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        default=False,
        help="When resuming, treat all errored rows as completed and skip them.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of Arena questions.",
    )
    parser.add_argument(
        "--first-k",
        type=int,
        default=None,
        dest="first_k",
        help="Use only first k items (domains: coding, generic).",
    )
    return parser.parse_args()


def resolve_env_path() -> Path:
    local_env = HERE / ".env"
    if local_env.exists():
        return local_env
    return REPO_ROOT / ".env"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: str(get_env_value(m.group(1)) or m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    section = config.get(args.domain, {}).get("evaluation", {})

    if args.input_file is None:
        args.input_file = section.get("input_file", "data/arena_math_900.json")
    if args.models is None:
        args.models = section.get("models", list(DEFAULT_MODELS))
    if args.output_dir is None:
        args.output_dir = section.get("output_dir", "results/static_eval/arena_math_v0")
    if args.max_workers is None:
        args.max_workers = int(section.get("max_workers", 4))
    if args.save_every is None:
        args.save_every = int(section.get("save_every", 50))
    if args.generation_timeout is None:
        args.generation_timeout = int(section.get("generation_timeout", 120))
    if not args.use_litellm:
        args.use_litellm = bool(section.get("use_litellm", False))
    if args.litellm_models is None and section.get("litellm_models") is not None:
        args.litellm_models = list(section.get("litellm_models", []))
    if not args.resume:
        args.resume = bool(section.get("resume", False))
    if args.retry_statuses is None:
        args.retry_statuses = list(section.get("retry_statuses", ["generation_error"]))
    if args.no_retry:
        args.retry_statuses = []
    if args.max_rows is None and section.get("max_rows") is not None:
        args.max_rows = int(section.get("max_rows"))
    if args.generation_max_tokens is None:
        args.generation_max_tokens = int(section.get("generation_max_tokens", 32768))
    args.model_max_tokens = {k: (int(v) if v is not None else None) for k, v in section.get("model_max_tokens", {}).items()}
    if args.first_k is None:
        args.first_k = section.get("first_k")
    return args


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise SystemExit(f"Expected a JSON array in {path}.")
        return [dict(row) for row in parsed]

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(dict(json.loads(line)))
    return rows


def build_question_items(rows: list[dict[str, Any]], domain: str, answer_instruction: str) -> list[dict[str, Any]]:
    item_prefix = DOMAIN_ITEM_PREFIXES[domain]
    items: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        question = str(row.get("question") or row.get("prompt") or "").strip()
        if not question:
            continue
        item_id = str(row.get("id") or f"{item_prefix}{row_index:06d}")
        prompt = f"{question}\n\n{answer_instruction}" if answer_instruction else question
        items.append(
            {
                "item_id": item_id,
                "row_index": row_index,
                "question": question,
                "prompt": prompt,
            }
        )
    return items


def call_openai_compatible(
    client: OpenAI,
    model_id: str,
    *,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
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
        resp = client.chat.completions.create(**request_kwargs)
    except BadRequestError as exc:
        message = str(exc).lower()
        if (
            "max_tokens" in request_kwargs
            and "unsupported" in message
            and "max_tokens" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("max_tokens", None)
            retry_kwargs["max_completion_tokens"] = max_tokens
            resp = client.chat.completions.create(**retry_kwargs)
        elif (
            "max_completion_tokens" in request_kwargs
            and "unsupported" in message
            and "max_completion_tokens" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("max_completion_tokens", None)
            retry_kwargs["max_tokens"] = max_tokens
            resp = client.chat.completions.create(**retry_kwargs)
        elif (
            "temperature" in request_kwargs
            and "unsupported" in message
            and "temperature" in message
        ):
            retry_kwargs = dict(request_kwargs)
            retry_kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**retry_kwargs)
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
) -> str:
    request_kwargs: dict[str, Any] = {"model": model_id, "input": user_prompt}
    if max_output_tokens is not None:
        request_kwargs["max_output_tokens"] = max_output_tokens
    if reasoning_effort is not None:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    resp = client.responses.create(**request_kwargs)
    return resp.output_text


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

    missing = sorted({MODEL_LOOKUP[name].provider for name in selected_models} - set(clients.keys()) - {"openrouter"})
    if "openrouter" in {MODEL_LOOKUP[name].provider for name in selected_models} and "openrouter" not in clients:
        missing.append("openrouter")
    if missing:
        raise SystemExit(f"Missing API credentials/config for providers: {', '.join(sorted(set(missing)))}")
    clients["_generation_timeout"] = generation_timeout
    return clients


def _should_use_litellm(spec, *, use_litellm: bool, litellm_models: set[str] | None) -> bool:
    if litellm_models is not None:
        return spec.label in litellm_models and spec.litellm_model_id is not None
    return should_use_litellm_for_model(spec, use_litellm=use_litellm)


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
            raise SystemExit(
                "When --use-litellm is enabled, set LITELLM_API_KEY (or OPENAI_API_KEY) "
                "and LITELLM_BASE_URL (or OPENAI_BASE_URL)."
            )
        clients["litellm"] = OpenAI(api_key=key, base_url=base_url, timeout=generation_timeout)
    return clients


def call_model(spec, clients: dict[str, Any], prompt: str, *, generation_max_tokens: int | None, use_litellm: bool = False, litellm_models: set[str] | None = None) -> str:
    # Gemini 2.5/3.1 are thinking models that consume tokens for internal reasoning
    _GOOGLE_THINKING_PREFIXES = ("2.5", "3.1")
    google_max_tokens = (
        65536
        if any(p in spec.model_id for p in _GOOGLE_THINKING_PREFIXES)
        else generation_max_tokens
    )
    if _should_use_litellm(spec, use_litellm=use_litellm, litellm_models=litellm_models):
        return call_openai_compatible(
            clients["litellm"],
            get_routed_model_id(spec, use_litellm=True),
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
        )
    if spec.provider == "openai":
        return call_openai_responses(
            clients["openai"],
            spec.model_id,
            user_prompt=prompt,
            max_output_tokens=generation_max_tokens,
            reasoning_effort=spec.reasoning_effort,
        )
    if spec.provider == "google":
        _timeout = clients.get("_generation_timeout")
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
                return _fut.result(timeout=_timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Request timed out after {_timeout}s.")
    if spec.provider == "openrouter":
        return call_openai_compatible(
            clients["openrouter"],
            spec.model_id,
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=generation_max_tokens,
        )
    if spec.provider == "anthropic":
        anthropic_kwargs: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": generation_max_tokens or 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
        if spec.effort:
            anthropic_kwargs["extra_body"] = {"output_config": {"effort": spec.effort}}
        resp = clients["anthropic"].messages.create(**anthropic_kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
    raise ValueError(f"Unsupported provider: {spec.provider}")


def make_eval_key(item_or_record: dict[str, Any], model_label: str | None = None) -> str:
    label = model_label if model_label is not None else str(item_or_record["model_label"])
    return f"{item_or_record['item_id']}::{label}"


def build_eval_order_lookup(
    items: list[dict[str, Any]],
    selected_model_specs: list[Any],
) -> dict[str, int]:
    order_lookup: dict[str, int] = {}
    order = 0
    for item in items:
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
            row["item_id"],
            row["model_label"],
        ),
    )


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


def has_empty_response(record: dict[str, Any]) -> bool:
    response_text = record.get("response_text")
    if response_text is None:
        return True
    if isinstance(response_text, str):
        return not response_text.strip()
    return False


def split_resume_results(
    rows: list[dict[str, Any]],
    *,
    retry_statuses: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split prior rows into kept results and rows that should be retried."""
    kept: list[dict[str, Any]] = []
    retried: list[dict[str, Any]] = []
    for row in rows:
        if has_empty_response(row) or row.get("status") in retry_statuses:
            retried.append(row)
        else:
            kept.append(row)
    return kept, retried


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_label"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for model_label, group in sorted(grouped.items()):
        num_generated = sum(1 for row in group if row.get("response_text") is not None and str(row.get("response_text", "")).strip())
        summary_rows.append(
            {
                "model_label": model_label,
                "num_items": len(group),
                "num_generated": num_generated,
                "generation_errors": sum(1 for row in group if row["status"] == "generation_error"),
            }
        )
    return summary_rows


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_label",
                "num_items",
                "num_generated",
                "generation_errors",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


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
    write_summary_csv(output_dir / "summary.csv", summarize_results(ordered))
    checkpoint = {
        "completed_evals": completed_evals,
        "total_evals": total_evals,
        "saved_at": datetime.now().isoformat(),
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def evaluate_item(
    item: dict[str, Any],
    *,
    selected_model_specs: list[Any],
    clients: dict[str, Any],
    completed_keys: set[str],
    use_litellm: bool,
    litellm_models: set[str] | None = None,
    generation_max_tokens: int | None = None,
    model_max_tokens: dict[str, int | None] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    item_results: list[tuple[str, dict[str, Any]]] = []

    for model_spec in selected_model_specs:
        eval_key = make_eval_key(item, model_spec.label)
        if eval_key in completed_keys:
            continue

        record: dict[str, Any] = {
            "item_id": item["item_id"],
            "row_index": item["row_index"],
            "model_label": model_spec.label,
            "model_provider": model_spec.provider,
            "model_id": model_spec.model_id,
            "question": item["question"],
            "prompt": item["prompt"],
            "response_text": None,
            "reward": None,
            "status": "ok",
            "error": None,
            "generation_latency_s": None,
            "reward_latency_s": None,
            "total_latency_s": None,
            "processed_at": None,
        }

        started = time.time()
        generation_started = time.time()
        try:
            effective_max_tokens = (model_max_tokens or {}).get(model_spec.label, generation_max_tokens) or generation_max_tokens
            response_text = call_model(model_spec, clients, item["prompt"], generation_max_tokens=effective_max_tokens, use_litellm=use_litellm, litellm_models=litellm_models)
            record["generation_latency_s"] = round(time.time() - generation_started, 2)
            if not response_text.strip():
                raise ValueError("Empty response text")
            record["response_text"] = response_text
        except Exception as exc:
            record["status"] = "generation_error"
            record["generation_latency_s"] = round(time.time() - generation_started, 2)
            record["total_latency_s"] = round(time.time() - started, 2)
            record["error"] = str(exc)
            record["processed_at"] = datetime.now().isoformat()
            item_results.append((eval_key, record))
            continue

        record["total_latency_s"] = round(time.time() - started, 2)
        record["processed_at"] = datetime.now().isoformat()
        item_results.append((eval_key, record))

    return item_results


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nGeneration summary")
    print("-" * 70)
    for row in summary_rows:
        print(
            f"{row['model_label']:24} "
            f"generated={row['num_generated']:4}/{row['num_items']:4} "
            f"gen_err={row['generation_errors']:3}"
        )
    print("-" * 70)


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())
    args = apply_config_defaults(args)

    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1.")
    if args.save_every < 1:
        raise SystemExit("--save-every must be at least 1.")

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_model_specs = [MODEL_LOOKUP[label] for label in args.models]
    litellm_models = set(args.litellm_models) if args.litellm_models else None
    clients = build_clients_with_mode(
        args.models,
        args.generation_timeout,
        use_litellm=args.use_litellm,
        litellm_models=litellm_models,
    )

    raw_rows = load_input_rows(input_path)
    if args.max_rows is not None:
        raw_rows = raw_rows[: args.max_rows]
    first_k = getattr(args, "first_k", None)
    if first_k is not None:
        raw_rows = raw_rows[:first_k]

    answer_instruction = DOMAIN_ANSWER_INSTRUCTIONS[args.domain]
    items = build_question_items(raw_rows, domain=args.domain, answer_instruction=answer_instruction)

    eval_order_lookup = build_eval_order_lookup(items, selected_model_specs)
    responses_path = output_dir / "responses.jsonl"
    results: list[dict[str, Any]] = []
    retried_results: list[dict[str, Any]] = []
    retry_statuses = set(args.retry_statuses or [])
    if args.resume:
        existing_results = load_jsonl(responses_path)
        results, retried_results = split_resume_results(
            existing_results,
            retry_statuses=retry_statuses,
        )
    completed_keys = {make_eval_key(record) for record in results}
    total_evals = len(items) * len(selected_model_specs)
    completed_evals = len(completed_keys)

    run_config = {
        "domain": args.domain,
        "config": args.config,
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "models": args.models,
        "resume": args.resume,
        "save_every": args.save_every,
        "max_workers": args.max_workers,
        "generation_timeout": args.generation_timeout,
        "use_litellm": args.use_litellm,
        "retry_statuses": sorted(retry_statuses),
        "max_rows": args.max_rows,
        "first_k": args.first_k,
        "total_items": len(items),
        "total_evals": total_evals,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if retried_results:
        retry_counts = Counter(
            "empty_response" if has_empty_response(row) else str(row.get("status", "unknown"))
            for row in retried_results
        )
        retry_summary = ", ".join(f"{status}={count}" for status, count in sorted(retry_counts.items()))
        print(f"Retrying prior failed rows on resume: {retry_summary}")

    progress = tqdm(
        total=total_evals,
        initial=completed_evals,
        desc=DOMAIN_TQDM_DESCS[args.domain],
        unit="eval",
    )

    pending_items = [
        item
        for item in items
        if any(make_eval_key(item, model_spec.label) not in completed_keys for model_spec in selected_model_specs)
    ]
    completed_keys_snapshot = set(completed_keys)

    def record_result(eval_key: str, record: dict[str, Any]) -> None:
        nonlocal completed_evals
        results.append(record)
        completed_keys.add(eval_key)
        progress.set_postfix_str(
            f"{record['model_label']} | question #{record['row_index'] + 1}",
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
                selected_model_specs=selected_model_specs,
                clients=clients,
                completed_keys=completed_keys_snapshot,
                use_litellm=args.use_litellm,
                litellm_models=litellm_models,
                generation_max_tokens=args.generation_max_tokens,
                model_max_tokens=args.model_max_tokens,
            )
            for eval_key, record in item_results:
                record_result(eval_key, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_item,
                    item,
                    selected_model_specs=selected_model_specs,
                    clients=clients,
                    completed_keys=completed_keys_snapshot,
                    use_litellm=args.use_litellm,
                    litellm_models=litellm_models,
                    generation_max_tokens=args.generation_max_tokens,
                    model_max_tokens=args.model_max_tokens,
                )
                for item in pending_items
            ]
            for future in concurrent.futures.as_completed(futures):
                item_results = future.result()
                for eval_key, record in item_results:
                    record_result(eval_key, record)

    progress.close()
    summary_rows = summarize_results(order_results(results, eval_order_lookup))
    save_checkpoint(
        output_dir=output_dir,
        results=results,
        eval_order_lookup=eval_order_lookup,
        completed_evals=completed_evals,
        total_evals=total_evals,
    )
    print_summary(summary_rows)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
