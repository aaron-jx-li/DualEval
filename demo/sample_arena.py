#!/usr/bin/env python3
"""
Unified Arena question sampler: filter math, coding, or generic prompts from
Hugging Face Arena datasets using an LLM judge.

Examples:
  python demo/sample_arena.py --domain math    --config demo/config_arena.yaml
  python demo/sample_arena.py --domain coding  --config demo/config_arena.yaml
  python demo/sample_arena.py --domain generic --config demo/config_arena.yaml

Notes:
  - misc uses pre-existing sampled files and has no sampling script.
  - Config is read from config[domain]["sampling"] section of the unified YAML.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Judge prompts — one system + user template per domain
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM: dict[str, str] = {
    "math": (
        "You are a strict prompt classifier. "
        "Decide whether the user prompt is math-focused."
    ),
    "coding": (
        "You are a strict domain classifier. "
        "Decide whether the user prompt is primarily about coding or software engineering."
    ),
    "generic": (
        "You are a strict quality filter for an LLM evaluation benchmark. "
        "Your job is to identify substantive, non-trivial everyday tasks where "
        "response quality varies meaningfully between capable models."
    ),
}

JUDGE_USER_TEMPLATE: dict[str, str] = {
    "math": """\
Classify whether this user prompt is math-focused.

Definition of math-focused:
- The core task requires mathematical reasoning/calculation/proof/derivation,
  symbolic manipulation, quantitative word-problem solving, geometry, algebra,
  number theory, probability/statistics, optimization, or equation solving.
- Include prompts that are primarily about solving/understanding a math problem.
- Exclude general coding prompts unless math reasoning is central.
- Exclude general trivia, writing, translation, legal/medical advice, etc.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_math": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}}
""",
    "coding": """\
Classify whether this user prompt is coding-focused.

Definition of coding-focused:
- The core task is about writing, debugging, explaining, reviewing, refactoring,
  testing, optimizing, or running code.
- Include software engineering, scripts, SQL, regex, shell commands, APIs,
  frameworks, web development, ML engineering, devops, build systems, and
  code-generation tasks.
- Include requests that ask for code snippets, implementation plans tied to
  code, or fixing program behavior.
- Exclude pure math, pure writing, general knowledge, product advice, business
  strategy, or other non-programming tasks, even if they mention technology.
- Exclude prompts that only ask about using software as an end user unless the
  answer mainly requires programming or technical implementation.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_coding": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label",
  "reason": "short explanation"
}}
""",
    "generic": """\
Decide whether this prompt is suitable for an LLM benchmark that tests \
everyday writing and reasoning quality.

ACCEPT (is_generic: true) only if ALL of the following hold:
1. No specialised domain knowledge is needed — a thoughtful non-expert can
   give a fully competent answer without training in medicine, law, finance,
   science, engineering, history, or any other professional field.
2. No programming knowledge or code is required.
3. No mathematical calculation, proof, or quantitative reasoning is required.
4. Style, tone, clarity, or creativity matters — a better writer produces a
   noticeably better answer than a mediocre one.
5. The prompt is SUBSTANTIVE: it provides enough context or constraints that
   two capable models would produce meaningfully different responses worth
   comparing. A benchmark judge could clearly distinguish a great answer from
   a merely adequate one.

Typical examples that PASS:
- Write / edit / improve an email, message, cover letter, essay, or story
  with specific context or constraints given.
- Give thoughtful advice on a concrete everyday situation (relationships,
  career decisions, travel planning, etc.) with enough detail to reason about.
- Summarise or rewrite a provided piece of text with a specific goal.
- Brainstorm with a clear creative brief (product names for X, slogans for Y).
- Explain a concept with a clear audience and purpose in mind.
- Role-play or conversational tasks with a well-defined scenario.

REJECT (is_generic: false) if ANY of the following is true:
- Answering well requires expert domain knowledge (medical diagnosis, legal
  analysis, financial modelling, scientific derivation, historical scholarship,
  engineering specs, etc.).
- The prompt is about software, code, or technical systems.
- The prompt requires arithmetic, statistics, or any mathematical reasoning.
- The question has a single correct factual answer (trivia, definitions,
  vocabulary lookups, spell-checks, etc.).
- The prompt is a greeting, pleasantry, or casual conversation opener with
  no substantive task ("How are you", "Hey, what's up", "Hello", etc.).
- The prompt is so short and vague that any competent model would give
  essentially the same response — there is nothing to differentiate quality
  (e.g. "Write a short story", "Tell me a joke", "Write a poem").
- The request is for a single joke, riddle, or punchline with no creative
  constraints or context.
- The request is for a single word, synonym, antonym, or trivial vocabulary
  lookup.
- The prompt could be fully answered in one or two sentences with no
  meaningful variation in quality between responses.

When in doubt, reject. This benchmark needs tasks where a great response is
clearly better than a mediocre one, and where that difference comes from
writing skill, judgment, or reasoning — not just from knowing the answer.

Prompt:
{prompt}

Return ONLY JSON:
{{
  "is_generic": true or false,
  "confidence": 0.0 to 1.0,
  "category": "short label (e.g. email writing, creative writing, advice, summarisation, brainstorming, etc.)",
  "reason": "one sentence"
}}""",
}

# The JSON key that signals acceptance for each domain.
DOMAIN_FLAG: dict[str, str] = {
    "math": "is_math",
    "coding": "is_coding",
    "generic": "is_generic",
}

# Statuses that count as a completed (non-error) judge attempt.
_DONE_STATUSES = {"ok", "skipped_no_prompt", "skipped_prefilter", "skipped_language"}


# ──────────────────────────────────────────────────────────────────────────────
# Shared IO utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def resolve_env_path() -> Path:
    here = Path(__file__).resolve()
    local = here.with_name(".env")
    if local.exists():
        return local
    return here.parent.parent / ".env"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml_config(path: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set[str]:
    """Read IDs already written to an output JSONL (used for simple resume)."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(obj.get("id", "")).strip()
            if rid:
                done.add(rid)
    return done


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Prompt extraction (shared across all domains)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t.strip()
    return ""


def _extract_from_serialized(raw: str) -> str:
    """Extract the first user-turn text from a Python-repr-serialized conversation."""
    if not raw:
        return ""
    m = re.search(
        r"'role':\s*'user'.*?'text':\s*(?P<text>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",
        raw,
        flags=re.DOTALL,
    )
    if not m:
        return ""
    try:
        value = ast.literal_eval(m.group("text"))
    except (SyntaxError, ValueError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def extract_user_prompt(row: dict[str, Any]) -> str:
    for key in ("conversation_a", "conversation_b"):
        conv = row.get(key)
        if isinstance(conv, list):
            for msg in conv:
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                    text = _extract_text(msg.get("content"))
                    if text:
                        return text
        elif isinstance(conv, str):
            text = _extract_from_serialized(conv)
            if text:
                return text

    full = row.get("full_conversation")
    if isinstance(full, list):
        for turn in full:
            if not isinstance(turn, dict):
                continue
            user_blob = turn.get("user")
            if isinstance(user_blob, dict):
                text = _extract_text(user_blob.get("content"))
                if text:
                    return text
    elif isinstance(full, str):
        text = _extract_from_serialized(full)
        if text:
            return text

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific pre-filters
# ──────────────────────────────────────────────────────────────────────────────

def _category_tag(row: dict[str, Any]) -> dict[str, Any]:
    ct = row.get("category_tag")
    if isinstance(ct, str):
        try:
            ct = json.loads(ct)
        except json.JSONDecodeError:
            return {}
    return ct if isinstance(ct, dict) else {}


def prefilter_math(row: dict[str, Any]) -> bool:
    """Math domain: no cheap pre-filter — all rows are passed to the judge."""
    return True


def prefilter_coding(row: dict[str, Any], allowed_languages: list[str] | None) -> bool:
    """Coding domain: optionally filter by language."""
    if allowed_languages:
        lang = str(row.get("language") or "").strip().lower()
        allowed_norm = {str(v).strip().lower() for v in allowed_languages if str(v).strip()}
        if lang not in allowed_norm:
            return False
    return True


def prefilter_generic(
    row: dict[str, Any],
    allowed_languages: list[str],
    min_prompt_len: int,
    min_prompt_words: int,
    prompt: str,
) -> bool:
    """Generic domain: exclude dataset-labelled coding/math, short prompts, and non-English."""
    lang = str(row.get("language") or "").strip().lower()
    allowed_norm = {v.strip().lower() for v in allowed_languages}
    if lang not in allowed_norm:
        return False
    if row.get("is_code"):
        return False
    ct = _category_tag(row)
    math_tag = ct.get("math_v0.1") or {}
    if isinstance(math_tag, dict) and bool(math_tag.get("math")):
        return False
    if len(prompt) < min_prompt_len:
        return False
    if len(prompt.split()) < min_prompt_words:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# LLM judge
# ──────────────────────────────────────────────────────────────────────────────

def build_judge_client(
    api_key_env: str,
    base_url: str | None,
    timeout: int,
) -> OpenAI:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"{api_key_env} is required for the judge client.")
    resolved_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAI(api_key=api_key, base_url=resolved_url, timeout=timeout)


def call_judge(
    client: OpenAI,
    model: str,
    domain: str,
    prompt: str,
) -> dict[str, Any]:
    flag = DOMAIN_FLAG[domain]
    user_msg = JUDGE_USER_TEMPLATE[domain].format(prompt=prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM[domain]},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        lower = raw.lower()
        parsed = {
            flag: (f'"{flag}": true' in lower) or (f'"{flag}":true' in lower),
            "confidence": 0.0,
            "category": "parse_fallback",
            "reason": raw[:300],
        }
    result: dict[str, Any] = {
        flag: bool(parsed.get(flag, False)),
        "confidence": float(parsed.get("confidence") or 0.0),
        "reason": str(parsed.get("reason", "")),
        "judge_raw": raw,
    }
    if domain in ("coding", "generic"):
        result["category"] = str(parsed.get("category", ""))
    return result


def judge_one(
    client: OpenAI,
    judge_model: str,
    domain: str,
    row_id: str,
    prompt: str,
    min_confidence: float,
) -> dict[str, Any]:
    flag = DOMAIN_FLAG[domain]
    if not prompt:
        return {
            "id": row_id, "prompt": "", "status": "skipped_no_prompt",
            flag: False, "confidence": 0.0,
            "category": "", "reason": "", "judge_raw": "",
            "judge_model": judge_model, "selected": False,
        }
    try:
        result = call_judge(client, judge_model, domain, prompt)
        selected = bool(result[flag]) and result["confidence"] >= min_confidence
        return {
            "id": row_id, "prompt": prompt, "status": "ok",
            **result,
            "judge_model": judge_model, "selected": selected,
        }
    except Exception as exc:
        return {
            "id": row_id, "prompt": prompt, "status": "judge_error",
            flag: False, "confidence": 0.0,
            "category": "judge_error", "reason": str(exc), "judge_raw": "",
            "judge_model": judge_model, "selected": False,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Output record builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_id(raw_row: dict[str, Any], row_index: int, domain: str) -> str:
    value = str(raw_row.get("id", "")).strip()
    if value:
        return value
    return f"arena_{domain}_{row_index:06d}"


def build_math_record(
    raw_row: dict[str, Any],
    judged: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    return {
        "id": judged["id"],
        "model_a": raw_row.get("model_a"),
        "model_b": raw_row.get("model_b"),
        "winner": raw_row.get("winner"),
        "language": raw_row.get("language"),
        "is_code": raw_row.get("is_code"),
        "is_math": judged["is_math"],
        "confidence": judged["confidence"],
        "reason": judged["reason"],
        "prompt": prompt,
    }


def build_coding_record(
    raw_row: dict[str, Any],
    judged: dict[str, Any],
    dataset: str,
    split: str,
) -> dict[str, Any]:
    return {
        "id": judged["id"],
        "dataset": dataset,
        "split": split,
        "model_a": raw_row.get("model_a"),
        "model_b": raw_row.get("model_b"),
        "winner": raw_row.get("winner"),
        "evaluation_order": raw_row.get("evaluation_order"),
        "language": raw_row.get("language"),
        "occupational_tags": raw_row.get("occupational_tags"),
        "prompt": judged["prompt"],
        "conversation_a": raw_row.get("conversation_a"),
        "conversation_b": raw_row.get("conversation_b"),
        "judge_model": judged["judge_model"],
        "is_coding": judged["is_coding"],
        "confidence": judged["confidence"],
        "category": judged.get("category", ""),
        "reason": judged["reason"],
        "judge_raw": judged["judge_raw"],
    }


def build_generic_record(
    raw_row: dict[str, Any],
    judged: dict[str, Any],
    prompt: str,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    return {
        "id": str(raw_row.get("id", "")).strip(),
        "dataset": dataset,
        "split": split,
        "model_a": raw_row.get("model_a"),
        "model_b": raw_row.get("model_b"),
        "winner": raw_row.get("winner"),
        "language": raw_row.get("language"),
        "is_code": raw_row.get("is_code"),
        "category_tag": raw_row.get("category_tag"),
        "prompt": prompt,
        "conversation_a": raw_row.get("conversation_a"),
        "conversation_b": raw_row.get("conversation_b"),
        "judge_model": judged["judge_model"],
        "is_generic": judged["is_generic"],
        "confidence": judged["confidence"],
        "category": judged.get("category", ""),
        "reason": judged["reason"],
        "judge_raw": judged["judge_raw"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Domain runners
# ──────────────────────────────────────────────────────────────────────────────

def run_math(cfg: dict[str, Any], client: OpenAI) -> None:
    """Sequential judging for math (mirrors original sample_arena_math.py logic)."""
    dataset_id = cfg.get("dataset", "lmarena-ai/arena-human-preference-140k")
    subset = cfg.get("subset")
    split = cfg.get("split", "train")
    judge_model = cfg.get("judge_model", "gpt-4.1-mini")
    output = Path(cfg.get("output", "data/arena_140k_math.jsonl"))
    save_every = int(cfg.get("save_every", 200))
    min_confidence = float(cfg.get("min_confidence", 0.0))
    resume = bool(cfg.get("resume", False))
    max_rows = cfg.get("max_rows")

    if subset:
        ds = load_dataset(dataset_id, subset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    rows = [dict(r) for r in ds]
    if max_rows is not None:
        rows = rows[:int(max_rows)]

    done_ids: set[str] = set()
    if resume:
        done_ids |= load_done_ids(output)

    kept_buffer: list[dict[str, Any]] = []
    judged = kept = 0

    pbar = tqdm(rows, desc="Judging Arena math", unit="row")
    for row_index, row in enumerate(pbar):
        rid = str(row.get("id", "")).strip() or f"arena_math_{row_index:06d}"
        if rid in done_ids:
            continue

        prompt = extract_user_prompt(row)
        if not prompt:
            done_ids.add(rid)
            judged += 1
            if save_every > 0 and judged % save_every == 0:
                append_jsonl(output, kept_buffer)
                kept_buffer.clear()
            continue

        result = judge_one(client, judge_model, "math", rid, prompt, min_confidence)
        judged += 1

        if result["selected"]:
            kept_buffer.append(build_math_record(row, result, prompt))
            kept += 1
        done_ids.add(rid)

        pbar.set_postfix_str(f"judged={judged} kept={kept}", refresh=False)
        if save_every > 0 and judged % save_every == 0:
            append_jsonl(output, kept_buffer)
            kept_buffer.clear()

    pbar.close()
    append_jsonl(output, kept_buffer)
    print(f"Done. judged={judged} kept={kept}")
    print(f"Output: {output}")


def run_coding(cfg: dict[str, Any], client: OpenAI) -> None:
    """Parallel judging for coding (mirrors sample_arena_coding.py logic)."""
    dataset_id = cfg.get("dataset", "lmarena-ai/arena-expert-5k")
    subset = cfg.get("subset")
    split = cfg.get("split", "train")
    judge_model = cfg.get("judge_model", "gpt-4.1-mini")
    output = Path(cfg.get("output", "data/arena_expert_5k_coding.jsonl"))
    checkpoint_path = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else \
        output.with_name(f"{output.stem}_checkpoint.json")
    save_every = int(cfg.get("save_every", 100))
    min_confidence = float(cfg.get("min_confidence", 0.0))
    max_workers = int(cfg.get("max_workers", 8))
    resume = bool(cfg.get("resume", False))
    max_rows = cfg.get("max_rows")
    max_kept = cfg.get("max_kept")
    seed = cfg.get("seed")
    allowed_languages: list[str] | None = cfg.get("allowed_languages")

    if not resume:
        for p in (output, checkpoint_path):
            if p.exists():
                p.unlink()

    if subset:
        dataset = load_dataset(dataset_id, subset, split=split)
    else:
        dataset = load_dataset(dataset_id, split=split)
    raw_rows = [dict(r) for r in dataset]

    prepared: list[dict[str, Any]] = []
    for idx, raw_row in enumerate(raw_rows):
        rid = _build_id(raw_row, idx, "coding")
        prompt = extract_user_prompt(raw_row)
        prepared.append({
            "id": rid,
            "row_index": idx,
            "prompt": prompt,
            "language": raw_row.get("language"),
            "raw_row": raw_row,
        })

    if seed is not None:
        rng = random.Random(int(seed))
        rng.shuffle(prepared)
    if max_rows is not None:
        prepared = prepared[:int(max_rows)]

    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    done_ids: set[str] = {str(v) for v in checkpoint.get("done_ids", [])}
    kept_existing = load_jsonl(output) if resume else []
    kept_count = len(kept_existing)
    error_count = int(checkpoint.get("judge_errors", 0))
    judged_count = int(checkpoint.get("judged_rows", len(done_ids)))
    status_counts: dict[str, int] = {str(k): int(v) for k, v in (checkpoint.get("status_counts", {}) or {}).items()}

    if max_kept is not None and kept_count >= int(max_kept):
        print(f"Already have {kept_count} kept rows, meets --max-kept={max_kept}.")
        return

    pending = [r for r in prepared if r["id"] not in done_ids]

    kept_buffer: list[dict[str, Any]] = []
    processed_since_flush = 0

    def flush() -> None:
        nonlocal processed_since_flush
        append_jsonl(output, kept_buffer)
        kept_buffer.clear()
        save_checkpoint(checkpoint_path, {
            "done_ids": sorted(done_ids),
            "judged_rows": judged_count,
            "kept_rows": kept_count,
            "judge_errors": error_count,
            "status_counts": status_counts,
        })
        processed_since_flush = 0

    progress = tqdm(
        total=len(prepared), initial=judged_count,
        desc="Judging Arena coding", unit="row",
    )
    stop_submitting = False

    def _should_skip_language(row: dict[str, Any]) -> bool:
        if not allowed_languages:
            return False
        lang = str(row.get("language") or "").strip().lower()
        return lang not in {str(v).strip().lower() for v in allowed_languages}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending_iter = iter(pending)
        futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

        def maybe_submit() -> bool:
            nonlocal stop_submitting
            if stop_submitting:
                return False
            if max_kept is not None and kept_count >= int(max_kept):
                stop_submitting = True
                return False
            try:
                row = next(pending_iter)
            except StopIteration:
                return False
            if _should_skip_language(row):
                return True
            fut = executor.submit(
                judge_one, client, judge_model, "coding",
                row["id"], row["prompt"], float(min_confidence),
            )
            futures[fut] = row
            return True

        pending_list = list(pending)
        pending_iter = iter(pending_list)
        futures.clear()
        for _ in range(min(max_workers, len(pending_list))):
            if not maybe_submit():
                break

        while futures:
            done_futs, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done_futs:
                row = futures.pop(fut)
                judged = fut.result()
                processed_since_flush += 1
                status = str(judged["status"])
                status_counts[status] = status_counts.get(status, 0) + 1

                if status in _DONE_STATUSES:
                    done_ids.add(row["id"])
                    judged_count += 1
                    progress.update(1)
                elif status == "judge_error":
                    error_count += 1

                if judged["selected"]:
                    if max_kept is None or kept_count < int(max_kept):
                        kept_buffer.append(build_coding_record(row["raw_row"], judged, dataset_id, split))
                        kept_count += 1
                    else:
                        stop_submitting = True

                progress.set_postfix_str(
                    f"judged={judged_count} kept={kept_count} errors={error_count}",
                    refresh=False,
                )
                if processed_since_flush >= save_every:
                    flush()
                maybe_submit()

    progress.close()
    flush()

    summary = {
        "dataset": dataset_id, "subset": subset, "split": split,
        "allowed_languages": allowed_languages,
        "judge_model": judge_model, "min_confidence": min_confidence,
        "max_workers": max_workers, "seed": seed,
        "max_rows": max_rows, "max_kept": max_kept,
        "judged_rows": judged_count, "kept_rows": kept_count,
        "judge_errors": error_count, "status_counts": status_counts,
        "output": str(output), "checkpoint": str(checkpoint_path),
    }
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. judged={judged_count} kept={kept_count} errors={error_count}")
    print(f"Output: {output}")
    print(f"Summary: {summary_path}")


def run_generic(cfg: dict[str, Any], client: OpenAI) -> None:
    """Two-phase (pre-filter + parallel judge) sampling for generic domain."""
    dataset_id = cfg.get("dataset", "lmarena-ai/arena-human-preference-140k")
    split = cfg.get("split", "train")
    judge_model = cfg.get("judge_model", "gpt-4.1-mini")
    output = Path(cfg.get("output", "data/arena_140k_generic_judged.jsonl"))
    judged_path = output.with_name(f"{output.stem}_judged.jsonl")
    ckpt_path = output.with_name(f"{output.stem}_checkpoint.json")
    max_kept = int(cfg.get("max_kept", 1000))
    allowed_languages: list[str] = cfg.get("allowed_languages", ["en"])
    min_prompt_len = int(cfg.get("min_prompt_len", 150))
    min_prompt_words = int(cfg.get("min_prompt_words", 20))
    min_confidence = float(cfg.get("min_confidence", 0.85))
    max_workers = int(cfg.get("max_workers", 8))
    save_every = int(cfg.get("save_every", 100))
    seed = int(cfg.get("seed", 42))
    resume = bool(cfg.get("resume", False))
    max_candidates = cfg.get("max_candidates")

    if not resume:
        for p in (output, judged_path, ckpt_path):
            if p.exists():
                p.unlink()

    print(f"Loading {dataset_id} ({split})...")
    ds = load_dataset(dataset_id, split=split)
    raw_rows = [dict(r) for r in ds]
    print(f"  Loaded {len(raw_rows):,} rows.")

    # Pre-filter
    candidates_all: list[dict[str, Any]] = []
    skipped_coding = skipped_math = skipped_prompt = skipped_lang = 0
    allowed_norm = {v.strip().lower() for v in allowed_languages}
    for row in raw_rows:
        prompt = extract_user_prompt(row)
        if not prefilter_generic(row, allowed_languages, min_prompt_len, min_prompt_words, prompt):
            lang = str(row.get("language") or "").strip().lower()
            if lang not in allowed_norm:
                skipped_lang += 1
            elif row.get("is_code"):
                skipped_coding += 1
            else:
                ct = _category_tag(row)
                math_tag = ct.get("math_v0.1") or {}
                if isinstance(math_tag, dict) and bool(math_tag.get("math")):
                    skipped_math += 1
                else:
                    skipped_prompt += 1
            continue
        candidates_all.append({
            "id": str(row.get("id", "")).strip(),
            "prompt": prompt,
            "raw_row": row,
        })

    print(
        f"  After pre-filter: {len(candidates_all):,} candidates "
        f"(skipped lang={skipped_lang} coding={skipped_coding} "
        f"math={skipped_math} short_prompt={skipped_prompt})."
    )

    rng = random.Random(seed)
    rng.shuffle(candidates_all)
    if max_candidates is not None:
        candidates_all = candidates_all[:int(max_candidates)]
        print(f"  Capped to {len(candidates_all):,} via max_candidates.")

    ckpt = load_checkpoint(ckpt_path) if resume else {}
    done_ids: set[str] = {str(v) for v in ckpt.get("done_ids", [])}
    judged_count = int(ckpt.get("judged_count", len(done_ids)))
    accepted_count = int(ckpt.get("accepted_count", 0))
    error_count = int(ckpt.get("error_count", 0))

    pending = [c for c in candidates_all if c["id"] not in done_ids]
    print(f"  Pending: {len(pending):,} (already done: {len(done_ids):,}).")

    write_buffer: list[dict[str, Any]] = []
    processed_since_flush = 0

    def flush() -> None:
        nonlocal processed_since_flush
        append_jsonl(judged_path, write_buffer)
        write_buffer.clear()
        save_checkpoint(ckpt_path, {
            "done_ids": sorted(done_ids),
            "judged_count": judged_count,
            "accepted_count": accepted_count,
            "error_count": error_count,
        })
        processed_since_flush = 0

    pbar = tqdm(
        total=len(candidates_all), initial=judged_count,
        desc="Judging generic candidates", unit="row",
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                cand = next(pending_iter)
            except StopIteration:
                return False
            fut = executor.submit(
                judge_one, client, judge_model, "generic",
                cand["id"], cand["prompt"], min_confidence,
            )
            future_map[fut] = cand
            return True

        for _ in range(min(max_workers, len(pending))):
            submit_next()

        while future_map:
            done_futs, _ = concurrent.futures.wait(
                future_map, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done_futs:
                cand = future_map.pop(fut)
                result = fut.result()
                status = result["status"]
                processed_since_flush += 1

                if status in _DONE_STATUSES:
                    done_ids.add(cand["id"])
                    judged_count += 1
                    pbar.update(1)
                elif status == "judge_error":
                    error_count += 1

                if result.get("selected"):
                    rec = build_generic_record(
                        cand["raw_row"], result, cand["prompt"], dataset_id, split,
                    )
                    write_buffer.append(rec)
                    accepted_count += 1

                pbar.set_postfix_str(
                    f"judged={judged_count} accepted={accepted_count} errors={error_count}",
                    refresh=False,
                )
                if processed_since_flush >= save_every:
                    flush()
                submit_next()

    pbar.close()
    flush()

    # Post-filter: drop trivial categories, keep high-confidence, sample up to max_kept
    all_accepted = load_jsonl(judged_path)
    _TRIVIAL_CATEGORIES = {
        "casual conversation", "conversational", "conversational greeting",
        "greeting", "casual greeting", "casual question", "general inquiry",
        "casual response", "casual opinion",
    }
    before = len(all_accepted)
    all_accepted = [
        r for r in all_accepted
        if r.get("category", "").strip().lower() not in _TRIVIAL_CATEGORIES
    ]
    if before != len(all_accepted):
        print(f"  Removed {before - len(all_accepted):,} rows with trivial categories.")

    high_conf = [r for r in all_accepted if r.get("confidence", 0.0) >= 0.9]
    print(f"  Rows with confidence >= 0.9: {len(high_conf):,}")

    if len(high_conf) <= max_kept:
        selected = high_conf
        if len(selected) < max_kept:
            shortfall = max_kept - len(selected)
            print(
                f"WARNING [generic]: requested {max_kept}, available {len(selected)} "
                f"— {shortfall} short. Lower min_confidence or increase max_candidates.",
                file=sys.stderr,
            )
    else:
        rng.shuffle(high_conf)
        selected = high_conf[:max_kept]
        print(f"  Randomly sampled {len(selected):,} from {len(high_conf):,} high-confidence rows.")

    write_jsonl(output, selected)

    summary = {
        "dataset": dataset_id, "split": split,
        "allowed_languages": sorted(allowed_norm),
        "judge_model": judge_model, "min_confidence": min_confidence, "seed": seed,
        "domain_label_pre_filter": {
            "excluded_lang": skipped_lang,
            "excluded_coding": skipped_coding,
            "excluded_math": skipped_math,
            "excluded_short_prompt": skipped_prompt,
            "candidates_after_filter": len(candidates_all),
        },
        "judged": judged_count, "accepted": len(all_accepted),
        "selected": len(selected), "target": max_kept,
        "judge_errors": error_count,
        "output": str(output), "judged_log": str(judged_path), "checkpoint": str(ckpt_path),
    }
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. judged={judged_count} accepted={len(all_accepted)} "
          f"selected={len(selected)} errors={error_count}")
    print(f"Output: {output}")
    print(f"Summary: {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Arena question sampler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--domain",
        choices=["math", "coding", "generic"],
        required=True,
        help="Domain to sample. misc has no sampling script (uses pre-existing files).",
    )
    p.add_argument(
        "--config",
        default="demo/config_arena.yaml",
        help="Path to unified config_arena.yaml.",
    )
    p.add_argument(
        "--judge-api-key-env",
        default=None,
        help="Env var holding the judge API key (overrides config).",
    )
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="Base URL for judge endpoint (overrides config).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(resolve_env_path())

    full_cfg = load_yaml_config(args.config)
    domain_cfg = full_cfg.get(args.domain, {})
    sampling_cfg: dict[str, Any] = domain_cfg.get("sampling", {})

    api_key_env = args.judge_api_key_env or sampling_cfg.get("judge_api_key_env", "OPENAI_API_KEY")
    base_url = args.judge_base_url or sampling_cfg.get("judge_base_url")
    timeout = int(sampling_cfg.get("timeout", 60))

    client = build_judge_client(api_key_env, base_url, timeout)

    if args.domain == "math":
        run_math(sampling_cfg, client)
    elif args.domain == "coding":
        run_coding(sampling_cfg, client)
    elif args.domain == "generic":
        run_generic(sampling_cfg, client)


if __name__ == "__main__":
    main()
