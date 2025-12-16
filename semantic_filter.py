"""Semantic filtering for rhyme candidates.

This module optionally ranks chrhyme outputs against the user provided
theme/content hints and keeps the top-k most relevant words.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from difflib import SequenceMatcher

from llm_utils import generate_with_llm
from log_utils import print_step

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    SentenceTransformer = None  # type: ignore


_EMBEDDER = None
_EMBEDDER_FAILED = False


def _get_embedder():
    """Lazy-loads the sentence-transformer used for semantic ranking."""

    global _EMBEDDER, _EMBEDDER_FAILED

    if _EMBEDDER_FAILED:
        return None

    if _EMBEDDER is not None:
        return _EMBEDDER

    if SentenceTransformer is None:
        _EMBEDDER_FAILED = True
        return None

    model_name = os.getenv(
        "SEMANTIC_FILTER_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    try:
        _EMBEDDER = SentenceTransformer(model_name)
        return _EMBEDDER
    except Exception as exc:  # pragma: no cover - model download/initialise errors
        print(f"[WARN] semantic filter model load failed: {exc}")
        _EMBEDDER_FAILED = True
        _EMBEDDER = None
        return None


def _flatten_candidates(rhyme_dict: Dict[str, Sequence[str]]) -> List[str]:
    seen = set()
    flattened: List[str] = []
    for words in rhyme_dict.values():
        for item in words or []:
            term = (item or "").strip()
            if not term or term in seen:
                continue
            seen.add(term)
            flattened.append(term)
    return flattened


def _embed_similarity(reference: str, candidates: Sequence[str]) -> List[Tuple[str, float]]:
    embedder = _get_embedder()
    if not embedder:
        return []

    try:
        encoded = embedder.encode(
            [reference, *candidates],
            normalize_embeddings=True,
        )
        query_vec = encoded[0]
        candidate_matrix = encoded[1:]
        if not len(candidate_matrix):
            return []

        similarities = list(candidate_matrix @ query_vec)
        return list(zip(candidates, similarities))
    except Exception as exc:  # pragma: no cover - runtime embed errors
        print(f"[WARN] semantic filter inference failed: {exc}")
        return []


def _overlap_similarity(reference: str, candidates: Sequence[str]) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for word in candidates:
        ratio = SequenceMatcher(None, reference, word).ratio()
        char_overlap = 0.0
        if reference and word:
            ref_set = set(reference)
            word_set = set(word)
            denom = len(ref_set | word_set)
            if denom:
                char_overlap = len(ref_set & word_set) / denom
        score = 0.7 * ratio + 0.3 * char_overlap
        scored.append((word, score))
    return scored


def _rank_candidates(reference: str, candidates: Sequence[str]) -> List[Tuple[str, float]]:
    if not reference or not candidates:
        return []

    scored = _embed_similarity(reference, candidates)
    if not scored:
        scored = _overlap_similarity(reference, candidates)

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _format_rhyme_dict_for_prompt(rhyme_dict: Dict[str, Iterable[str]]) -> str:
    lines: List[str] = []
    for label, words in rhyme_dict.items():
        cleaned = [w for w in (words or []) if w]
        if not cleaned:
            continue
        joined = ", ".join(cleaned)
        lines.append(f"{label}: {joined}")
    return "\n".join(lines)


def _parse_llm_filter_response(
    text: str,
    candidates: Sequence[str],
) -> List[str]:
    candidate_set = {item for item in candidates}
    ordered: List[str] = []

    def _append(word: str):
        norm = word.strip()
        if not norm or norm not in candidate_set or norm in ordered:
            return
        ordered.append(norm)

    if not text:
        return ordered

    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            payload = data.get("filtered_words")
        else:
            payload = data
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    _append(item)
                else:
                    _append(str(item))
            return ordered
    except Exception:
        pass

    for line in stripped.splitlines():
        cleaned = line.strip().lstrip("-•")
        if not cleaned:
            continue
        for part in cleaned.split(","):
            _append(part)
    return ordered


def _llm_filter_candidates(
    theme: str,
    content: str,
    reference: str,
    rhyme_dict: Dict[str, Iterable[str]],
    candidates: Sequence[str],
    top_k: int,
    provider: str,
    model: Optional[str],
) -> Tuple[List[str], str]:
    if not candidates or not reference:
        return [], ""

    formatted_rhymes = _format_rhyme_dict_for_prompt(rhyme_dict)
    candidate_lines = "\n".join(f"- {word}" for word in candidates)
    k = min(top_k, len(candidates))

    prompt = (
        "You filter Chinese rhyme candidates for a rap lyric generator.\n"
        "Given the theme, content hint, and chrhyme outputs, return the most relevant words.\n"
        "Only choose from the provided candidates and keep cultural nuance.\n"
        f"Respond with valid JSON only: {{\"filtered_words\": [\"word1\", ...]}}. The array MUST contain exactly {k} items unless fewer candidates exist.\n"
        "If fewer candidates exist, return all of them. Never return more than {k}.\n"
        "Do not add commentary or explanations.\n"
        f"\nTheme: {theme or '（无）'}"
        f"\nContent hint: {content or '（无）'}"
        f"\nCombined reference: {reference}"
        "\n\nchrhyme groups:\n"
        f"{formatted_rhymes or '无'}"
        "\n\nCandidate words:\n"
        f"{candidate_lines}"
        f"\n\nReturn exactly the JSON object and ensure the list has exactly {k} items unless the candidate list is shorter."
    )

    response = generate_with_llm(
        prompt,
        provider=provider,
        model=model,
        temperature=0.2,
        max_tokens=512,
    )
    if not response or response.startswith("❌"):
        return [], prompt

    parsed = _parse_llm_filter_response(response, candidates)
    return parsed[:k], prompt


def apply_semantic_filter(
    rhyme_dict: Dict[str, Iterable[str]] | None,
    theme: str,
    content: str,
    top_k: int = 30,
    enabled: bool = False,
    filter_mode: str = "bertscore",
    llm_provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], bool]:
    """Filters rhyme words with semantic similarity.

    Returns (new_rhyme_dict, applied_flag).
    If filtering cannot run the original dictionary is returned with applied=False.
    """

    rhyme_dict = rhyme_dict or {}
    mode = (filter_mode or "").strip().lower()
    if not enabled or mode in {"", "off", "none"}:
        return rhyme_dict, False

    reference = " ".join([part for part in [theme, content] if part]).strip()
    if not reference:
        return rhyme_dict, False

    candidates = _flatten_candidates(rhyme_dict)
    if not candidates:
        return rhyme_dict, False

    try:
        k = int(top_k)
    except (TypeError, ValueError):
        k = 0
    k = max(1, k)

    filtered_words: List[str] = []
    llm_prompt: Optional[str] = None
    if mode == "llm":
        filtered_words, llm_prompt = _llm_filter_candidates(
            theme=theme,
            content=content,
            reference=reference,
            rhyme_dict=rhyme_dict,
            candidates=candidates,
            top_k=k,
            provider=llm_provider,
            model=llm_model,
        )
        before_fallback = len(filtered_words)
        if before_fallback < k:
            ranked = _rank_candidates(reference, candidates)
            supplement: List[str] = []
            for word, _ in ranked:
                if word in filtered_words:
                    continue
                filtered_words.append(word)
                supplement.append(word)
                if len(filtered_words) >= k:
                    break
            if supplement:
                detail = (
                    f"LLM 仅返回 {before_fallback} 个候选，"
                    f"通过 BERTScore 排序补充 {len(supplement)} 个：{', '.join(supplement)}"
                )
                print_step("Step 2 - Semantic Filtering Fallback", detail)
            elif before_fallback < k:
                print_step(
                    "Step 2 - Semantic Filtering Fallback",
                    "LLM 返回的候选不足且无可用 BERTScore 备选，可能是候选列表过短。",
                )
    else:
        ranked = _rank_candidates(reference, candidates)
        filtered_words = [word for word, _ in ranked[:k]]

    if mode == "llm":
        print_step(
            "Step 2 - Semantic Filtering Prompt",
            llm_prompt or "LLM 语义筛选 prompt 构建失败。",
        )
    else:
        print_step(
            "Step 2 - Semantic Filtering Prompt",
            f"BERTScore 模式参考文本：{reference}",
        )

    if not filtered_words:
        print_step("Step 2 - Semantic Filtering Result", "（语义筛选失败，返回空列表）")
        return rhyme_dict, False

    label = "BERTScore语义筛选" if mode != "llm" else "LLM语义筛选"
    joined = ", ".join(filtered_words)
    print(f"[Semantic Filter] {label} Top{k}: {joined}")
    print_step("Step 2 - Semantic Filtering Result", "、".join(filtered_words))
    filtered = {f"{label}Top{k}": filtered_words}
    return filtered, True
