"""Utility functions for rhyme and semantic metrics."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pypinyin import pinyin, Style

from bert_score import score as bertscore

from sentence_transformers import SentenceTransformer, util


_TOKEN_REGEX = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]")
_TOPIC_MODEL: Optional[SentenceTransformer] = None


def _get_topic_model() -> Optional[SentenceTransformer]:
    global _TOPIC_MODEL
    if _TOPIC_MODEL is None and SentenceTransformer is not None:
        _TOPIC_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _TOPIC_MODEL


def extract_rhyme_unit(line: str, n: int = 1) -> Optional[str]:
    line = (line or "").strip()
    if not line:
        return None
    tail = list(line[-n:])
    py = pinyin(tail, style=Style.FINALS, strict=False)
    rhyme = "-".join(p[0] for p in py if p and p[0])
    return rhyme if rhyme else None


def _get_rhyme_sequence(text: str, n: int = 1) -> Tuple[List[Optional[str]], List[str]]:
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    rhyme_seq = [extract_rhyme_unit(line, n) for line in lines]
    return rhyme_seq, lines


def _tokenize_text(text: str) -> List[str]:
    """
    Rough tokenizer that keeps English words, digits, single Chinese chars,
    and any remaining non-space symbols.
    """
    if not text:
        return []
    return _TOKEN_REGEX.findall(text.strip())


def compute_distinct_n(text: str, n: int = 1) -> float:
    tokens = _tokenize_text(text)
    if len(tokens) < n or n <= 0:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    score = len(set(ngrams)) / len(ngrams)
    return round(score, 4)


def compute_rhyme_accuracy(text: str, n: int = 1) -> float:
    rhyme_seq, _ = _get_rhyme_sequence(text, n)
    if len(rhyme_seq) < 2:
        return 0.0
    rhymed_pairs = sum(
        1 for i in range(1, len(rhyme_seq)) if rhyme_seq[i] and rhyme_seq[i] == rhyme_seq[i - 1]
    )
    return round(rhymed_pairs / (len(rhyme_seq) - 1), 4)


def compute_rhyme_density(text: str, n: int = 1) -> float:
    rhyme_seq, _ = _get_rhyme_sequence(text, n)
    if not rhyme_seq:
        return 0.0
    rhymed_lines = set()
    for i in range(1, len(rhyme_seq)):
        if rhyme_seq[i] and rhyme_seq[i] == rhyme_seq[i - 1]:
            rhymed_lines.add(i)
            rhymed_lines.add(i - 1)
    return round(len(rhymed_lines) / len(rhyme_seq), 4)


def compute_combo_n(text: str, n: int = 1) -> int:
    rhyme_seq, _ = _get_rhyme_sequence(text, n)
    if not rhyme_seq:
        return 0
    max_run = run = 1
    for i in range(1, len(rhyme_seq)):
        if rhyme_seq[i] and rhyme_seq[i] == rhyme_seq[i - 1]:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    return max(max_run, run)


def compute_semantic_bertscore(reference: str, generated: str, model_type: str = "xlm-roberta-large") -> Optional[float]:
    reference = (reference or "").strip()
    generated = (generated or "").strip()
    if not reference or not generated or bertscore is None:
        return None
    P, R, F1 = bertscore(
        [generated],
        [reference],
        lang="zh",
        idf=False,
        rescale_with_baseline=False,
        model_type=model_type,
        nthreads=1,
    )
    return float(F1[0])


def compute_topic_similarity(reference: str, generated: str) -> Optional[float]:
    reference = (reference or "").strip()
    generated = (generated or "").strip()
    if not reference or not generated or util is None:
        return None
    model = _get_topic_model()
    if not model:
        return None
    emb_ref = model.encode(reference, convert_to_tensor=True)
    emb_gen = model.encode(generated, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_gen)
    return float(sim.item())


__all__ = [
    "compute_distinct_n",
    "compute_rhyme_accuracy",
    "compute_rhyme_density",
    "compute_combo_n",
    "compute_semantic_bertscore",
    "compute_topic_similarity",
]
