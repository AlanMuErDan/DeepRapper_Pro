import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
from pypinyin import pinyin, Style
import numpy as np
from sentence_transformers import SentenceTransformer, util
import sacrebleu
import re


def extract_rhyme_unit(line, n=1):
    """Return the finals of the last n characters in a lyric line."""
    line = line.strip()
    if not line:
        return None
    
    tail = list(line[-n:])
    py = pinyin(tail, style=Style.FINALS, strict=False)
    rhyme = "-".join(p[0] for p in py if p and p[0])
    return rhyme if rhyme else None


def get_rhyme_seq(text, n=1):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rhymes = [extract_rhyme_unit(l, n) for l in lines]
    return rhymes, lines


def compute_bleu(reference, generated):
    """Compute BLEU via sacrebleu for reference vs. generated lyrics."""
    ref = reference.replace("\n", " ")
    gen = generated.replace("\n", " ")
    bleu = sacrebleu.sentence_bleu(gen, [ref])
    return bleu.score / 100.0


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving line breaks."""
    if not isinstance(text, str):
        text = str(text)
    placeholder = " <LB> "
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", placeholder)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(placeholder, "\n")
    return text

def compute_bertscore(reference, generated, model_type="xlm-roberta-large"):
    reference = _normalize_text(reference)
    generated = _normalize_text(generated)

    P, R, F1 = bertscore(
        [generated], [reference],
        model_type=model_type,
        lang="zh",
        idf=True,
        rescale_with_baseline=True,
    )
    return float(F1[0])



style_model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")

def compute_style_similarity(reference, generated):
    emb_ref = style_model.encode(reference, convert_to_tensor=True)
    emb_gen = style_model.encode(generated, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_gen)
    return float(sim.item())


def compute_rhyme_coverage(reference, generated, n=1):
    ref_rhymes, _ = get_rhyme_seq(reference, n=n)
    gen_rhymes, _ = get_rhyme_seq(generated, n=n)

    ref_set = set(r for r in ref_rhymes if r)
    gen_set = set(r for r in gen_rhymes if r)

    if not ref_set:
        return 0.0

    overlap = ref_set.intersection(gen_set)
    score = len(overlap) / len(ref_set)
    return round(score, 4)


topic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compute_topic_similarity(reference, generated):
    emb_ref = topic_model.encode(reference, convert_to_tensor=True)
    emb_gen = topic_model.encode(generated, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_gen)
    return float(sim.item())


if __name__ == "__main__":
    reference = "天上的星星亮晶晶\n我在屋顶想着你"
    generated = "夜里的风轻轻吹\n我独自回忆着你"

    print("BLEU:", compute_bleu(reference, generated))
    print("BERTScore:", compute_bertscore(reference, generated))
    print("Style Similarity:", compute_style_similarity(reference, generated))
    print("Rhyme Coverage (n=1):", compute_rhyme_coverage(reference, generated, n=1))
    print("Topic Similarity:", compute_topic_similarity(reference, generated))
