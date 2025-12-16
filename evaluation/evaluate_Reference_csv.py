import sacrebleu
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util
from pypinyin import pinyin, Style
import numpy as np
import pandas as pd
from tqdm import tqdm

style_model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
topic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def extract_rhyme_unit(line, n=1):
    line = line.strip()
    if not line:
        return None

    tail = list(line[-n:])
    py = pinyin(tail, style=Style.FINALS, strict=False)
    rhyme = "-".join(p[0] for p in py if p and p[0])
    return rhyme if rhyme else None


def get_rhyme_seq(text, n=1):
    text = text.replace("\\n", "\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rhymes = [extract_rhyme_unit(l, n) for l in lines]
    return rhymes, lines


def compute_bleu(ref, gen):
    ref = ref.replace("\n", " ")
    gen = gen.replace("\n", " ")
    bleu = sacrebleu.sentence_bleu(gen, [ref])
    return bleu.score / 100.0


def compute_bertscore(ref, gen, model_type="bert-base-multilingual-cased"):
    P, R, F1 = bertscore([gen], [ref], lang=None, model_type=model_type)
    return float(F1[0])


def compute_style_similarity(ref, gen):
    emb_ref = style_model.encode(ref, convert_to_tensor=True)
    emb_gen = style_model.encode(gen, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_gen)
    return float(sim.item())


def compute_rhyme_coverage(ref, gen, n=1):
    ref_rhymes, _ = get_rhyme_seq(ref, n)
    gen_rhymes, _ = get_rhyme_seq(gen, n)

    ref_set = set(r for r in ref_rhymes if r)
    gen_set = set(r for r in gen_rhymes if r)

    if not ref_set:
        return 0.0
    overlap = ref_set.intersection(gen_set)
    return len(overlap) / len(ref_set)


def compute_topic_similarity(ref, gen):
    emb_ref = topic_model.encode(ref, convert_to_tensor=True)
    emb_gen = topic_model.encode(gen, convert_to_tensor=True)
    sim = util.cos_sim(emb_ref, emb_gen)
    return float(sim.item())


def evaluate_reference_csv(csv_path, ref_column, model_columns, rhyme_n=1):
    """
    reference_column: ground truth column
    model_columns: list of model output columns
    rhyme_n: rhyme N-gram
    """

    df = pd.read_csv(csv_path)
    results = {}

    if ref_column not in df.columns:
        raise ValueError(f"Reference column '{ref_column}' not found.")

    ref_values = df[ref_column].fillna("").tolist()

    for col in tqdm(model_columns, desc="Evaluating reference-based metrics"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

        raw_gen = df[col].fillna("").tolist()

        pairs = [(r, g) for r, g in zip(ref_values, raw_gen)
                 if r.strip() != "" and g.strip() != ""]

        if len(pairs) == 0:
            results[col] = {
                "BLEU": np.nan,
                "BERTScore": np.nan,
                "Style_Sim": np.nan,
                "Rhyme_Coverage": np.nan,
                "Topic_Sim": np.nan,
            }
            continue

        bleu_scores = []
        bert_scores = []
        style_scores = []
        rhyme_scores = []
        topic_scores = []

        for ref, gen in tqdm(pairs, desc=f"Processing {col}", leave=False):
            bleu_scores.append(compute_bleu(ref, gen))
            bert_scores.append(compute_bertscore(ref, gen))
            style_scores.append(compute_style_similarity(ref, gen))
            rhyme_scores.append(compute_rhyme_coverage(ref, gen, rhyme_n))
            topic_scores.append(compute_topic_similarity(ref, gen))

        results[col] = {
            "BLEU": round(np.nanmean(bleu_scores), 4),
            "BERTScore": round(np.nanmean(bert_scores), 4),
            "Style_Sim": round(np.nanmean(style_scores), 4),
            "Rhyme_Coverage": round(np.nanmean(rhyme_scores), 4),
            "Topic_Sim": round(np.nanmean(topic_scores), 4),
        }

    return pd.DataFrame(results)


if __name__ == "__main__":
    csv_path = "/Users/liyuanheng/Desktop/NLP/ProjectCode/deeprapper_pro/data/rap_lyrics_ablation.csv"

    ref_col = "Reference lyrics"
    model_cols = [
        "gpt-4.1-base",
        "gpt-4.1-chrhyme-S-5",
        "gpt-4.1-chrhyme-S-10",
        "gpt-4.1-chrhyme-S-15",
        "gpt-4.1-chrhyme-S-20",
        "gpt-4.1-chrhyme",
    ]

    table = evaluate_reference_csv(csv_path, ref_col, model_cols, rhyme_n=2)
    print(table)
