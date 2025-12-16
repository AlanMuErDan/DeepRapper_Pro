from pypinyin import pinyin, Style
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_rhyme_unit(line, n=1):
    """Return the finals of the last n characters in a lyric line."""
    line = line.strip()
    if not line:
        return None

    tail = list(line[-n:])
    py = pinyin(tail, style=Style.FINALS, strict=False)
    rhyme = "-".join(p[0] for p in py if p and p[0])
    return rhyme if rhyme else None


def get_rhyme_sequence(text, n=1):
    text = text.replace("\\n", "\n")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rhyme_seq = [extract_rhyme_unit(line, n) for line in lines]
    return rhyme_seq, lines


def compute_rhyme_accuracy(text, n=1):
    rhyme_seq, _ = get_rhyme_sequence(text, n)
    if len(rhyme_seq) < 2:
        return 0.0
    rhymed_pairs = sum(
        1 for i in range(1, len(rhyme_seq))
        if rhyme_seq[i] == rhyme_seq[i - 1] and rhyme_seq[i]
    )
    return round(rhymed_pairs / (len(rhyme_seq) - 1), 4)


def compute_rhyme_density(text, n=1):
    rhyme_seq, _ = get_rhyme_sequence(text, n)
    if not rhyme_seq:
        return 0.0
    rhymed_lines = set()
    for i in range(1, len(rhyme_seq)):
        if rhyme_seq[i] == rhyme_seq[i - 1] and rhyme_seq[i]:
            rhymed_lines.add(i)
            rhymed_lines.add(i - 1)
    return round(len(rhymed_lines) / len(rhyme_seq), 4)


def compute_combo_n(text, n=1):
    rhyme_seq, _ = get_rhyme_sequence(text, n)
    if not rhyme_seq:
        return 0
    max_run, run = 1, 1
    for i in range(1, len(rhyme_seq)):
        if rhyme_seq[i] == rhyme_seq[i - 1] and rhyme_seq[i]:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    return max(max_run, run)


def compute_rhyme_diversity(text, n=1):
    rhyme_seq, _ = get_rhyme_sequence(text, n)
    if not rhyme_seq:
        return 0.0
    unique = set(r for r in rhyme_seq if r)
    return round(len(unique) / len(rhyme_seq), 4)


def evaluate_rhyme_csv(csv_path, text_columns, n=1):
    """Batch-compute rhyme metrics for the specified CSV columns."""

    df = pd.read_csv(csv_path)
    results = {}

    for col in tqdm(text_columns, desc="Evaluating rhyme columns"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

        raw_values = df[col].fillna("").tolist()
        col_values = [x for x in raw_values if x.strip() != ""]

        if len(col_values) == 0:
            results[col] = {
                "Rhyme_Accuracy": np.nan,
                "Rhyme_Density": np.nan,
                "Combo_N": np.nan,
                "Rhyme_Diversity": np.nan,
            }
            continue

        acc_scores = []
        dens_scores = []
        combo_scores = []
        div_scores = []

        for text in tqdm(col_values, desc=f"Processing {col}", leave=False):
            acc_scores.append(compute_rhyme_accuracy(text, n))
            dens_scores.append(compute_rhyme_density(text, n))
            combo_scores.append(compute_combo_n(text, n))
            div_scores.append(compute_rhyme_diversity(text, n))

        results[col] = {
            "Rhyme_Accuracy": round(np.nanmean(acc_scores), 4),
            "Rhyme_Density": round(np.nanmean(dens_scores), 4),
            "Combo_N": round(np.nanmean(combo_scores), 4),
            "Rhyme_Diversity": round(np.nanmean(div_scores), 4),
        }

    return pd.DataFrame(results)


if __name__ == "__main__":
    csv_path = "/Users/liyuanheng/Desktop/NLP/ProjectCode/deeprapper_pro/data/cheating.csv"

    cols = [
        "gpt-4.1-chrhyme-S-5-LLM",
        "gpt-4.1-chrhyme-S-10-LLM",
        "gpt-4.1-chrhyme-S-15-LLM",
        "gpt-4.1-chrhyme-S-20-LLM"
    ]

    table = evaluate_rhyme_csv(csv_path, cols, n=2)
    print(table)
