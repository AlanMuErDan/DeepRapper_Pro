import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import jieba
import numpy as np
import pandas as pd

_GPT2_TOKENIZER = None
_GPT2_MODEL = None

def load_gpt2(model_name="gpt2"):
    """Load GPT-2 only once (massive speed improvement)."""
    global _GPT2_TOKENIZER, _GPT2_MODEL
    if _GPT2_TOKENIZER is None:
        _GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained(model_name)
    if _GPT2_MODEL is None:
        _GPT2_MODEL = GPT2LMHeadModel.from_pretrained(model_name)
        _GPT2_MODEL.eval()
    return _GPT2_TOKENIZER, _GPT2_MODEL


def compute_perplexity(text, model_name="gpt2"):
    text = text.replace("\\n", "\n")

    if text.strip() == "":
        return float("nan")

    tokenizer, model = load_gpt2(model_name)

    encodings = tokenizer(text, return_tensors="pt", truncation=False)

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    if seq_len == 0:
        return float("nan")

    lls = []

    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)

        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        mask_len = end_loc - i
        total_len = input_ids.size(1)

        target_ids = input_ids.clone()

        if mask_len < total_len:
            target_ids[:, : total_len - mask_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            if torch.isnan(outputs.loss):
                continue
            lls.append(outputs.loss * mask_len)

    if len(lls) == 0:
        return float("nan")

    total_loss = torch.stack(lls).sum()
    ppl = torch.exp(total_loss / seq_len)
    return float(ppl)




def tokenize_text(text):
    """Tokenize text with jieba while preserving newlines for line metrics."""
    tokens = []
    temp = text.strip().split("\n")
    for line in temp:
        for word in jieba.cut(line):
            sub_tokens = word.strip().split()
            tokens.extend([w for w in sub_tokens if w])
    return tokens



def compute_distinct_n(text, n=1):
    tokens = tokenize_text(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return round(len(set(ngrams)) / len(ngrams), 4)


def compute_length_variance(text):
    """Return (avg_line_length, variance)."""
    text = text.replace("\\n", "\n")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0, 0.0
    lengths = [len(tokenize_text(line)) for line in lines]
    return round(np.mean(lengths), 2), round(np.var(lengths), 2)


def compute_entropy(text):
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0

    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    total = len(tokens)
    probs = [c / total for c in freq.values()]

    entropy = -sum(p * math.log(p + 1e-10, 2) for p in probs)
    return round(entropy, 4)


def compute_lexical_density(text):
    import jieba.posseg as pseg

    tokens = list(pseg.cut(text))
    if not tokens:
        return 0.0

    function_tags = set(["u", "p", "r", "c", "d", "x", "w", "t", "m"])
    content_count = sum(1 for _, f in tokens if f and f[0] not in function_tags)

    return round(content_count / len(tokens), 4)


def evaluate_csv(csv_path, text_columns):
    df = pd.read_csv(csv_path)
    results = {}

    for col in tqdm(text_columns, desc="Evaluating columns"):

        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

        raw_values = df[col].fillna("").tolist()
        col_values = [x for x in raw_values if x.strip() != ""]
        
        if len(col_values) == 0:
            results[col] = {
                "PPL": np.nan,
                "Distinct-1": np.nan,
                "Distinct-2": np.nan,
                "Avg_Line_Length": np.nan,
                "Line_Var": np.nan,
                "Entropy": np.nan,
                "Lexical_Density": np.nan,
            }
            continue

        ppl_scores = []
        distinct1_scores = []
        distinct2_scores = []
        avg_len_scores = []
        var_len_scores = []
        entropy_scores = []
        lex_density_scores = []

        for text in tqdm(col_values, desc=f"Processing {col}", leave=False):
            ppl_scores.append(compute_perplexity(text))
            distinct1_scores.append(compute_distinct_n(text, 1))
            distinct2_scores.append(compute_distinct_n(text, 2))
            avg_len, var_len = compute_length_variance(text)
            avg_len_scores.append(avg_len)
            var_len_scores.append(var_len)
            entropy_scores.append(compute_entropy(text))
            lex_density_scores.append(compute_lexical_density(text))

        results[col] = {
            "PPL": round(np.nanmean(ppl_scores), 4),
            "Distinct-1": round(np.nanmean(distinct1_scores), 4),
            "Distinct-2": round(np.nanmean(distinct2_scores), 4),
            "Avg_Line_Length": round(np.nanmean(avg_len_scores), 4),
            "Line_Var": round(np.nanmean(var_len_scores), 4),
            "Entropy": round(np.nanmean(entropy_scores), 4),
            "Lexical_Density": round(np.nanmean(lex_density_scores), 4),
        }

    return pd.DataFrame(results)

if __name__ == "__main__":
    csv_path = "/Users/liyuanheng/Desktop/NLP/ProjectCode/deeprapper_pro/data/rap_lyrics_ablation.csv"

    cols = [
        "Reference lyrics",
        "gpt-4.1-base",
        "gpt-4.1-chrhyme-S-5",
        "gpt-4.1-chrhyme-S-10",
        "gpt-4.1-chrhyme-S-15",
        "gpt-4.1-chrhyme-S-20",
        "gpt-4.1-chrhyme",
    ]

    result_table = evaluate_csv(csv_path, cols)
    print(result_table)
