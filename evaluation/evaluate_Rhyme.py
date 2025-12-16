from pypinyin import pinyin, Style
import numpy as np


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
    unique_rhymes = set(r for r in rhyme_seq if r)
    return round(len(unique_rhymes) / len(rhyme_seq), 4)


if __name__ == "__main__":
    text = (
        "压力太大了\n"
        "差一点炸了\n"
        "心脏在跳了\n"
        "兄弟们笑了\n"
        "梦想太远了\n"
        "现实太狠了\n"
        "我还没怕了\n"
    )

    n = 2
    print(f"Using {n}-gram rhyme units:")
    print("Rhyme Accuracy:", compute_rhyme_accuracy(text, n))
    print("Rhyme Density:", compute_rhyme_density(text, n))
    print("Combo-N:", compute_combo_n(text, n))
    print("Rhyme Diversity:", compute_rhyme_diversity(text, n))
