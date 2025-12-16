import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import textstat
import jieba
import numpy as np 

def compute_perplexity(text, model_name="gpt2"):
    """Compute perplexity with GPT-2 as a fluency proxy."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        max_length = model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len
            lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return float(ppl)
    


def tokenize_text(text):
    """Tokenize by combining jieba output with whitespace splitting."""
    text = text.replace("\n", " ").strip()
    tokens = []
    for word in jieba.cut(text):
        sub_tokens = word.strip().split()
        tokens.extend([w for w in sub_tokens if w])
    return tokens



def compute_distinct_n(text, n=1):
    """Compute the Distinct-n ratio."""
    tokens = tokenize_text(text)
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    distinct = len(set(ngrams)) / len(ngrams)
    return round(distinct, 4)



def compute_length_variance(text):
    """Return the mean line length and variance."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    lengths = [len(tokenize_text(line)) for line in lines]
    avg_len = np.mean(lengths)
    var_len = np.var(lengths)
    return round(avg_len, 2), round(var_len, 2)


def compute_entropy(text):
    """Compute Shannon entropy over the token frequency distribution."""
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens)
    probs = [count / total for count in freq.values()]
    entropy = -sum(p * math.log(p + 1e-10, 2) for p in probs)
    return round(entropy, 4)


def compute_lexical_density(text):
    """Estimate lexical density via coarse POS filtering."""
    import jieba.posseg as pseg

    tokens = list(pseg.cut(text))
    if not tokens:
        return 0.0

    function_tags = set(["u", "p", "r", "c", "d", "x", "w", "t", "m"])
    content_count = sum(1 for w, f in tokens if f and f[0] not in function_tags)
    density = content_count / len(tokens)
    return round(density, 4)





if __name__ == "__main__":
    text = "MY BRO 玩双色球 玩菠菜 赛马 打快三BRO\n带劳力士吃快餐\n开香堂和手足们拜山头\n玩的是荣华富贵 和气生财 不是六臂和三头\n从新加坡到马拉西亚 *到上海外滩BRO\n拿三条A 通杀牌 打东南西北中发白\n叔辈们喝的是年份茅台茶杯装的是冬瓜排骨\n东京 东莞 东港 东南亚 我们来 抢了东家的台\n狂风骤雨兄弟们见的太多 你就尽情让风刮来\n菲律宾没有雪 马尼拉没有爱 柬埔寨的男人不回家\n兄弟们24 HOUR 在澳门 野心太大 不陪她\n我要拿钱 拿分 拿走属于我的一切 装满后备箱FULL THIS CAR\n你们玩的那套早已经过时让西固城的小伙来DO THIS PART\n没钞票哪有情义买\n我们靠的就是名气拽\nFEEL LIKE东兴乌鸦 难办就别办 十八万要零一百\n穿VERSACE开570的\n半两的量 我要赌一瓶的\n抽黑兰州不是抽利群的\n不用自我介绍你也知道我的名字"

    print("Perplexity (GPT2):", compute_perplexity(text))
    print("Distinct-1:", compute_distinct_n(text, 1))
    print("Distinct-2:", compute_distinct_n(text, 2))

    avg_len, var_len = compute_length_variance(text)
    print(f"Average line length: {avg_len}")
    print(f"Line length variance: {var_len}")
    print("Entropy:", compute_entropy(text))
    print("Lexical Density:", compute_lexical_density(text))
