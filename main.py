import re
from typing import Dict, List, Optional, Tuple

from rhyme_utils import get_rhyming_words
from prompt_templates import build_prompt
from llm_utils import generate_with_llm
from semantic_filter import apply_semantic_filter
from metrics_utils import (
    compute_distinct_n,
    compute_combo_n,
    compute_rhyme_accuracy,
    compute_rhyme_density,
    compute_semantic_bertscore,
    compute_topic_similarity,
)
from log_utils import print_step


def _postprocess_lyrics(lyrics: str, provider: str, model: Optional[str]) -> Optional[str]:
    if not (lyrics or "").strip():
        return None
    prompt = (
        "你是一个歌词数据清理助手。\n"
        "请将我提供的中文说唱歌词整理成一个单个字符串，格式要求如下：\n"
        "1. 整首歌词用双引号包裹。\n"
        "2. 每一行之间用 \\n 连接。\n"
        "3. 不要添加任何额外说明、解释或标点。\n"
        "4. 输出中不能出现代码块符号或其他格式化符号。\n"
        "5. 只输出最终字符串结果。\n\n"
        "以下是需要处理的歌词：\n"
        f"{lyrics.strip()}"
    )
    response = generate_with_llm(
        prompt,
        model=model,
        provider=provider,
        temperature=0.0,
        max_tokens=256,
    )
    if not response or response.startswith("❌"):
        return None
    return response.strip()


def _normalize_cleaned_lyrics(cleaned: Optional[str]) -> str:
    if not cleaned:
        return ""
    text = cleaned.strip()
    if text.startswith("\"") and text.endswith("\"") and len(text) >= 2:
        text = text[1:-1]
    text = text.replace("\\n", "\n")
    return text.strip()


def _llm_semantic_judge(
    theme: str,
    content: str,
    lyrics: str,
    provider: str,
    model: Optional[str],
) -> Optional[float]:
    if not lyrics.strip():
        return None
    prompt = (
        "你是一个中文说唱歌词审查专家。\n"
        "请根据主题和内容提示，评估歌词是否语义契合。\n"
        "从 1-10 打分，可使用小数，分数越高表示越契合。\n"
        "只输出最终分数，不要任何解释。\n\n"
        f"主题：{theme or '（无）'}\n"
        f"内容提示：{content or '（无）'}\n"
        f"歌词：\n{lyrics.strip()}\n"
        "分数："
    )
    response = generate_with_llm(
        prompt,
        model=model,
        provider=provider,
        temperature=0.0,
        max_tokens=16,
    )
    if not response or response.startswith("❌"):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", response)
    if not match:
        return None
    return float(match.group(1))


def _llm_profanity_score(
    lyrics: str,
    provider: str,
    model: Optional[str],
) -> Optional[float]:
    text = (lyrics or "").strip()
    if not text:
        return None
    prompt = (
        "你是一个中文歌词敏感度审查员。\n"
        "我会给你一首中文说唱歌词，请评估其中是否包含脏话、侮辱或低俗冒犯内容。\n"
        "从 0-1 打分，可使用小数。0 表示完全安全，1 表示极其严重且无法接受。\n"
        "如果只有轻微或模棱两可的内容，可以给出 0.1-0.3 左右的分数。\n"
        "严格只输出数字，不要解释。\n\n"
        f"歌词：\n{text}\n"
        "分数："
    )
    response = generate_with_llm(
        prompt,
        provider=provider,
        model=model,
        temperature=0.0,
        max_tokens=16,
    )
    if not response or response.startswith("❌"):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", response)
    if not match:
        return None
    return float(match.group(1))


def _summarize_lyrics_content(
    lyrics: str,
    provider: str,
    model: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    text = (lyrics or "").strip()
    if not text:
        return None, None

    prompt = (
        "你不是一个分析机器，而是一位真正热爱音乐、懂情感的听众。\n"
        "我会给你一首完整的中文说唱歌词，请你像听完这首歌后写下感受那样，"
        "用你的人类直觉去提炼这首歌的主题和内容。\n\n"
        "要求如下：\n"
        "1 内容：用一句第一人称的自然表达，语气真诚，不超过30字。\n"
        "2 严格按照格式输出，不要任何额外解释。\n\n"
        "格式：\n"
        "内容：<一句第一人称的自然表达>\n\n"
        f"下面是歌词内容：\n{text}"
    )

    response = generate_with_llm(
        prompt,
        provider=provider,
        model=model,
        temperature=0.2,
        max_tokens=256,
    )
    if not isinstance(response, str) or response.startswith("❌"):
        return None, None

    theme_line = None
    content_line = None
    for line in response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("主题：") and theme_line is None:
            theme_line = stripped[len("主题：") :].strip()
            continue
        if stripped.startswith("内容：") and content_line is None:
            content_line = stripped[len("内容：") :].strip()
    return theme_line or None, content_line or None


def _build_metrics_table(
    lyrics: str,
    reference: str,
    theme: str,
    content: str,
    rhyme_len: int,
    provider: str,
    model: Optional[str],
) -> List[Dict[str, object]]:
    distinct1 = compute_distinct_n(lyrics, 1)
    distinct2 = compute_distinct_n(lyrics, 2)
    accuracy = compute_rhyme_accuracy(lyrics, rhyme_len)
    density = compute_rhyme_density(lyrics, rhyme_len)
    combo_n = compute_combo_n(lyrics, rhyme_len)
    bert = compute_semantic_bertscore(reference, lyrics)
    _, summarized_content = _summarize_lyrics_content(lyrics, provider, model)
    topic = None
    reference_for_topic = (content or "").strip()
    if reference_for_topic and summarized_content:
        topic = compute_topic_similarity(reference_for_topic, summarized_content)
        print("[TopicSim] User content:", reference_for_topic)
        print("[TopicSim] LLM summary:", summarized_content)
    judge = _llm_semantic_judge(theme, content, lyrics, provider, model)
    profanity = _llm_profanity_score(lyrics, provider, model)

    metrics = [
        ("Distinct-1", 0.5, distinct1),
        ("Distinct-2", 0.8, distinct2),
        ("Rhyme Accuracy", 0.15, accuracy),
        ("Rhyme Density", 0.2, density),
        ("Combo-N", 2.0, combo_n),
        ("BERTScore", 0.8, bert),
        ("Topic_Sim", 0.6, topic),
        ("LLM as a Judge", 9.0, judge),
        ("Profanity Check", 0.0, profanity),
    ]

    rows: List[Dict[str, object]] = []
    for name, threshold, score in metrics:
        if score is None:
            checked = "—"
        else:
            if name == "Profanity Check":
                checked = "✔️" if abs(score) < 1e-6 else "✘"
            else:
                checked = "✔️" if score >= threshold else "✘"
        rows.append(
            {
                "Metric": name,
                "Threshold": threshold,
                "Score": score,
                "Checked": checked,
            }
        )
    return rows

def generate_lyrics(
    first_line: str,
    theme: str,
    content: str,
    rhyme_len: int,
    model_name: Optional[str] = None,
    provider: str = "ollama",
    temperature: float = 0.7,
    max_tokens: int = 320,
    system_prompt: Optional[str] = None,
    enable_slur: bool = False,
    frequency: str = "M",
    semantic_filter_mode: str = "off",
    semantic_top_k: int = 30,
    exclude_chrhyme: bool = False,
):
    first_line = first_line.strip()
    theme = (theme or "").strip()
    content = (content or "").strip()
    reference_text = " ".join(
        part for part in [first_line, theme, content] if part
    ).strip()

    def _error_response(message: str) -> Dict[str, object]:
        placeholder = _build_metrics_table(
            lyrics="",
            reference=reference_text,
            theme=theme,
            content=content,
            rhyme_len=rhyme_len,
            provider=provider,
            model=model_name,
        )
        return {"text": message, "metrics": placeholder}

    if len(first_line) < rhyme_len:
        return _error_response("❌ 错误：歌词长度不足以提取韵脚。")

    filter_mode = (semantic_filter_mode or "").strip().lower()
    filter_enabled = filter_mode not in {"", "off", "none"}

    rhyme_dict = {}
    if not exclude_chrhyme:
        rhyme_base = first_line[-rhyme_len:]
        rhyme_dict = get_rhyming_words(rhyme_base, rhyme_len)
        if not rhyme_dict:
            return _error_response("⚠️ 未找到押韵词，请检查chrhyme配置。")
        print_step("Step 1 - chrhyme Rhymes", _format_rhyme_dict_for_log(rhyme_dict))

        rhyme_dict, applied = apply_semantic_filter(
            rhyme_dict=rhyme_dict,
            theme=theme,
            content=content,
            top_k=semantic_top_k,
            enabled=filter_enabled,
            filter_mode=filter_mode,
            llm_provider=provider,
            llm_model=model_name,
        )
        if not applied:
            if filter_enabled:
                print_step("Step 2 - Semantic Filtering", "语义过滤已启用但未生效，保留原始 chrhyme 韵脚。")
            else:
                print_step("Step 2 - Semantic Filtering", "语义过滤未启用，使用原始 chrhyme 韵脚。")
        if applied and not any(rhyme_dict.values()):
            return _error_response("⚠️ 语义过滤后没有可用的押韵词，请调整筛选阈值。")
    else:
        print_step("Step 1 - chrhyme Rhymes", "用户选择排除 chrhyme hints，跳过韵脚注入。")
        print_step("Step 2 - Semantic Filtering", "因未使用 chrhyme 提示，语义过滤被跳过。")

    prompt = build_prompt(
        first_line=first_line,
        theme=theme,
        content=content,
        rhymes=rhyme_dict,
        rhyme_len=rhyme_len,
        provider=provider,
        system_prompt=system_prompt,
        enable_slur=enable_slur,
        frequency=frequency,
        use_rhyme_hint=not exclude_chrhyme,
    )
    print_step("Step 3 - Generation Prompt", prompt)

    generated = generate_with_llm(
        prompt,
        model=model_name,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not isinstance(generated, str):
        return _error_response("⚠️ 模型返回异常。")

    if generated.startswith("❌"):
        return _error_response(generated)
    print_step("Step 3 - Generation Output", generated.strip())

    cleaned = _postprocess_lyrics(generated, provider=provider, model=model_name)
    display_text = cleaned or generated
    metrics_text = _normalize_cleaned_lyrics(cleaned) or generated.strip()
    print_step("Step 3 - Final Lyrics", display_text.strip())

    metrics_table = _build_metrics_table(
        lyrics=metrics_text,
        reference=reference_text,
        theme=theme,
        content=content,
        rhyme_len=rhyme_len,
        provider=provider,
        model=model_name,
    )

    return {"text": display_text, "metrics": metrics_table}
def _format_rhyme_dict_for_log(rhyme_dict: Dict[str, List[str]]) -> str:
    if not rhyme_dict:
        return "（无押韵词返回）"
    lines = []
    for label, words in rhyme_dict.items():
        joined = "、".join(words) if words else "（空）"
        lines.append(f"{label}: {joined}")
    return "\n".join(lines)
