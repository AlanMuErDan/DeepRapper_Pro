import os

DEFAULT_SYSTEM_PROMPTS = {
    "ollama": (
        "你是一名中文说唱创作者，请根据提供的信息续写歌词，语言简洁、押韵准确。"
    ),
    "openai": (
        "你是一位擅长中文说唱的资深作词人。"
        "请根据提供的信息续写歌词，语言简洁、押韵准确。。"
    ),
}

OPENAI_SLUR_PROMPT = (
    "你是一位擅长中文说唱的资深作词人，你非常粗鲁喜欢在歌词里说一些脏话，务必要说但不宜过多"
    "请根据提供的信息续写歌词，语言简洁、押韵准确。。"
)


def _resolve_system_prompt(provider: str, slur: bool = False) -> str:
    provider = (provider or "ollama").lower()
    env_var = None
    if provider == "openai":
        env_var = os.getenv("OPENAI_SYSTEM_PROMPT")
    else:
        env_var = os.getenv("OLLAMA_SYSTEM_PROMPT")
    if env_var:
        return env_var.strip()
    if provider == "openai" and slur:
        return OPENAI_SLUR_PROMPT
    return DEFAULT_SYSTEM_PROMPTS.get(provider, DEFAULT_SYSTEM_PROMPTS["ollama"])


def build_prompt(
    first_line: str,
    theme: str,
    content: str,
    rhymes: dict,
    rhyme_len: int,
    provider: str = "ollama",
    system_prompt: str | None = None,
    enable_slur: bool = False,
    frequency: str = "M",
    use_rhyme_hint: bool = True,
):
    """Construct the generation prompt for the target LLM."""
    rhyme_words = []
    if use_rhyme_hint:
        seen = set()
        for title, words in (rhymes or {}).items():
            limit = 8 if "俗语" in title else 15
            if title and "筛选" in title:
                limit = len(words)
            for term in words[:limit]:
                if term not in seen:
                    seen.add(term)
                    rhyme_words.append(term)

    truncated = False
    if len(rhyme_words) > 120:
        rhyme_words = rhyme_words[:120]
        truncated = True

    rhyme_hint = "、".join(rhyme_words)

    if truncated:
        rhyme_hint += "（更多押韵词可自行搭配）"

    theme_text = theme if theme else "暂无特定主题，自由发挥"
    content_text = content if content else "围绕首句延展画面感与情绪"

    instructions = (
        system_prompt.strip()
        if system_prompt
        else _resolve_system_prompt(provider, slur=enable_slur)
    )

    if provider == "ollama":
        prompt = (
            f"{instructions}\n\n"
            f"歌词首句：{first_line}\n"
            f"主题：{theme_text}\n"
            f"内容侧重：{content_text}\n\n"
        )
    
    if provider == "openai":
        frequency_map = {
            "F": "每句歌词的长度要很长。",
            "M": "每句歌词长度适中。",
            "S": "每句歌词的长度要很短。",
        }
        freq_key = (frequency or "M").strip().upper()
        freq_instruction = frequency_map.get(freq_key)

        output_lines = [
            "1. 输出行数够一首歌的长度，每行字数与首句接近，严禁添加标题、解释或额外空行。",
            "2. 内容紧扣主题与提示，确保语义连贯，格式不必太过拘束，歌词必须是说唱的风格。",
            "3. 首句话必须是输入的首句。",
            "4. 押韵参考词仅供参考，不必强行使用，如与主题内容相关请尽力使用。",
            "5. 句意连贯通顺有内容，契合主题和内容侧重",
            "6. 不要使用标点符号，直接换行分隔每行歌词。",
        ]
        if freq_instruction:
            output_lines.append(f"6. {freq_instruction}")
        output_block = "\n".join(output_lines)

        rhyme_block = ""
        if use_rhyme_hint:
            hint_text = (
                rhyme_hint
                if rhyme_hint
                else "当前无示例，请自行构思押韵词，但务必押韵。"
            )
            rhyme_block = f"- 可参考押韵词：{hint_text}\n"

        prompt = (
            f"{instructions}\n\n"
            "【写作背景】\n"
            f"- 首句：{first_line}\n"
            f"- 主题：{theme_text}\n"
            f"- 内容侧重：{content_text}\n\n"
            "【押韵要求】\n"
            f"- 押韵字数：每行末尾需与{rhyme_len}个字“{first_line[-rhyme_len:]}”押韵。\n"
            f"{rhyme_block}"
            "\n【输出规范】\n"
            f"{output_block}"
        )

    return prompt.strip()
