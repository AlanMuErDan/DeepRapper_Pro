import subprocess
import re
from typing import Dict, List


def _extract_words(section_content: str) -> List[str]:
    """Parse chrhyme raw text into a deduplicated list of terms."""
    candidates = re.findall(r"'([^']+)'", section_content)
    if not candidates:
        rough = re.split(r"[\s,，]+", section_content)
        candidates = [item.strip("[]'\"") for item in rough if item.strip("[]'\"")]

    seen = set()
    words = []
    for term in candidates:
        if term and term not in seen:
            seen.add(term)
            words.append(term)
    return words


def get_rhyming_words(word: str, n: int = 2) -> Dict[str, List[str]]:
    """Call the chrhyme CLI tool and parse the grouped rhyme words."""
    try:
        result = subprocess.run(
            ["chrhyme"], input=f"{word}\n0\n0\n", capture_output=True, text=True
        )
        output = result.stdout

        sections = re.split(r">>>", output)
        rhyme_dict = {}
        for sec in sections:
            match = re.match(r"\s*([^:\n]+):\s*\n(.*)", sec, re.S)
            if match:
                title = match.group(1).strip()
                content = match.group(2).strip()
                words = _extract_words(content)
                rhyme_dict[title] = words
        return rhyme_dict
    except Exception as e:
        print(f"[ERROR] chrhyme调用失败: {e}")
        return {}
