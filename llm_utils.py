import os
from typing import Iterable, List, Optional
import ollama

OPENAI_MODEL_CHOICES: List[str] = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
]


def get_available_models() -> List[str]:
    """Return the list of locally installed Ollama models."""
    try:
        response = ollama.list()
        raw_models: Iterable = getattr(response, "models", None)

        names: List[str] = []
        if raw_models is None and isinstance(response, dict):
            raw_models = response.get("models")

        if raw_models:
            for item in raw_models:
                name = None
                if isinstance(item, str):
                    name = item
                elif isinstance(item, dict):
                    name = item.get("model") or item.get("name")
                else:
                    name = getattr(item, "model", None) or getattr(item, "name", None)
                if name:
                    names.append(name)

        deduped = sorted(dict.fromkeys(names))
        return deduped
    except Exception as exc:
        print(f"[WARN] 无法获取 Ollama 模型列表: {exc}")
        return []


def _resolve_model_name(explicit_model: Optional[str]) -> str:
    """Resolve the Ollama model name from parameters, env vars, or installed models."""
    available = get_available_models()

    if explicit_model:
        if available and explicit_model not in available:
            raise ValueError(
                f"当前环境未安装模型 `{explicit_model}`，可用模型：{', '.join(available)}"
            )
        return explicit_model

    default_model = os.getenv("OLLAMA_DEFAULT_MODEL")
    if default_model:
        if available and default_model not in available:
            raise ValueError(
                f"环境变量 OLLAMA_DEFAULT_MODEL 指定的模型 `{default_model}` 不可用，"
                f"可用模型：{', '.join(available)}"
            )
        return default_model

    if available:
        return available[0]

    return "qwen3:4b"


def generate_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.8
):
    """Generate text with a local Ollama model."""
    try:
        resolved_model = _resolve_model_name(model)

        response = ollama.generate(
            model=resolved_model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response["response"].strip()

    except Exception as e:
        return f"❌ 本地模型调用失败：{e}"


def _resolve_openai_model(explicit_model: Optional[str]) -> str:
    if explicit_model:
        return explicit_model
    env_model = os.getenv("OPENAI_DEFAULT_MODEL")
    if env_model:
        return env_model
    return OPENAI_MODEL_CHOICES[0]


def generate_with_openai(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 400,
    temperature: float = 0.7
) -> str:
    """Generate text through the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "❌ 缺少 OPENAI_API_KEY，无法调用 OpenAI 模型。"

    try:
        from openai import OpenAI
    except ImportError:
        return "❌ 未安装 openai 包，请先执行 `pip install openai`。"

    model_name = _resolve_openai_model(model)
    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        collected: List[str] = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "text":
                        collected.append(content.text)
        if collected:
            return "\n".join(part.strip() for part in collected if part.strip())

        return "❌ OpenAI 返回格式异常，未解析到文本。"

    except Exception as exc:
        return f"❌ OpenAI 模型调用失败：{exc}"


def generate_text(
    prompt: str,
    provider: str = "ollama",
    model: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.8
) -> str:
    provider = (provider or "ollama").lower()
    if provider == "openai":
        return generate_with_openai(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return generate_with_ollama(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def generate_with_llm(
    prompt: str,
    model: Optional[str] = None,
    provider: str = "ollama",
    max_tokens: int = 300,
    temperature: float = 0.8
) -> str:
    """Backwards compatible wrapper that calls the selected LLM provider."""
    return generate_text(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
