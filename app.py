import os
import gradio as gr
from llm_utils import get_available_models, OPENAI_MODEL_CHOICES
from main import generate_lyrics


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "yes", "on"}:
            return True
        if val in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)

OLLAMA_MODELS = get_available_models()
if not OLLAMA_MODELS:
    fallback_model = os.getenv("OLLAMA_DEFAULT_MODEL")
    if fallback_model:
        OLLAMA_MODELS = [fallback_model]

def interface_fn(
    first_line,
    theme,
    content,
    rhyme_len,
    provider,
    ollama_model,
    openai_model,
    temperature,
    enable_slur,
    frequency,
    semantic_filter_mode,
    semantic_top_k,
    exclude_chrhyme,
):
    provider = (provider or "ollama").lower()
    selected_model = (
        (openai_model or "").strip()
        if provider == "openai"
        else (ollama_model or "").strip()
    )
    result = generate_lyrics(
        first_line=first_line.strip(),
        theme=(theme or "").strip(),
        content=(content or "").strip(),
        rhyme_len=int(rhyme_len),
        model_name=selected_model or None,
        provider=provider,
        temperature=float(temperature),
        max_tokens=1280,
        enable_slur=_to_bool(enable_slur),
        frequency=(frequency or "M"),
        semantic_filter_mode=(semantic_filter_mode or "off"),
        semantic_top_k=int(semantic_top_k),
        exclude_chrhyme=_to_bool(exclude_chrhyme),
    )
    if isinstance(result, dict):
        metrics = result.get("metrics", [])
        table_rows = [
            [
                row.get("Metric"),
                row.get("Threshold"),
                row.get("Score"),
                row.get("Checked"),
            ]
            for row in metrics
        ]
        return result.get("text", ""), table_rows
    return result, []

demo = gr.Interface(
    fn=interface_fn,
    inputs=[
        gr.Textbox(label="Opening Line", placeholder="例如：烟雨江南梦初醒"),
        gr.Textbox(label="Theme", placeholder="例如：追梦"),
        gr.Textbox(label="Content Hint", placeholder="例如：描写夜色与远方的呼唤"),
        gr.Slider(1, 4, value=2, step=1, label="Rhyme Length n"),
        gr.Dropdown(
            choices=["ollama", "openai"],
            value="ollama",
            label="Model Provider",
            info="Choose between local Ollama models or the OpenAI API",
        ),
        gr.Dropdown(
            choices=OLLAMA_MODELS or [],
            value=OLLAMA_MODELS[0] if OLLAMA_MODELS else None,
            label="Ollama Model",
            info="If empty, check your local Ollama setup or set OLLAMA_DEFAULT_MODEL",
            allow_custom_value=True,
        ),
        gr.Dropdown(
            choices=OPENAI_MODEL_CHOICES,
            value=os.getenv("OPENAI_DEFAULT_MODEL", OPENAI_MODEL_CHOICES[0]),
            label="OpenAI Model",
            info="Requires OPENAI_API_KEY environment variable",
            allow_custom_value=True,
        ),
        gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature (creativity)"),
        gr.Checkbox(label="Include slurs", value=False),
        gr.Radio(
            choices=["F", "M", "S"],
            value="M",
            label="Lyric Pace [F fast | M medium | S slow]",
            info="Controls the average line length",
        ),
        gr.Radio(
            choices=["off", "bertscore", "llm"],
            value="off",
            label="Top-k filtering mode",
            info="Choose no filtering, BERTScore filtering, or LLM filtering",
        ),
        gr.Slider(
            5,
            80,
            value=30,
            step=5,
            label="Semantic filter top-k",
            info="Used by the selected filtering mode to keep the most relevant rhyming words",
        ),
        gr.Checkbox(
            label="Exclude chrhyme hints",
            value=False,
            info="If enabled, do not inject chrhyme candidates into the prompt",
        ),
    ],
    outputs=[
        gr.Textbox(label="AI Rap Lyrics", lines=8),
        gr.Dataframe(
            headers=["Metric", "Threshold", "Score", "Checked"],
            datatype=["str", "number", "number", "str"],
            label="Quality Checklist",
            col_count=(4, "fixed"),
            row_count=(9, "fixed"),
        ),
    ],
    title="🎤 DeepRapper Pro",
    description="Enter the opening line, theme, and content hints. Choose rhyme length, creativity, slur option, and pace to let the AI craft rap lyrics. 🎶"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
