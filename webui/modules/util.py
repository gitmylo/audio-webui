import gradio
import numpy as np


def make_waveform(
    audio: str | tuple[int, np.ndarray],
    *,
    bg_color: str = "#f3f4f6",
    bg_image: str | None = None,
    fg_alpha: float = 1.00,  # (was 0.75)
    bars_color: str | tuple[str, str] = ("#65B5FF", "#1B76FF"),  # (was ("#fbbf24", "#ea580c"))
    bar_count: int = 50,
    bar_width: float = 0.6,
):
    return gradio.make_waveform(audio, bg_color=bg_color, bg_image=bg_image, fg_alpha=fg_alpha, bars_color=bars_color, bar_count=bar_count, bar_width=bar_width)
