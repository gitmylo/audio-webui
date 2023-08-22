import shlex
import subprocess
import tempfile
import traceback
from typing import Any

import PIL
import gradio
import numpy as np
import scipy.io.wavfile
from gradio import processing_utils, utils
from gradio_client.client import DEFAULT_TEMP_DIR
from matplotlib import pyplot as plt

import setup_tools.os
from webui.args import args
from webui.ui.tabs import settings


def showwaves(
        audio: str | tuple[int, np.ndarray]
):
    try:
        if isinstance(audio, str):
            audio_file = audio
            audio = processing_utils.audio_from_file(audio)
        else:
            tmp_wav = tempfile.NamedTemporaryFile(dir=DEFAULT_TEMP_DIR, suffix=".wav", delete=False)
            processing_utils.audio_to_file(audio[0], audio[1], tmp_wav.name, format="wav")
            audio_file = tmp_wav.name

        output_mp4 = tempfile.NamedTemporaryFile(dir=DEFAULT_TEMP_DIR, suffix=".mkv", delete=False)

        command = f'ffmpeg -y -i {audio_file} -filter_complex "[0:a]showwaves=s=1280x720:mode=line,format=yuv420p[v]" -map "[v]" -map 0:a -preset veryfast -c:v libx264 -c:a copy {output_mp4.name}'

        if not setup_tools.os.is_windows():
            command = shlex.split(command)

        run = subprocess.run(command)
        return output_mp4.name if run.returncode == 0 else None
    except Exception as e:
        traceback.print_exception(e)
        return None


def make_waveform(
    audio: str | tuple[int, np.ndarray],
    *,
    bg_color: str = "#f3f4f6",
    bg_image: str | None = None,
    fg_alpha: float = 1.00,  # (was 0.75)
    bars_color: str | tuple[str, str] = ("#65B5FF", "#1B76FF"),  # (was ("#fbbf24", "#ea580c"))
    bar_count: int = 50,
    bar_width: float = 0.6,
    wav_type: str = None
):
    if wav_type is None:
        wav_type = settings.get('wav_type').casefold()
    match wav_type:
        case 'showwaves':
            return showwaves(audio)
        case 'gradio':
            return gradio.make_waveform(audio, bg_color=bg_color, bg_image=bg_image, fg_alpha=fg_alpha, bars_color=bars_color, bar_count=bar_count, bar_width=bar_width)
        case 'none' | _:
            return None
