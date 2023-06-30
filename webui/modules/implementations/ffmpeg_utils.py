import shlex
import subprocess
from tempfile import NamedTemporaryFile

import gradio

from setup_tools.os import is_windows


def run_command(command) -> subprocess.CompletedProcess:
    if not is_windows():
        command = shlex.split(command)
    return subprocess.run(command)


def image_audio():
    with gradio.Row():
        with gradio.Column():
            image = gradio.Image(label='Image', type='filepath')
            audio = gradio.Audio(label='Audio', type='filepath')
        output = gradio.PlayableVideo(label='Output video')
    combine_button = gradio.Button('Combine', variant='primary')

    def image_audio_func(i, a):
        out_file = NamedTemporaryFile(delete=False, suffix='.mp4').name
        result = run_command(f'ffmpeg -y -loop 1 -i "{i}" -i "{a}" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest "{out_file}"')
        assert result.returncode == 0
        return out_file

    combine_button.click(fn=image_audio_func, inputs=[image, audio], outputs=output)


def video_audio():
    with gradio.Row():
        with gradio.Column():
            video = gradio.Video(label='Video')
            audio = gradio.Audio(label='Audio', type='filepath')
        output = gradio.PlayableVideo(label='Output video')
    combine_button = gradio.Button('Combine', variant='primary')

    def image_audio_func(v, a):
        out_file = NamedTemporaryFile(delete=False, suffix='.mp4').name
        result = run_command(f'ffmpeg -y -i "{v}" -i "{a}" -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -shortest "{out_file}"')
        assert result.returncode == 0
        return out_file

    combine_button.click(fn=image_audio_func, inputs=[video, audio], outputs=output)


def ffmpeg_utils_tab():
    with gradio.Tabs():
        with gradio.Tab('🖼 + 🔊 = 📽'):
            image_audio()
        with gradio.Tab('📽 + 🔊 = 📽'):
            video_audio()
