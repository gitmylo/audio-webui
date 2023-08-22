import shlex
import subprocess
from tempfile import NamedTemporaryFile
from gradio_client.client import DEFAULT_TEMP_DIR

import gradio

from setup_tools.os import is_windows


def run_command(command) -> subprocess.CompletedProcess:
    if not is_windows():
        command = shlex.split(command)
    return subprocess.run(command)


def clear(count):
    return [None]*count


def image_audio():
    with gradio.Row():
        with gradio.Column():
            image = gradio.Image(label='Image', type='filepath')
            audio = gradio.Audio(label='Audio', type='filepath')
        output = gradio.PlayableVideo(label='Output video')
    with gradio.Row():
        combine_button = gradio.Button('Combine', variant='primary')
        clear_button = gradio.Button('Clear')

    def image_audio_func(i, a):
        out_file = NamedTemporaryFile(delete=False, suffix='.mp4').name
        result = run_command(f'ffmpeg -y -loop 1 -i "{i}" -i "{a}" -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest "{out_file}"')
        assert result.returncode == 0
        return out_file

    combine_button.click(fn=image_audio_func, inputs=[image, audio], outputs=output)
    clear_button.click(fn=lambda: clear(3), outputs=[image, audio, output])


def video_audio():
    with gradio.Row():
        with gradio.Column():
            video = gradio.Video(label='Video')
            audio = gradio.Audio(label='Audio', type='filepath')
        output = gradio.PlayableVideo(label='Output video')
    with gradio.Row():
        combine_button = gradio.Button('Combine', variant='primary')
        clear_button = gradio.Button('Clear')

    def image_audio_func(v, a):
        out_file = NamedTemporaryFile(dir=DEFAULT_TEMP_DIR, delete=False, suffix='.mp4').name
        result = run_command(f'ffmpeg -y -i "{v}" -i "{a}" -c:v copy -map 0:v:0 -map 1:a:0 -c:a aac -b:a 192k -shortest "{out_file}"')
        assert result.returncode == 0
        return out_file

    combine_button.click(fn=image_audio_func, inputs=[video, audio], outputs=output)
    clear_button.click(fn=lambda: clear(3), outputs=[video, audio, output])


def video_strip():
    with gradio.Row():
        video = gradio.File(label='Video or other file with audio.')
        output = gradio.Audio(label='Audio', interactive=False)
    with gradio.Row():
        strip_button = gradio.Button('Strip', variant='primary')
        clear_button = gradio.Button('Clear')

    def strip(a):
        out_file = NamedTemporaryFile(dir=DEFAULT_TEMP_DIR, delete=False, suffix='.wav').name
        result = run_command(f'ffmpeg -y -i "{a.name}" "{out_file}"')
        assert result.returncode == 0
        return out_file

    strip_button.click(fn=strip, inputs=video, outputs=output)
    clear_button.click(fn=lambda: clear(2), outputs=[video, output])


def ffmpeg_utils_tab():
    with gradio.Tabs():
        with gradio.Tab('📽 = 🔊'):
            video_strip()
        with gradio.Tab('📷 + 🔊 = 📽'):
            image_audio()
        with gradio.Tab('📽 + 🔊 = 📽'):
            video_audio()
