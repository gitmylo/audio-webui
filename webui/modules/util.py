import subprocess
import tempfile
from typing import Any

import PIL
import gradio
import numpy as np
from gradio import processing_utils, utils
from matplotlib import pyplot as plt


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
    """
        Generates a waveform video from an audio file. Useful for creating an easy to share audio visualization. The output should be passed into a `gr.Video` component.
        Parameters:
            audio: Audio file path or tuple of (sample_rate, audio_data)
            bg_color: Background color of waveform (ignored if bg_image is provided)
            bg_image: Background image of waveform
            fg_alpha: Opacity of foreground waveform
            bars_color: Color of waveform bars. Can be a single color or a tuple of (start_color, end_color) of gradient
            bar_count: Number of bars in waveform
            bar_width: Width of bars in waveform. 1 represents full width, 0.5 represents half width, etc.
        Returns:
            A filepath to the output video.
        """
    if isinstance(audio, str):
        audio_file = audio
        audio = processing_utils.audio_from_file(audio)
    else:
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        processing_utils.audio_to_file(audio[0], audio[1], tmp_wav.name, format="wav")
        audio_file = tmp_wav.name
    duration = round(len(audio[1]) / audio[0], 4)

    # Helper methods to create waveform
    def hex_to_rgb(hex_str):
        return [int(hex_str[i: i + 2], 16) for i in range(1, 6, 2)]

    def get_color_gradient(c1, c2, n):
        assert n > 1
        c1_rgb = np.array(hex_to_rgb(c1)) / 255
        c2_rgb = np.array(hex_to_rgb(c2)) / 255
        mix_pcts = [x / (n - 1) for x in range(n)]
        rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
        return [
            "#" + "".join(f"{int(round(val * 255)):02x}" for val in item)
            for item in rgb_colors
        ]

    # Reshape audio to have a fixed number of bars
    samples = audio[1]
    if len(samples.shape) > 1:
        samples = np.mean(samples, 1)
    bins_to_pad = bar_count - (len(samples) % bar_count)
    samples = np.pad(samples, [(0, bins_to_pad)])
    samples = np.reshape(samples, (bar_count, -1))
    samples = np.abs(samples)
    samples = np.max(samples, 1)

    with utils.MatplotlibBackendMananger():
        plt.clf()
        # Plot waveform
        color = (
            bars_color
            if isinstance(bars_color, str)
            else get_color_gradient(bars_color[0], bars_color[1], bar_count)
        )
        plt.bar(
            np.arange(0, bar_count),
            samples * 2,
            bottom=(-1 * samples),
            width=bar_width,
            color=color,
        )
        plt.axis("off")
        plt.margins(x=0)
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        savefig_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if bg_image is not None:
            savefig_kwargs["transparent"] = True
        else:
            savefig_kwargs["facecolor"] = bg_color
        plt.savefig(tmp_img.name, **savefig_kwargs)
        waveform_img = PIL.Image.open(tmp_img.name)
        waveform_img = waveform_img.resize((1000, 200))

        # Composite waveform with background image
        if bg_image is not None:
            waveform_array = np.array(waveform_img)
            waveform_array[:, :, 3] = waveform_array[:, :, 3] * fg_alpha
            waveform_img = PIL.Image.fromarray(waveform_array)

            bg_img = PIL.Image.open(bg_image)
            waveform_width, waveform_height = waveform_img.size
            bg_width, bg_height = bg_img.size
            if waveform_width != bg_width:
                bg_img = bg_img.resize(
                    (waveform_width, 2 * int(bg_height * waveform_width / bg_width / 2))
                )
                bg_width, bg_height = bg_img.size
            composite_height = max(bg_height, waveform_height)
            composite = PIL.Image.new(
                "RGBA", (waveform_width, composite_height), "#FFFFFF"
            )
            composite.paste(bg_img, (0, composite_height - bg_height))
            composite.paste(
                waveform_img, (0, composite_height - waveform_height), waveform_img
            )
            composite.save(tmp_img.name)
            img_width, img_height = composite.size
        else:
            img_width, img_height = waveform_img.size
            waveform_img.save(tmp_img.name)

    # Convert waveform to video with ffmpeg
    output_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    ffmpeg_cmd = f"""ffmpeg -loop 1 -i {tmp_img.name} -i {audio_file} -vf "color=c=#FFFFFF77:s={img_width}x{img_height}[bar];[0][bar]overlay=-w+(w/{duration})*t:H-h:shortest=1" -t {duration} -y {output_mp4.name}"""

    subprocess.call(ffmpeg_cmd, shell=True)
    return output_mp4.name
