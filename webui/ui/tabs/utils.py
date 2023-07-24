import os.path
import time
from tempfile import NamedTemporaryFile

import gradio
import numpy
import soundfile
import torch
import torchaudio
import torchaudio.functional as F

import librosa

import webui.ui.tabs.rvc as rvc
from webui.modules.implementations.ffmpeg_utils import ffmpeg_utils_tab


def denoise_tab():
    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio')
        audio_out = gradio.Audio(label='Denoised audio', interactive=False)
    denoise_button = gradio.Button('Denoise', variant='primary')

    def denoise_func(audio):
        sr, wav = audio
        import noisereduce.noisereduce as noisereduce
        wav = noisereduce.reduce_noise(wav, sr)
        return sr, wav

    with gradio.Row():
        with gradio.Column():
            in_directory = gradio.Textbox(label='Input directory')
            out_directory = gradio.Textbox(label='Output directory')
        batch_progress = gradio.Textbox(label='Batch processing progress')
    denoise_batch = gradio.Button('Denoise batch', variant='primary')

    def batch_denoise(in_dir, out_dir):
        import noisereduce.noisereduce as noisereduce
        if not os.path.isdir(in_dir):
            yield 'Error: input directory is not a directory'
            return
        os.makedirs(out_dir, exist_ok=True)
        output = f'Processing directory {in_dir}'
        yield output
        for f in os.listdir(in_dir):
            if os.path.splitext(f)[-1] not in ['.wav', '.mp3']:
                continue
            output += f'\nProcessing {f}'
            yield output
            full_path = os.path.join(in_dir, f)
            wav, sr = torchaudio.load(full_path)
            wav = wav.detach().cpu().numpy()
            wav = noisereduce.reduce_noise(wav, sr)
            wav = torch.tensor(wav)
            torchaudio.save(os.path.join(out_dir, f), wav, sr)
        output += '\nCompleted!'
        yield output

    denoise_button.click(fn=denoise_func, inputs=audio_in, outputs=audio_out)
    denoise_batch.click(fn=batch_denoise, inputs=[in_directory, out_directory], outputs=batch_progress)



def music_split_tab():
    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio')
        with gradio.Column():
            audio_vocal = gradio.Audio(label='Vocals', interactive=False)
            audio_background = gradio.Audio(label='Other audio', interactive=False)

    def music_split_func(audio):
        sr, wav = audio
        wav = torch.tensor(wav).float() / 32767.0
        if wav.shape[0] == 2:
            wav = wav.mean(0)
        import webui.modules.implementations.rvc.split_audio as split_audio
        vocal, background, sr = split_audio.split(sr, wav)
        if vocal.shape[0] == 2:
            vocal = vocal.mean(0)
        if background.shape[0] == 2:
            background = background.mean(0)
        return [(sr, vocal.squeeze().detach().numpy()), (sr, background.squeeze().detach().numpy())]

    split_button = gradio.Button('Split', variant='primary')
    split_button.click(fn=music_split_func, inputs=audio_in, outputs=[audio_vocal, audio_background])

    with gradio.Row():
        with gradio.Column():
            in_directory = gradio.Textbox(label='Input directory')
            out_directory = gradio.Textbox(label='Output directory')
        batch_progress = gradio.Textbox(label='Batch processing progress')
    split_batch = gradio.Button('Split batch', variant='primary')

    def batch_music_split(in_dir, out_dir):
        if not os.path.isdir(in_dir):
            yield 'Error: input directory is not a directory'
            return
        os.makedirs(os.path.join(out_dir, 'vocal'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'background'), exist_ok=True)
        output = f'Processing directory {in_dir}'
        yield output
        for f in os.listdir(in_dir):
            split = os.path.splitext(f)
            extension = split[-1]
            if extension not in ['.wav', '.mp3']:
                continue
            output += f'\nProcessing {f}'
            yield output
            full_path = os.path.join(in_dir, f)
            wav, sr = torchaudio.load(full_path)
            # Split
            if wav.dtype == numpy.int16:
                wav = wav.float() / 32767.0
            if wav.shape[0] == 2:
                wav = wav.mean(0)
            import webui.modules.implementations.rvc.split_audio as split_audio
            vocal, background, sr = split_audio.split(sr, wav)
            if vocal.shape[0] == 2:
                vocal = vocal.mean(0)
            if background.shape[0] == 2:
                background = background.mean(0)
            if len(vocal.shape) == 1:
                vocal = vocal.unsqueeze(0)
            if len(background.shape) == 1:
                background = background.unsqueeze(0)

            torchaudio.save(os.path.join(out_dir, 'vocal', f), vocal, sr)
            torchaudio.save(os.path.join(out_dir, 'background', f), background, sr)
        output += '\nCompleted!'
        yield output

    split_batch.click(fn=batch_music_split, inputs=[in_directory, out_directory], outputs=batch_progress)

    with gradio.Row():
        with gradio.Column():
            # audio_combine_1 = gradio.Audio(label='Input audio 1', type='filepath')
            audio_combine_1 = gradio.File(label='Input audio 1')
            # audio_combine_2 = gradio.Audio(label='Input audio 2', type='filepath')
            audio_combine_2 = gradio.File(label='Input audio 2')
        audio_out = gradio.Audio(label='Combined audio', interactive=False)

    def music_merge_func(audio1, audio2):
        x, sr = torchaudio.load(audio1.name)
        y, sry = torchaudio.load(audio2.name)

        if x.shape[0] == 2:
            x = x.mean(0)
        if y.shape[0] == 2:
            y = y.mean(0)
        if x.shape[-1] == 2:
            x = x.mean(-1)
        if y.shape[-1] == 2:
            y = y.mean(-1)

        len_x = x.shape[-1] / sr
        len_y = y.shape[-1] / sry

        y = F.resample(y, sry, sr)
        y = F.resample(y, sr, int(sr * len_x/len_y))
        y = y.flatten()
        x = x.flatten()
        if x.shape[0] > y.shape[0]:
            x = x[-y.shape[0]:]
        else:
            y = y[-x.shape[0]:]
        return sr, x.add(y).flatten().cpu().detach().numpy()

    split_button = gradio.Button('Merge', variant='primary')
    split_button.click(fn=music_merge_func, inputs=[audio_combine_1, audio_combine_2], outputs=audio_out)


def audio_download_tab():
    import webui.modules.implementations.audio_download as ad
    with gradio.Row():
        with gradio.Column():
            url_type = gradio.Dropdown(['youtube'], value='youtube', label='Type')
            url = gradio.Textbox(max_lines=1, label='Url')
        file_out = gradio.File(label='Downloaded audio')
    download_button = gradio.Button('Download', variant='primary')
    download_button.click(fn=ad.download_audio, inputs=[url_type, url], outputs=file_out)


def waveform_tab():
    def create_waveform(audio):
        import webui.modules.util as u
        return u.make_waveform(audio)

    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio')
        video_out = gradio.PlayableVideo(label='Output waveform video', interactive=False)

    create_waveform_button = gradio.Button('Create waveform video', variant='primary')
    create_waveform_button.click(fn=create_waveform, inputs=audio_in, outputs=video_out)


def enhance_tab():
    def enhance_audio(file_in):
        output_file = NamedTemporaryFile(suffix='.wav', delete=False).name

        sampling_rate = 44100
        y, sr = librosa.load(file_in, sr=sampling_rate)

        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=12, res_type="soxr_vhq")
        soundfile.write(output_file, y + y_shifted, sampling_rate)
        return output_file

    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio', type='filepath')
        audio_out = gradio.Audio(label='Output audio', interactive=False)

    create_waveform_button = gradio.Button('Enhance audio quality (Mainly for audioldm).', variant='primary')
    create_waveform_button.click(fn=enhance_audio, inputs=audio_in, outputs=audio_out)


def utils_tab():
    with gradio.Tabs():
        with gradio.Tab('🧹 Denoise'):
            denoise_tab()
        with gradio.Tab('🔊▶🗣/🎵 Music splitting'):
            music_split_tab()
        with gradio.Tab('🔎 Enhance'):
            enhance_tab()
        with gradio.Tab('📈 Audio waveforms'):
            waveform_tab()
        with gradio.Tab('👽 FFMPEG'):
            ffmpeg_utils_tab()
        with gradio.Tab('🔽 Audio downloads'):
            audio_download_tab()
        import webui.extensionlib.callbacks as cb
        cb.get_manager('webui.tabs.utils')()
