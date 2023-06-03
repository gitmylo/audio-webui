import gradio
import torch
import webui.ui.tabs.rvc as rvc


def denoise_tab():
    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio')
        audio_out = gradio.Audio(label='Denoised audio')
    denoise_button = gradio.Button('Denoise', variant='primary')

    def denoise_func(audio):
        sr, wav = audio
        import noisereduce.noisereduce as noisereduce
        wav = noisereduce.reduce_noise(y=wav, sr=sr)
        return sr, wav

    denoise_button.click(fn=denoise_func, inputs=audio_in, outputs=audio_out)


def music_split_tab():
    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio')
        with gradio.Column():
            audio_vocal = gradio.Audio(label='Vocals')
            audio_background = gradio.Audio(label='Other audio')

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
            # audio_combine_1 = gradio.Audio(label='Input audio 1', type='filepath')
            audio_combine_1 = gradio.File(label='Input audio 1')
            # audio_combine_2 = gradio.Audio(label='Input audio 2', type='filepath')
            audio_combine_2 = gradio.File(label='Input audio 2')
        audio_out = gradio.Audio(label='Combined audio')

    def music_merge_func(audio1, audio2):
        import torchaudio
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

        import torchaudio.functional as F
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


def utils_tab():
    with gradio.Tabs():
        with gradio.Tab('denoise'):
            denoise_tab()
        with gradio.Tab('music splitting'):
            music_split_tab()
        with gradio.Tab('audio downloads'):
            audio_download_tab()
