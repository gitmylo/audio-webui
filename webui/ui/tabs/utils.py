import gradio
import torch
import webui.ui.tabs.rvc as rvc


def denoise_tab():
    with gradio.Row():
        audio_in = gradio.Audio(label='Input audio')
        audio_out = gradio.Audio(label='Denoised audio')
    denoise_button = gradio.Button('Denoise')

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

    split_button = gradio.Button('Split')
    split_button.click(fn=music_split_func, inputs=audio_in, outputs=[audio_vocal, audio_background])


def utils_tab():
    with gradio.Tabs():
        with gradio.Tab('denoise'):
            denoise_tab()
        with gradio.Tab('music splitting'):
            music_split_tab()
