import gc

import numpy as np
import scipy.io.wavfile
import torch.cuda
import torchaudio
from TTS.api import TTS
import gradio

from webui.modules.download import fill_models

flag_strings = ['denoise', 'separate music']

tts_model = None
tts_model_name = None


def get_models_installed():
    return fill_models('rvc')


def unload_rvc():
    import webui.modules.implementations.rvc.rvc as rvc
    rvc.unload_rvc()
    return gradio.update(selected='')


def load_rvc(model):
    if not model:
        return unload_rvc()
    import webui.modules.implementations.rvc.rvc as rvc
    rvc.load_rvc(model)
    return gradio.update()


def gen(rvc_model_selected, pitch_extract, tts, text_in, audio_in, flag):
    if not audio_in:
        global tts_model, tts_model_name
        if tts_model_name != tts:
            if tts_model is not None:
                tts_model = None
                gc.collect()
                torch.cuda.empty_cache()

            tts_model_name = tts
            print('Loading TTS model')
            tts_model = TTS(tts)
        audio_in, sr = torch.tensor(tts_model.tts(text_in)), tts_model.synthesizer.output_sample_rate
    else:
        sr, audio_in = audio_in
        audio_in = torch.tensor(audio_in)
    audio_tuple = (sr, audio_in)

    if len(audio_tuple[1].shape) == 2:
        audio_tuple = (audio_tuple[0], audio_tuple[1][:, 1].squeeze().unsqueeze(0))
    else:
        audio_tuple = (audio_tuple[0], audio_tuple[1].unsqueeze(0))

    if 'separate music' in flag:
        pass
    if 'denoise' in flag:
        if len(audio_tuple[1].shape) == 1:
            audio_tuple = (audio_tuple[0], audio_tuple[1].unsqueeze(0))
        torchaudio.save('speakeraudio.wav', audio_tuple[1], audio_tuple[0])  # Workaround
        audio_tuple = (audio_tuple[0], scipy.io.wavfile.read('speakeraudio.wav')[1])
        import noisereduce.noisereduce as noisereduce
        audio_tuple = (audio_tuple[0], torch.tensor(noisereduce.reduce_noise(y=audio_tuple[1], sr=audio_tuple[0])))

    if rvc_model_selected:
        if len(audio_tuple[1].shape) == 1:
            audio_tuple = (audio_tuple[0], audio_tuple[1].unsqueeze(0))
        torchaudio.save('speakeraudio.wav', audio_tuple[1], audio_tuple[0])

        import webui.modules.implementations.rvc.rvc as rvc
        out1, out2 = rvc.vc_single(0, 'speakeraudio.wav', 0, None, pitch_extract, rvc_model_selected, None, 0.88, 3, 0, 1, 0.33)

        audio_tuple = out2

    if torch.is_tensor(audio_tuple[1]):
        audio_tuple = (audio_tuple[0], audio_tuple[1].flatten().detach().cpu().numpy())

    return [audio_tuple, gradio.make_waveform(audio_tuple)]




def rvc():
    all_tts = TTS.list_models()
    with gradio.Row():
        with gradio.Column():
            with gradio.Accordion('TTS', open=False):
                selected_tts = gradio.Dropdown(all_tts, label='TTS model', info='The TTS model to use for text-to-speech')
                text_input = gradio.TextArea(label='Text to speech text', info='Text to speech text if no audio file is used as input.')
            with gradio.Accordion('Audio input', open=False):
                audio_input = gradio.Audio(label='Audio input')
            with gradio.Accordion('RVC'):
                with gradio.Row():
                    selected = gradio.Dropdown(get_models_installed(), label='RVC Model')
                    with gradio.Column(elem_classes='smallsplit'):
                        refresh = gradio.Button('🔃', variant='tool secondary')
                        unload = gradio.Button('💣', variant='tool primary')
                    refresh.click(fn=get_models_installed, outputs=selected, show_progress=True)
                    unload.click(fn=unload_rvc, outputs=selected, show_progress=True)
                    selected.select(fn=load_rvc, inputs=selected, outputs=selected, show_progress=True)
            pitch_extract = gradio.Radio(choices=["pm", "harvest", "crepe"], label='Pitch extraction', value='pm', interactive=True)
            flags = gradio.Dropdown(flag_strings, label='Flags', info='Things to apply on the audio input/output', multiselect=True)
        with gradio.Column():
            generate = gradio.Button('Generate')
            audio_out = gradio.Audio()
            video_out = gradio.Video()

        generate.click(fn=gen, inputs=[selected, pitch_extract, selected_tts, text_input, audio_input, flags], outputs=[audio_out, video_out])
