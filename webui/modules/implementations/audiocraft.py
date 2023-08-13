import gc

import gradio
import torch
from audiocraft.models import MusicGen
from audiocraft.models import AudioGen

model: MusicGen = None
loaded = False
used_model = ''
device: str = None

melody_models = ['facebook/musicgen-melody']
audiogen_models = ['facebook/audiogen-medium']
models = ['facebook/musicgen-small', 'facebook/musicgen-medium', 'facebook/musicgen-large'] + melody_models + audiogen_models


def supports_melody():
    return used_model in melody_models


def create_model(pretrained='medium', map_device='cuda' if torch.cuda.is_available() else 'cpu'):
    if is_loaded():
        delete_model()
    global model, loaded, device, used_model
    try:
        model = MusicGen.get_pretrained(pretrained, device=map_device) if pretrained not in audiogen_models else AudioGen.get_pretrained(pretrained, device=map_device)
        device = map_device
        used_model = pretrained
        loaded = True
    except:
        raise gradio.Error('Could not load model!')


def delete_model():
    global model, loaded, device
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        loaded = False
        device = None
    except:
        raise gradio.Error('Could not unload model!')


def is_loaded():
    return loaded


def generate(prompt='', input_audio=None, use_sample=True, top_k=250, top_p=0.0, temp=1, duration=8, cfg_coef=3, progress=gradio.Progress()):
    if is_loaded():
        model.set_generation_params(use_sample, top_k, top_p, temp, duration, cfg_coef)
        progress(0, desc='Generating')

        def progress_callback(p, t):
            progress((p, t), desc='Generating')

        model.set_custom_progress_callback(progress_callback)


        input_audio_not_none = input_audio is not None

        sr, wav = 0, None

        if input_audio_not_none:
            sr, wav = input_audio
            wav = torch.tensor(wav)
            if wav.dtype == torch.int16:
                wav = (wav.float() / 32767.0)
            if wav.dim() == 2 and wav.shape[1] == 2:
                wav = wav.mean(dim=1)

        if input_audio_not_none and supports_melody():
            wav = model.generate_with_chroma([prompt if prompt else None], wav[None].expand(1, -1, -1), sr, True)
        elif input_audio_not_none:
            model.set_generation_params(use_sample, top_k, top_p, temp, duration, cfg_coef)
            wav = model.generate_continuation(wav[None].expand(1, -1, -1), sr, [prompt if prompt else None], True)
        elif not prompt:
            wav = model.generate_unconditional(1, True)
        else:
            wav = model.generate([prompt], True)

        wav = wav.cpu().flatten().numpy()
        return model.sample_rate, wav
    raise gradio.Error('No model loaded! Please load a model first.')
