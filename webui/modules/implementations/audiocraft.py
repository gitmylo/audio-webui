import gc
import os

import torch
from audiocraft.models import MusicGen

model: MusicGen = None
loaded = False
used_model = ''
device: str = None

melody_models = ['melody']
models = ['small', 'medium', 'large'] + melody_models


def supports_melody():
    return used_model in melody_models


def create_model(pretrained='medium', map_device='cuda' if torch.cuda.is_available() else 'cpu'):
    if is_loaded():
        delete_model()
    global model, loaded, device, used_model
    try:
        model = MusicGen.get_pretrained(pretrained, device=map_device)
        device = map_device
        used_model = pretrained
        loaded = True
    except:
        pass


def delete_model():
    global model, loaded, device
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        loaded = False
        device = None
    except:
        pass


def is_loaded():
    return loaded


def generate(prompt='', input_audio=None, use_sample=True, top_k=250, top_p=0.0, temp=1, duration=8, cfg_coef=3):
    if is_loaded():
        model.set_generation_params(use_sample, top_k, top_p, temp, duration, cfg_coef)

        if input_audio is not None and supports_melody():
            sr, wav = input_audio
            wav = (torch.tensor(wav).float() / 32767.0)
            wav = model.generate_with_chroma([prompt if prompt else None], wav[None].expand(1, -1, -1), sr, True)
        elif not prompt:
            wav = model.generate_unconditional(1, True)
        else:
            wav = model.generate([prompt], True)

        wav = wav.cpu().flatten().numpy()
        return model.sample_rate, wav
    return 'No model loaded! Please load a model first.'
