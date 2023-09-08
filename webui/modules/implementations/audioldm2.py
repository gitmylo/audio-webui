import gc
import os.path

import diffusers
import torch.cuda
import transformers
import librosa

model: diffusers.AudioLDM2Pipeline = None
loaded = False
device: str = None

models = ['cvssp/audioldm2', 'cvssp/audioldm2-large', 'cvssp/audioldm2-music']


def create_model(pretrained='cvssp/audioldm2', map_device='cuda' if torch.cuda.is_available() else 'cpu'):
    if is_loaded():
        delete_model()
    global model, loaded, device
    try:
        cache_dir = os.path.join('data', 'models', 'audioldm')
        model = diffusers.AudioLDM2Pipeline.from_pretrained(pretrained, cache_dir=cache_dir).to(map_device)
        device = map_device
        loaded = True
    except:
        pass


def is_loaded():
    return loaded


def delete_model():
    global model, loaded, clap_model, processor, device
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        loaded = False
        device = None
    except:
        pass


def generate(prompt='', negative_prompt='', steps=10, duration=5.0, cfg=2.5, seed=-1, callback=None):
    if is_loaded():
        try:
            sample_rate = 16000
            seed = seed if seed >= 0 else torch.seed()
            torch.manual_seed(seed)
            output = model(prompt=prompt, negative_prompt=negative_prompt, audio_length_in_s=duration, num_inference_steps=steps, guidance_scale=cfg, callback=callback)
            waveforms = output.audios
            waveform = waveforms[0]

            return seed, (sample_rate, waveform)
        except Exception as e:
            return f'An exception occurred: {str(e)}'
    return 'No model loaded! Please load a model first.'
