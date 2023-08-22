import gc
import os.path

import diffusers
import torch.cuda
import transformers
import librosa

model: diffusers.AudioLDMPipeline = None
loaded = False
clap_model: transformers.ClapModel = None
processor: transformers.ClapProcessor = None
device: str = None

models = ['cvssp/audioldm', 'cvssp/audioldm-s-full-v2', 'cvssp/audioldm-m-full', 'cvssp/audioldm-l-full']


def create_model(pretrained='cvssp/audioldm-m-full', map_device='cuda' if torch.cuda.is_available() else 'cpu'):
    if is_loaded():
        delete_model()
    global model, loaded, clap_model, processor, device
    try:
        cache_dir = os.path.join('data', 'models', 'audioldm')
        model = diffusers.AudioLDMPipeline.from_pretrained(pretrained, cache_dir=cache_dir).to(map_device)
        clap_model = transformers.ClapModel.from_pretrained("sanchit-gandhi/clap-htsat-unfused-m-full", cache_dir=cache_dir).to(map_device)
        processor = transformers.AutoProcessor.from_pretrained("sanchit-gandhi/clap-htsat-unfused-m-full", cache_dir=cache_dir)
        device = map_device
        loaded = True
    except:
        pass


def delete_model():
    global model, loaded, clap_model, processor, device
    try:
        del model, clap_model, processor
        gc.collect()
        torch.cuda.empty_cache()
        loaded = False
        device = None
    except:
        pass


def is_loaded():
    return loaded


def score_waveforms(text, waveforms):
    inputs = processor(text=text, audios=list(waveforms), return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}
    with torch.no_grad():
        logits_per_text = clap_model(**inputs).logits_per_text  # this is the audio-text similarity score
        probs = logits_per_text.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        most_probable = torch.argmax(probs)  # and now select the most likely audio waveform
    waveform = waveforms[most_probable]
    return waveform


def generate(prompt='', negative_prompt='', steps=10, duration=5.0, cfg=2.5, seed=-1, wav_best_count=1, enhance=False, callback=None):
    if is_loaded():
        try:
            sample_rate = 16000
            seed = seed if seed >= 0 else torch.seed()
            torch.manual_seed(seed)
            output = model(prompt, negative_prompt=negative_prompt if negative_prompt else None,
                           audio_length_in_s=duration, num_inference_steps=steps, guidance_scale=cfg,
                           num_waveforms_per_prompt=wav_best_count, callback=callback)
            waveforms = output.audios
            if waveforms.shape[0] > 1:
                waveform = score_waveforms(prompt, waveforms)
            else:
                waveform = waveforms[0]
            if enhance:  # https://github.com/gitmylo/audio-webui/issues/36#issuecomment-1627380868
                sample_rate = 44100
                audio_resampled = librosa.resample(waveform, orig_sr=16000, target_sr=sample_rate)
                waveform = audio_resampled + librosa.effects.pitch_shift(audio_resampled, sr=sample_rate, n_steps=12, res_type="soxr_vhq")

            return seed, (sample_rate, waveform)
        except Exception as e:
            return f'An exception occurred: {str(e)}'
    return 'No model loaded! Please load a model first.'
