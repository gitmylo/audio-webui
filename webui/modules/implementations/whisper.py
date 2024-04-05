import gc
import os.path
from tempfile import NamedTemporaryFile
from typing import Union

import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutomaticSpeechRecognitionPipeline
from gradio_client.client import DEFAULT_TEMP_DIR

import model_manager

processor: WhisperProcessor = None
model: Union[WhisperForConditionalGeneration, AutomaticSpeechRecognitionPipeline] = None
device: str = None
loaded_model: str = None


def get_official_models():
    models = whisper._MODELS
    return models.keys()


def unload():
    global model, processor, device, loaded_model
    model = None
    processor = None
    device = None
    loaded_model = None
    gc.collect()
    torch.cuda.empty_cache()
    return 'Unloaded'


def load(pretrained_model='openai/whisper-base', map_device='cuda' if torch.cuda.is_available() else 'cpu'):
    global model, processor, device, loaded_model
    try:
        if loaded_model != pretrained_model:
            unload()
            print(f'Loading {pretrained_model}')
            whisper_models = whisper._MODELS
            official_models = get_official_models()
            if pretrained_model in official_models:
                model_url = whisper_models[pretrained_model]
                model_name = os.path.basename(model_url)
                model_path = model_manager.get_model_path(model_url, model_type="whisper", single_file=True, single_file_name=model_name)
                model = whisper.load_model(model_path, map_device)
                loaded_model = pretrained_model
                device = map_device
            else:
                raise Exception(f'Model {pretrained_model} not found; available models = {get_official_models()}')
        return f'Loaded {pretrained_model}'
    except Exception as e:
        unload()
        return f'Failed to load, {e}'


def transcribe(wav, files) -> tuple[tuple[int, torch.Tensor], list[str]]:
    return transcribe_wav(wav), transcribe_files(files)


def transcribe_wav(wav):
    global model, processor, device, loaded_model
    if loaded_model is not None:
        if wav is None:
            return None
        sr, wav = wav
        import traceback
        try:
            if sr != 16000:
                import torchaudio.functional as F
                wav = F.resample((torch.tensor(wav).to(device).float() / 32767.0).mean(-1).squeeze().unsqueeze(0), sr, 16000).flatten().cpu().detach().numpy()
                sr = 16000
            return whisper.transcribe(model, wav)['text'].strip()
        except Exception as e:
            traceback.print_exception(e)
            return f'Exception: {e}'
    else:
        return 'No model loaded! Please load a model.'


def transcribe_files(files: list) -> list[str]:
    if files is None or len(files) == 0:
        return []
    out_list = []
    global model, processor, device, loaded_model
    if loaded_model is not None:
        for f in files:
            filename = os.path.basename(f.name)
            print('Processing ', filename)
            filename_noext, fileext = os.path.splitext(filename)
            out_file = NamedTemporaryFile(dir=DEFAULT_TEMP_DIR, mode='w', delete=False, suffix='.txt', prefix=filename_noext, encoding='utf8')

            out_file.write(whisper.transcribe(model, f.name)['text'].strip())

            out_list.append(out_file.name)
        return out_list
    else:
        return []
