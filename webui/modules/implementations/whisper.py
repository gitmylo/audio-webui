import gc
import os.path
from tempfile import NamedTemporaryFile

import torch
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, AutomaticSpeechRecognitionPipeline

processor: WhisperProcessor = None
model: WhisperForConditionalGeneration | AutomaticSpeechRecognitionPipeline = None
device: str = None
loaded_model: str = None


def get_official_models():
    # return [
    #     'openai/whisper-tiny.en',
    #     'openai/whisper-small.en',
    #     'openai/whisper-base.en',
    #     'openai/whisper-medium.en',
    #     'openai/whisper-tiny',
    #     'openai/whisper-small',
    #     'openai/whisper-base',
    #     'openai/whisper-medium',
    #     'openai/whisper-large',
    #     'openai/whisper-large-v2'
    # ]
    return [
        'tiny.en',
        'small.en',
        'base.en',
        'medium.en',
        'tiny',
        'small',
        'base',
        'medium',
        'large',
        'large-v2'
    ]


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
    print('Loading map_device=', map_device, flush=True)
    try:
        if loaded_model != pretrained_model:
            unload()
            print('Loading pretrained_model=', pretrained_model, flush=True)
            # model = pipeline('automatic-speech-recognition', pretrained_model, device=map_device, model_kwargs={'cache_dir': 'models/automatic-speech-recognition'})
            model = whisper.load_model(pretrained_model, map_device, 'data/models/automatic-speech-recognition/whisper')
            print('Loaded pretrained_model=', pretrained_model, flush=True)
            loaded_model = pretrained_model
            device = map_device
        else:
            print('Already loaded loaded_model=', loaded_model, flush=True)
        return f'Loaded {pretrained_model}'
    except Exception as e:
        unload()
        print('Loading failed', e, flush=True)
        return f'Failed to load, {e}'


def transcribe(wav, files) -> tuple[tuple[int, torch.Tensor], list[str]]:
    return transcribe_wav(wav), transcribe_files(files)


def transcribe_wav(wav):
    global model, processor, device, loaded_model
    if loaded_model is not None:
        if wav is None:
            print('Processing failed wav=', wav, flush=True)
            return None
        sr, wav = wav
        import traceback
        try:
            if sr != 16000:
                import torchaudio.functional as F
                import numpy as np
                print('Pre-processing wav ...', sr, wav.max(), wav.min(), flush=True)
                wav = np.frombuffer(wav, np.int16).flatten().astype(np.float32) / 32768.0
                wav = F.resample(torch.from_numpy(wav), sr, 16000)
                sr = 16000
                print('Pre-processed', wav.max(), wav.min(), len(wav), flush=True)
            transcribed = whisper.transcribe(model, wav)
            print(transcribed, flush=True)
            return transcribed['text'].strip()
        except Exception as e:
            traceback.print_exception(e)
            print('Processing failed', e, flush=True)
            return f'Exception: {e}'
    else:
        print('No model loaded! Please load a model.', flush=True)
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
            out_file = NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix=filename_noext, encoding='utf8')

            out_file.write(whisper.transcribe(model, f.name)['text'].strip())

            out_list.append(out_file.name)
        return out_list
    else:
        return []
