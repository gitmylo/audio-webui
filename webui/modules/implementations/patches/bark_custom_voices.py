import torch
import torchaudio
from bark.generation import SAMPLE_RATE, _grab_best_device, OFFLOAD_CPU, clean_models, models, models_devices
from encodec.utils import convert_audio

import model_manager
from hubert.customtokenizer import CustomTokenizer
from hubert.pre_kmeans_hubert import CustomHubert
from webui.modules.implementations.patches.bark_generation import generate_text_semantic_new, generate_coarse_new, \
    generate_fine_new, encodec_load_codec_model
from webui.ui.tabs import settings


def generate_semantic_fine(
        transcript='There actually isn\'t a way to do that. It\'s impossible. Please don\'t even bother.'):
    """
    Creates a speech file with semantics and fine audio
    :param transcript: The transcript.
    :return: tuple with (semantic, fine)
    """
    semantic = generate_text_semantic_new(transcript)  # We need speech patterns
    coarse = generate_coarse_new(semantic)  # Voice doesn't matter
    fine = generate_fine_new(coarse)  # Good audio, ready for what comes next
    return semantic, fine


huberts = {}


def load_hubert(clone_model):
    global huberts
    hubert_path = model_manager.get_model_path(model_url='https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', model_type='hubert', single_file=True, single_file_name='hubert_base_ls960.pt', save_file_name='hubert.pt')

    tokenizer_path = model_manager.get_model_path(model_url=clone_model['repo'], model_type='hubert', single_file=True,
                                                  single_file_name=clone_model['file'], save_file_name=clone_model['dlfilename'])
    if 'hubert' not in huberts:
        print('Loading HuBERT')
        huberts['hubert'] = CustomHubert(hubert_path)
    if 'tokenizer' not in huberts or (
            'tokenizer_name' in huberts and huberts['tokenizer_name'] != clone_model['name'].casefold()):
        print('Loading Custom Tokenizer')
        tokenizer = CustomTokenizer.load_from_checkpoint(tokenizer_path, map_location=torch.device('cpu'))
        huberts['tokenizer'] = tokenizer
        huberts['tokenizer_name'] = clone_model['name'].casefold()


def wav_to_semantics(file, clone_model) -> torch.Tensor:
    # Vocab size is 10,000.

    load_hubert(clone_model)

    wav, sr = torchaudio.load(file)
    # sr, wav = wavfile.read(file)
    # wav = torch.tensor(wav, dtype=torch.float32)

    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)
    if wav.shape[1] == 2:
        wav = wav.mean(1, keepdim=False).unsqueeze(-1)

    # Extract semantics in HuBERT style
    print('Extracting semantics')
    semantics = huberts['hubert'].forward(wav, input_sample_hz=sr)
    print('Tokenizing semantics')
    tokens = huberts['tokenizer'].get_token(semantics)
    return tokens


def eval_semantics(code):
    """
    BE CAREFUL, this will execute :code:
    :param code: The code to evaluate, out local will be used for the output.
    :return: The created numpy array.
    """
    _locals = locals()
    exec(code, globals(), _locals)
    return _locals['out']


def generate_course_history(fine_history):
    return fine_history[:2, :]


def load_codec_model(use_gpu=True, force_reload=False):
    device = _grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "codec"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = encodec_load_codec_model(device)
        models[model_key] = model
    models[model_key].to(device)
    return models[model_key]


def generate_fine_from_wav(file):
    model = load_codec_model(
        use_gpu=not settings.get('bark_use_cpu'))  # Don't worry about reimporting, it stores the loaded model in a dict
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, SAMPLE_RATE, model.channels)
    wav = wav.unsqueeze(0)
    if not settings.get('bark_use_cpu'):
        wav = wav.to('cuda')
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    codes = codes.cpu().numpy()

    return codes
