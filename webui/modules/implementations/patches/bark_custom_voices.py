import audio2numpy
import numpy as np
import torch
import torchaudio
from bark.generation import SAMPLE_RATE, load_codec_model
from webui.modules.implementations.patches.bark_generation import generate_text_semantic_new
from encodec.utils import convert_audio


def generate_semantic_history(transcript, duration):
    return generate_text_semantic_new(transcript, max_gen_duration_s=duration)


def generate_course_history(fine_history):
    return fine_history[:2, :]


def generate_fine_history(file):
    model = load_codec_model()  # Don't worry about reimporting, it stores the loaded model in a dict
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, SAMPLE_RATE, model.channels)
    wav = wav.unsqueeze(0).to('cuda')
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    seconds = wav.shape[-1] / model.sample_rate

    codes = codes.cpu().numpy()

    return codes, seconds
