import tempfile

import numpy
import torch
import torchaudio
from bark.generation import SAMPLE_RATE, load_codec_model
from scipy.io.wavfile import write as write_wav

from scripts.bark_speaker_info import codec_decode
from webui.modules.implementations.patches.bark_api import semantic_to_waveform_new
from webui.modules.implementations.patches.bark_generation import generate_text_semantic_new, generate_coarse_new, generate_fine_new
from encodec.utils import convert_audio
from TTS.api import TTS


def patch_speaker_npz(voice_to_clone: str, npz_file: str):
    data = numpy.load(npz_file)
    fine = data['fine_prompt']
    fine_audio_arr = codec_decode(fine)
    fine_file = tempfile.NamedTemporaryFile(delete=False)
    fine_file.name += '.wav'
    write_wav(fine_file.name, SAMPLE_RATE, fine_audio_arr)  # Convert the fine prompt into audio and store in temp file

    output = tempfile.NamedTemporaryFile(delete=False)
    output.name += '.wav'

    # Convert speaker
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=True,
              force_voice_convert=True)  # Monkeypatched flag to allow voice convert on your_tts
    tts.manager.output_prefix = 'data/models/coqui-tts/'  # TODO: force downloading into this dir
    tts.voice_conversion_to_file(source_wav=voice_to_clone, target_wav=fine_file.name, file_path=output.name)

    # Convert the new fine and new coarse prompts
    new_fine, _ = generate_fine_from_wav(output.name)
    new_coarse = generate_course_history(new_fine)
    return {
        'semantic_prompt': data['semantic_prompt'],
        'fine_prompt': new_fine,
        'coarse_prompt': new_coarse
    }


def generate_semantic_fine(transcript='There actually isn\'t a way to do that. It\'s impossible. Please don\'t even bother.'):
    """
    Creates a speech file with semantics and fine audio
    :param transcript: The transcript.
    :return: tuple with (semantic, fine)
    """
    semantic = generate_text_semantic_new(transcript)  # We need speech patterns
    coarse = generate_coarse_new(semantic)  # Voice doesn't matter
    fine = generate_fine_new(coarse)  # Good audio, ready for what comes next
    return semantic, fine


def generate_course_history(fine_history):
    return fine_history[:2, :]


def generate_fine_from_wav(file):
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


def transfer_speech(voice, sentence):
    pass
