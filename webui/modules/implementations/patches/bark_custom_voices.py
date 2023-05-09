import os
import tempfile
from pathlib import Path

import numpy
import torch
import torchaudio
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from bark.generation import SAMPLE_RATE, load_codec_model
from scipy.io.wavfile import write as write_wav

from scripts.bark_speaker_info import codec_decode
from webui.modules.implementations.patches.bark_generation import generate_text_semantic_new, generate_coarse_new, generate_fine_new
from encodec.utils import convert_audio
from webui.args import args
from webui.modules.implementations.patches.denoise import enhance_new
from audiolm_pytorch import HubertWithKmeans


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
    model_name = 'tts_models/multilingual/multi-dataset/your_tts'
    language_idx = 'en'
    gpu = not args.tts_use_cpu

    speaker_wav = voice_to_clone
    reference_wav = fine_file.name

    speakers_file_path = None
    language_ids_file_path = None
    encoder_path = None
    encoder_config_path = None
    vocoder_path = None
    vocoder_config_path = None

    import TTS.api
    path = Path(TTS.api.__file__).parent / ".models.json"
    manager = ModelManager(path, progress_bar=True)
    model_path, config_path, model_item = manager.download_model(model_name)

    tts_path = model_path
    tts_config_path = config_path
    if "default_vocoder" in model_item:
        vocoder_name = model_item["default_vocoder"]

    if vocoder_name is not None and not args.vocoder_path:
        vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

    synthesizer = Synthesizer(
        tts_checkpoint=tts_path,
        tts_config_path=tts_config_path,
        tts_speakers_file=speakers_file_path,
        tts_languages_file=language_ids_file_path,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        vc_checkpoint=encoder_path,
        vc_config=encoder_config_path,
        use_cuda=gpu
    )

    wav = synthesizer.tts(  # Why does it use TTS for voice conversion on vits?
        language_name=language_idx,
        speaker_wav=speaker_wav,
        reference_wav=reference_wav
    )

    synthesizer.save_wav(wav, output.name)
    # tts = TTS('voice_conversion_models/multilingual/vctk/freevc24', gpu=not args.tts_use_cpu)
    # tts.voice_conversion_to_file(
    #     source_wav=voice_to_clone,
    #     target_wav=fine_file.name,
    #     file_path=output.name
    # )

    class Args:
        pass

    # Denoise
    enhance_args = Args  # Not even instantiating because it really doesn't matter here

    enhance_args.streaming = False  # No streaming
    enhance_args.dry = 1  # Full denoising

    enhance_args.dns64 = False
    enhance_args.master64 = True  # Model
    enhance_args.valentini_nc = False
    enhance_args.model_path = ''  # Don't load model
    enhance_args.device = 'cuda' if not args.tts_use_cpu else 'cpu'

    enhance_args.out_file = output.name[:-4] + '_enhanced' + '.wav'

    enhance_new(enhance_args, output.name, enhance_args.out_file)

    # Convert the new fine and new coarse prompts
    new_fine, _ = generate_fine_from_wav(enhance_args.out_file)
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


def wav_to_semantics(file) -> torch.Tensor:  #TODO: find right model
    wav2vec = HubertWithKmeans(
        checkpoint_path='../data/models/hubert/hubert_base_ls960.pt',
        kmeans_path='../data/models/hubert/hubert_base_ls960_L9_km500.bin'
    )
    wav, sr = torchaudio.load(file)
    return wav2vec.forward(wav, True, sr)


def generate_course_history(fine_history):
    return fine_history[:2, :]


def generate_fine_from_wav(file):
    model = load_codec_model(use_gpu=not args.bark_use_cpu)  # Don't worry about reimporting, it stores the loaded model in a dict
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
