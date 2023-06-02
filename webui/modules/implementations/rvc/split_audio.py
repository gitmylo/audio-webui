import os.path

import torch


def _split(sr, audio):
    import scipy.io.wavfile
    import librosa

    scipy.io.wavfile.write('speakeraudio.wav', sr, audio.detach().cpu().numpy())

    audio, sr = librosa.load('speakeraudio.wav', sr=16000)

    # Code source: Brian McFee
    # License: ISC

    ##################
    # Standard imports
    import numpy as np
    import matplotlib.pyplot as plt

    import librosa.display

    S_full, phase = librosa.magphase(librosa.stft(audio))

    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    # S_full_audio = librosa.istft(S_full*phase)
    S_foreground_audio = librosa.istft(S_foreground*phase)
    S_background_audio = librosa.istft(S_background*phase)

    return S_foreground_audio, S_background_audio, sr


def split(sr, audio):
    import scipy.io.wavfile
    scipy.io.wavfile.write('speakeraudio.wav', sr, audio.detach().cpu().numpy())
    # import torchaudio
    # torchaudio.save('speakeraudio.wav', audio.abs().unsqueeze(0), sr)

    import demucs.separate
    import shlex
    # model_name = 'htdemucs'
    model_name = 'htdemucs_6s'
    # model_name = 'mdx_extra_q'
    args = shlex.split(f'speakeraudio.wav -n {model_name} --two-stems vocals --filename {{stem}}.{{ext}} --float32')
    demucs.separate.main(args)

    # audio_other_files = [os.path.join('separated', model_name, f+'.wav') for f in ['bass', 'drums', 'other', 'piano', 'guitar'] if os.path.isfile(os.path.join('separated', model_name, f+'.wav'))]
    # audio_other_files = [os.path.join('separated', model_name, f+'.wav') for f in ['bass', 'other', 'piano', 'guitar'] if os.path.isfile(os.path.join('separated', model_name, f+'.wav'))]
    audio_vocals_file = os.path.join('separated', model_name, 'vocals.wav')
    other_file = os.path.join('separated', model_name, 'no_vocals.wav')

    import torchaudio

    vocals, sr = torchaudio.load(audio_vocals_file)
    additional, _ = torchaudio.load(other_file)

    return vocals, additional, sr


# def split(sr, audio):
#     return audio, audio, sr
