import os
import shutil

from scipy import signal

import librosa
import numpy as np
from scipy.io import wavfile

from webui.modules.implementations.rvc.slicer2 import Slicer
from webui.ui.tabs.training.training.workspace import Workspace
from webui.modules.implementations.rvc.rvc import load_audio

workspace_path = os.path.join('data', 'training', 'RVC')


class RvcWorkspace(Workspace):
    base_path = 'RVC'

    def create(self, data):
        data_in = base_data.copy()
        if data['vsr'] == 'v1 40k':
            data_in['v'] = 1
            data_in['sr'] = 40000
        elif data['vsr'] == 'v1 48k':
            data_in['v'] = 1
            data_in['sr'] = 48_000
        else:
            data_in['v'] = 2
            data_in['sr'] = 40_000
        return super(RvcWorkspace, self).create(data_in)


def norm_write(tmp_audio, gt_wavs_dir, wavs16k_dir, name, sr, max, alpha):
    tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (max * alpha)) + (
        1 - alpha
    ) * tmp_audio
    wavfile.write(
        "%s/%s.wav" % (gt_wavs_dir, name),
        sr,
        tmp_audio.astype(np.float32),
    )
    tmp_audio = librosa.resample(
        tmp_audio, orig_sr=sr, target_sr=16000
    )  # , res_type="soxr_vhq"
    wavfile.write(
        "%s/%s.wav" % (wavs16k_dir, name),
        16000,
        tmp_audio.astype(np.float32),
    )


def process_dataset():
    if current_workspace is None:
        yield 'No workspace loaded, please load a workspace first.'
        return
    dataset = current_workspace.data['dataset']
    sr = current_workspace.data['sr']
    dir_gt = os.path.join(current_workspace.space_path, '0_gt')
    dir_16k = os.path.join(current_workspace.space_path, '0_16k')

    shutil.rmtree(dir_gt)
    shutil.rmtree(dir_16k)

    os.makedirs(dir_gt, exist_ok=True)
    os.makedirs(dir_16k, exist_ok=True)

    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
    per = 3.7
    overlap = 0.3
    tail = per + overlap
    max = 0.9
    alpha = 0.75

    if not os.path.isdir(dataset):
        yield f'{dataset} does not exist.'
        return
    slicer = Slicer(
        sr=sr,
        threshold=-42,
        min_length=1500,
        min_interval=400,
        hop_size=15,
        max_sil_kept=500
    )
    output = 'Resampling and then splitting audios into chunks.'
    yield output
    i = 0
    for f in os.listdir(dataset):
        split = os.path.splitext(f)
        filename = split[0]
        if split[-1] in ['.wav', '.mp3']:
            pass
        output += f'\nProcessing {f}'
        yield output
        full_path = os.path.join(dataset, f)
        try:
            audio = load_audio(full_path, sr)
            audio = signal.lfilter(bh, ah, audio)

            for audio in slicer.slice(audio):
                j = 0
                while 1:
                    start = int(sr * (per - overlap) * j)
                    j += 1
                    if len(audio[start:]) > tail * sr:
                        tmp_audio = audio[start: start + int(per * sr)]
                        norm_write(tmp_audio, dir_gt, dir_16k, str(i), sr, max, alpha)
                        i += 1
                    else:
                        tmp_audio = audio[start:]
                        i += 1
                        break
                norm_write(tmp_audio, dir_gt, dir_16k, str(i), sr, max, alpha)

        except Exception as e:
            output += f'\nException {e}. Skipping'
            yield output


    output += '\nFinished processing dataset.'
    yield output


def pitch_extract():
    pass



current_workspace: RvcWorkspace = None

base_data = {
    'v': 2,
    'sr': 40_000,
    'f0': 'harvest',
    'dataset': ''
}


def get_workspaces():
    os.makedirs(workspace_path, exist_ok=True)
    return os.listdir(workspace_path)
