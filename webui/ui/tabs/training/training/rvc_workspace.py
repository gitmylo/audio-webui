import gc
import os
import shutil

import huggingface_hub
import soundfile
import torch.cuda
import torch.nn.functional as F
from fairseq import checkpoint_utils
from scipy import signal

import librosa
import numpy as np
from scipy.io import wavfile

from hubert import hubert_manager
from webui.args import args
from webui.modules.implementations.rvc.slicer2 import Slicer
from webui.modules.implementations.rvc.custom_pitch_extraction import pitch_extract as pe
from webui.modules.implementations.rvc.rvc import load_audio
from webui.ui.tabs.training.training.workspace import Workspace

workspace_path = os.path.join('data', 'training', 'RVC')


class RvcWorkspace(Workspace):
    base_path = 'RVC'

    def create(self, data):
        data_in = base_data.copy()
        data_in['vsr'] = data['vsr']
        if data['vsr'] == 'v1 40k':
            data_in['v'] = 1
            data_in['sr'] = 40000
        elif data['vsr'] == 'v1 48k':
            data_in['v'] = 1
            data_in['sr'] = 48_000
        else:
            data_in['v'] = 2
            data_in['sr'] = 40_000
        download_base_models(data_in['vsr'])
        return super(RvcWorkspace, self).create(data_in)

    def load(self):
        model = super(RvcWorkspace, self).load()
        download_base_models(model.data['vsr'])
        return model


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


def readwave(wav_path, normalize=False):
    wav, sr = soundfile.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


def process_dataset():
    if current_workspace is None:
        yield 'No workspace loaded, please load a workspace first.'
        return
    dataset = current_workspace.data['dataset']
    sr = current_workspace.data['sr']
    dir_gt = os.path.join(current_workspace.space_path, '0_gt')
    dir_16k = os.path.join(current_workspace.space_path, '0_16k')

    shutil.rmtree(dir_gt, ignore_errors=True)
    shutil.rmtree(dir_16k, ignore_errors=True)

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
            output += f'\nException {e} Skipping'
            yield output


    output += '\nFinished processing dataset.'
    yield output


def coarse_f0(f0, f0_bin, f0_mel_min, f0_mel_max):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (
        f0_bin - 2
    ) / (f0_mel_max - f0_mel_min) + 1

    # use 0 or 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse


def pitch_extract():
    data = current_workspace.data
    space_path = current_workspace.space_path
    f0_method = data['f0']

    sr = 16_000
    hop = 160
    f0_bin = 256
    f0_max = 1100.0
    f0_min = 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    input_dir = os.path.join(space_path, '0_16k')  # %s/1_16k_wavs
    output_f0 = os.path.join(space_path, '1_f0')
    output_f0nsf = os.path.join(space_path, '1_f0nsf')
    output_feat = os.path.join(space_path, '1_feat')  # %s/3_feature256
    time_step = 160 / 16000 * 1000

    shutil.rmtree(output_f0, ignore_errors=True)
    shutil.rmtree(output_f0nsf, ignore_errors=True)
    shutil.rmtree(output_feat, ignore_errors=True)


    os.makedirs(output_f0, exist_ok=True)
    os.makedirs(output_f0nsf, exist_ok=True)
    os.makedirs(output_feat, exist_ok=True)

    output = 'Processing pitch...'
    yield output

    if f0_method != 'none':
        for f in os.listdir(input_dir):
            try:
                output += f'\nExtracting pitch from {f}'
                full_path = os.path.join(input_dir, f)
                npy_name = os.path.splitext(f)[0] + '.npy'
                npy_path_f0 = os.path.join(output_f0, npy_name)
                npy_path_f0nsf = os.path.join(output_f0nsf, npy_name)
                x = load_audio(full_path, sr)
                p_len = x.shape[0] // hop
                f0 = pe(f0_method, x, f0_min, f0_max, p_len, time_step, 16000, hop, 128)
                np.save(npy_path_f0nsf, f0)
                coarse_pitch = coarse_f0(f0, f0_bin, f0_mel_min, f0_mel_max)
                np.save(npy_path_f0, coarse_pitch)
            except Exception as e:
                output += f'\nException: {e} Skipping'
                yield output

    output += '\nLoading HuBERT model...'
    yield output

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_manager.HuBERTManager.make_sure_hubert_rvc_installed()],
        suffix="",
    )

    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    model = models[0]
    model = model.to(device)
    if device != "cpu":
        model = model.half()
    model.eval()

    for idx, file in enumerate(os.listdir(input_dir)):
        try:
            output += f'\nProcessing {file}'
            yield output

            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_feat, os.path.splitext(file)[0] + '.npy')

            if os.path.exists(out_path):
                continue

            features = readwave(in_path, normalize=saved_cfg.task.normalize)

            padding_mask = torch.BoolTensor(features.shape).fill_(False)
            inputs = {
                "source": features.half().to(device)
                if device not in ["mps", "cpu"]
                else features.to(device),
                "padding_mask": padding_mask.to(device),
                "output_layer": 9 if features == "v1" else 12,  # layer 9
            }

            with torch.no_grad():
                logits = model.extract_features(**inputs)
                feats = (
                    model.final_proj(logits[0]) if data['v'] == 1 else logits[0]
                )

            feats = feats.squeeze(0).float().cpu().numpy()
            if np.isnan(feats).sum() == 0:
                np.save(out_path, feats, allow_pickle=False)
            else:
                output += f'\n{file} contains NaN'
                yield output

        except Exception as e:
            output += f'\nException: {e}'
            yield output

    output += '\nProcessing features...'
    yield output


    del model
    del models
    gc.collect()
    torch.cuda.empty_cache()

    output += '\nDone!'
    yield output


def download_base_models(version_sr):
    repo = 'lj1995/VoiceConversionWebUI'
    version_sr_models = {
        'v1 40k': {'sf': 'pretrained', 'files': ['f0D40k.pth', 'f0G40k.pth']},
        'v1 48k': {'sf': 'pretrained', 'files': ['f0D48k.pth', 'f0G48k.pth']},
        'v2 40k': {'sf': 'pretrained_v2', 'files': ['f0D40k.pth', 'f0G40k.pth']}
    }
    dl_path = os.path.join('data', 'training', 'cache', 'RVC')
    for file in version_sr_models[version_sr]['files']:
        sf = version_sr_models[version_sr]['sf']
        if not os.path.isfile(os.path.join(dl_path, sf, file)):
            huggingface_hub.hf_hub_download(repo, file, subfolder=sf,
                                            local_dir=dl_path, local_dir_use_symlinks=False)


current_workspace: RvcWorkspace = None

base_data = {
    'v': 2,
    'sr': 40_000,
    'vsr': 'v2 40k',
    'f0': 'harvest',
    'dataset': ''
}


def get_workspaces():
    os.makedirs(workspace_path, exist_ok=True)
    return os.listdir(workspace_path)
