import gc
import json
import math
import os
import shutil
from time import sleep

import faiss
import huggingface_hub
import pandas
import soundfile
import torch.cuda
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from fairseq import checkpoint_utils
from scipy import signal

import librosa
import numpy as np
from scipy.io import wavfile
from torch.utils.data import DataLoader

from hubert import hubert_manager
from webui.modules.implementations.rvc import utils
from webui.modules.implementations.rvc.data_utils import TextAudioLoaderMultiNSFsid, TextAudioLoader, \
    DistributedBucketSampler, TextAudioCollateMultiNSFsid, TextAudioCollate, spec_to_mel_torch, mel_spectrogram_torch
from webui.modules.implementations.rvc.infer_pack import commons
from webui.modules.implementations.rvc.losses import kl_loss, feature_loss, generator_loss, discriminator_loss
from webui.modules.implementations.rvc.slicer2 import Slicer
from webui.modules.implementations.rvc.custom_pitch_extraction import pitch_extract as pe
from webui.modules.implementations.rvc.rvc import load_audio
from webui.modules.implementations.rvc.utils import savee
from webui.ui.tabs.training.training.workspace import Workspace

workspace_path = os.path.join('data', 'training', 'RVC')
loss_per = 100
graph_step = 5

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "51545"
dist.init_process_group(
    backend="gloo", init_method="env://", world_size=1, rank=0
)


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
        elif data['vsr'] == 'v2 40k':
            data_in['v'] = 2
            data_in['sr'] = 40_000
        elif data['vsr'] == 'v2 48k':
            data_in['v'] = 2
            data_in['sr'] = 48_000
        download_base_models(data_in['vsr'])
        return super(RvcWorkspace, self).create(data_in)

    def load(self):
        model = super(RvcWorkspace, self).load()
        for key in [key for key in base_data.keys() if key not in model.data]:
            model.data[key] = base_data[key]
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
    for f in os.listdir(dataset) if os.path.isdir(dataset) else [dataset]:
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

    # shutil.rmtree(output_f0, ignore_errors=True)
    # shutil.rmtree(output_f0nsf, ignore_errors=True)
    # shutil.rmtree(output_feat, ignore_errors=True)

    os.makedirs(output_f0, exist_ok=True)
    os.makedirs(output_f0nsf, exist_ok=True)
    os.makedirs(output_feat, exist_ok=True)

    output = 'Processing pitch...'
    yield output

    if f0_method != 'none':
        for i, f in enumerate(os.listdir(input_dir)):
            try:
                full_path = os.path.join(input_dir, f)
                if not os.path.isfile(full_path):
                    continue
                if i % 40 == 0:
                    output += f'\nExtracting pitch from {f}'
                    yield output
                npy_name = os.path.splitext(f)[0] + '.npy'
                npy_path_f0 = os.path.join(output_f0, npy_name)
                npy_path_f0nsf = os.path.join(output_f0nsf, npy_name)
                x = load_audio(full_path, sr)
                p_len = x.shape[0] // hop
                f0 = pe(f0_method, x, f0_min, f0_max, p_len, time_step, 16000, hop, data['crepe_hop_length'],
                        data['filter_radius'])
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
        model = model.float()
    model.eval()

    output += '\nProcessing features...'
    yield output

    for idx, file in enumerate(os.listdir(input_dir)):
        try:

            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_feat, os.path.splitext(file)[0] + '.npy')

            if not os.path.isfile(in_path):
                continue
            if idx % 40 == 0:
                output += f'\nProcessing {file}'
                yield output

            features = readwave(in_path, normalize=saved_cfg.task.normalize)

            padding_mask = torch.BoolTensor(features.shape).fill_(False)
            inputs = {
                "source": features.float().to(device)
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


    del model
    del models
    gc.collect()
    torch.cuda.empty_cache()

    output += '\nDone!'
    yield output


version_sr_models = {
    'v1 40k': {'sf': 'pretrained', 'files': ['f0D40k.pth', 'f0G40k.pth']},
    'v1 48k': {'sf': 'pretrained', 'files': ['f0D48k.pth', 'f0G48k.pth']},
    'v2 40k': {'sf': 'pretrained_v2', 'files': ['f0D40k.pth', 'f0G40k.pth']},
    'v2 48k': {'sf': 'pretrained_v2', 'files': ['f0D48k.pth', 'f0G48k.pth']}
}


def get_continue_models():
    models_path = os.path.join(current_workspace.space_path, 'models')
    if not os.path.isdir(models_path):
        return ['f0']
    return ['f0'] + os.listdir(models_path)


def copy_model(model):
    if model == 'f0':
        return 'Can\'t copy f0 model.'
    filename = current_workspace.name + '.pth'
    index_filename = f'{current_workspace.name}_added.index'
    index_path_filename = os.path.join(current_workspace.space_path, index_filename)
    model_path = os.path.join(current_workspace.space_path, 'models', model, filename)
    rvc_model_base_path = os.path.join('data', 'models', 'rvc', current_workspace.name)
    rvc_model_use_path = os.path.join(rvc_model_base_path, filename)
    index_file_out_path = os.path.join(rvc_model_base_path, index_filename)
    if os.path.isdir(rvc_model_use_path):
        shutil.rmtree(rvc_model_use_path, ignore_errors=True)
    os.makedirs(os.path.dirname(rvc_model_use_path), exist_ok=True)
    shutil.copyfile(model_path, rvc_model_use_path)
    if os.path.isfile(index_path_filename):
        shutil.copyfile(index_path_filename, index_file_out_path)
        return 'Copied model\nCopied index'


training = False


def simplify_loss_hist(loss_hist):
    return loss_hist['y']


def annotate_loss_hist(loss_hist):
    out_loss = {'x': [], 'y': []}
    loss_hist = loss_hist[-5000:]
    offset = max(len(loss_hist) - 5000, 0)
    for i, loss in enumerate(loss_hist):
        out_loss['x'].append(int(i * graph_step + offset))
        out_loss['y'].append(loss)
    return out_loss


def get_all_paths() -> tuple[str, str, str, str, int]:
    b = current_workspace.space_path
    file_names = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(b, '0_gt')) if f.endswith('.wav')]
    return [[
        os.path.join(b, '0_gt', f + '.wav'),
        os.path.join(b, '1_feat', f + '.npy'),
        os.path.join(b, '1_f0', f + '.npy'),
        os.path.join(b, '1_f0nsf', f + '.npy'),
        0
    ] for f in file_names]


def train_model(base_ckpt_, epochs):
    epochs = int(epochs)
    data = current_workspace.data
    f0 = data['f0'] != 'none'
    ckpt_path = os.path.join(current_workspace.space_path, 'models')
    base_ckpt, is_base = (
        os.path.join('data', 'training', 'cache', 'RVC', version_sr_models[data['vsr']]['sf']),
        True) if base_ckpt_ == 'f0' \
        else (os.path.join(ckpt_path, base_ckpt_), False)
    fea_dim = 256 if data['v'] == 1 else 768
    torch.manual_seed(1234)

    if not is_base:
        train_data = json.load(open(os.path.join(base_ckpt, 'train.json'), 'r'))
    else:
        train_data = {'epoch': 0, 'loss': []}

    file_D, file_G = version_sr_models[data['vsr']]['files']
    load_D, load_G = [os.path.join(base_ckpt, f) for f in (file_D, file_G)]
    base_ckpt = ckpt_path

    def get_save_paths(epoch, base_ckpt):
        base_ckpt = os.path.join(base_ckpt, 'e_' + str(epoch))
        return [os.path.join(base_ckpt, file_D), os.path.join(base_ckpt, file_G),
                os.path.join(base_ckpt, current_workspace.name + '.pth'), os.path.join(base_ckpt, 'train.json')]

    training_files = get_all_paths()

    print(data['vsr'])

    match data['vsr']:
        case 'v1 48k':
            _model = {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [10, 6, 2, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4, 4],
                "use_spectral_norm": False,
                "gin_channels": 256,
                "spk_embed_dim": 109
            }
        case 'v2 48k':
            _model = {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [12, 10, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [24, 20, 4, 4],
                "use_spectral_norm": False,
                "gin_channels": 256,
                "spk_embed_dim": 109
            }
        case 'v1 40k' | 'v2 40k' | _:
            _model = {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [10, 10, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "use_spectral_norm": False,
                "gin_channels": 256,
                "spk_embed_dim": 109
            }

    class HParams:  # Workaround
        max_wav_value = 32768.0
        sampling_rate = data['sr']
        sample_rate = str(int(data['sr']))[:-3] + 'k'
        filter_length = 2048
        hop_length = 400
        win_length = 2048
        min_text_len = 1
        max_text_len = 5000
        segment_size = 12800 if data['sr'] == 40_000 else 11520
        fp16_run = True
        learning_rate = float(data['lr'])
        betas = [0.8, 0.99]
        eps = 1e-9
        lr_decay = 0.999875
        n_mel_channels = 128
        mel_fmin = 0.0
        mel_fmax = None
        c_mel = 45
        c_kl = 1.0
        model = _model

    if f0:
        train_dataset = TextAudioLoaderMultiNSFsid(training_files, HParams)
    else:
        train_dataset = TextAudioLoader(training_files, HParams)

    train_sampler = DistributedBucketSampler(
        train_dataset,
        data['batch_size'],
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=1,
        rank=0,
        shuffle=True,
    )

    if f0:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=False,
        prefetch_factor=None,
    )

    if data['v'] == 1:
        from webui.modules.implementations.rvc.infer_pack.models import (
            SynthesizerTrnMs256NSFsid as RVC_Model_f0,
            SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminator,
        )
    else:
        from webui.modules.implementations.rvc.infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
        )

    if f0:
        net_g = RVC_Model_f0(
            HParams.filter_length // 2 + 1,
            HParams.segment_size // HParams.hop_length,
            **HParams.model,
            is_half=HParams.fp16_run,
            sr=HParams.sampling_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            HParams.filter_length // 2 + 1,
            HParams.segment_size // HParams.hop_length,
            **HParams.model,
            is_half=HParams.fp16_run,
        )

    if torch.cuda.is_available():
        net_g = net_g.cuda()
    net_d = MultiPeriodDiscriminator(HParams.model['use_spectral_norm'])
    if torch.cuda.is_available():
        net_d = net_d.cuda()
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        HParams.learning_rate,
        betas=HParams.betas,
        eps=HParams.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        HParams.learning_rate,
        betas=HParams.betas,
        eps=HParams.eps,
    )

    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[0])
        net_d = DDP(net_d, device_ids=[0])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            load_D, net_d, optim_d
        )  # D多半加载没事
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            load_G, net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        print(
            net_g.module.load_state_dict(
                torch.load(load_G, map_location="cpu")["model"]
            )
        )  ##测试不加载优化器
        print(
            net_d.module.load_state_dict(
                torch.load(load_D, map_location="cpu")["model"]
            )
        )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=HParams.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=HParams.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=HParams.fp16_run)

    output = 'Starting training'
    last_loss_hist = annotate_loss_hist(train_data['loss'])
    yield output, last_loss_hist

    # output += '\nSaving model.'
    # yield output, graph
    # for model in zip(models, get_save_paths()):
    #   save stuff here

    global training
    if training:
        return
    training = True

    epoch = train_data['epoch']
    last_saved = -1
    last_out = ''

    # Train code
    for epoch in range(train_data['epoch'], train_data['epoch'] + epochs):
        if not training:
            break

        # START OF TRAINING CODE
        train_loader.batch_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()

        data_iterator = enumerate(train_loader)

        for batch_idx, info in data_iterator:
            if not training:
                break
            # Data
            ## Unpack
            if f0:
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
            else:
                phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
            ## Load on CUDA
            if torch.cuda.is_available():
                phone = phone.cuda(0, non_blocking=True)
                phone_lengths = phone_lengths.cuda(0, non_blocking=True)
                if f0:
                    pitch = pitch.cuda(0, non_blocking=True)
                    pitchf = pitchf.cuda(0, non_blocking=True)
                sid = sid.cuda(0, non_blocking=True)
                spec = spec.cuda(0, non_blocking=True)
                spec_lengths = spec_lengths.cuda(0, non_blocking=True)
                wave = wave.cuda(0, non_blocking=True)

                # Calculate
            with autocast(enabled=HParams.fp16_run):
                if f0:
                    (
                        y_hat,
                        ids_slice,
                        x_mask,
                        z_mask,
                        (z, z_p, m_p, logs_p, m_q, logs_q),
                    ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                else:
                    (
                        y_hat,
                        ids_slice,
                        x_mask,
                        z_mask,
                        (z, z_p, m_p, logs_p, m_q, logs_q),
                    ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(
                    spec,
                    HParams.filter_length,
                    HParams.n_mel_channels,
                    HParams.sampling_rate,
                    HParams.mel_fmin,
                    HParams.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, HParams.segment_size // HParams.hop_length
                )
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        HParams.filter_length,
                        HParams.n_mel_channels,
                        HParams.sampling_rate,
                        HParams.hop_length,
                        HParams.win_length,
                        HParams.mel_fmin,
                        HParams.mel_fmax,
                    )
                if HParams.fp16_run:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(
                    wave, ids_slice * HParams.hop_length, HParams.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=HParams.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * HParams.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * HParams.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if global_step % loss_per == 0:
                lr = optim_g.param_groups[0]["lr"]
                # logger.info(
                #     "Train Epoch: {} [{:.0f}%]".format(
                #         epoch, 100.0 * batch_idx / len(train_loader)
                #     )
                # )

                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                # logger.info([global_step, lr])
                # logger.info(
                #     f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                # )
            if global_step % graph_step == 0:
                last_out = output + f'\nepoch: {epoch}\nglobal step: {global_step}'
                if last_saved >= 0:
                    last_out += f'\nlast saved epoch: {last_saved}'
                if torch.is_tensor(loss_kl):
                    loss_kl = loss_kl.item()
                train_data['loss'].append(loss_kl)
                last_loss_hist = pandas.DataFrame(annotate_loss_hist(train_data['loss']))
                yield last_out, last_loss_hist

            global_step += 1

        last_out = output + f'\nepoch: {epoch}\nglobal step: {global_step}'
        if last_saved >= 0:
            last_out += f'\nlast saved epoch: {last_saved}'
        yield last_out, last_loss_hist
        train_data['epoch'] = epoch
        if epoch != 0 and data['save_epochs'] != 0 and epoch % data['save_epochs'] == 0:
            d_save_path, g_save_path, finished_save_path, json_path = get_save_paths(epoch, base_ckpt)
            last_saved = epoch
            yield last_out, last_loss_hist
            utils.save_checkpoint(
                net_g,
                optim_g,
                HParams.learning_rate,
                epoch,
                g_save_path
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                HParams.learning_rate,
                epoch,
                d_save_path
            )
            json.dump(train_data, open(json_path, 'w'))
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            savee(
                ckpt,
                HParams.sample_rate,
                f0,
                finished_save_path,
                epoch,
                'v' + str(data['v']),
            )

        # END OF TRAINING CODE

        scheduler_g.step()
        scheduler_d.step()
    training = False
    output += '\nFinished training. Saving...'
    last_out = output + f'\nepoch: {epoch}\nglobal step: {global_step}'
    if last_saved >= 0:
        last_out += f'\nlast saved epoch: {last_saved}'
    yield last_out, last_loss_hist
    d_save_path, g_save_path, finished_save_path, json_path = get_save_paths(epoch, base_ckpt)
    utils.save_checkpoint(
        net_g,
        optim_g,
        HParams.learning_rate,
        epoch,
        g_save_path
    )
    utils.save_checkpoint(
        net_d,
        optim_d,
        HParams.learning_rate,
        epoch,
        d_save_path
    )
    json.dump(train_data, open(json_path, 'w'))
    if hasattr(net_g, "module"):
        ckpt = net_g.module.state_dict()
    else:
        ckpt = net_g.state_dict()
    savee(
        ckpt,
        HParams.sample_rate,
        f0,
        finished_save_path,
        epoch,
        'v' + str(data['v']),
    )
    yield last_out, last_loss_hist


def create_index():
    space_path = current_workspace.space_path
    data = current_workspace.data
    _name = current_workspace.name
    version = 'v' + str(data['v'])

    exp_dir = os.path.join(space_path)

    feature_dir = os.path.join(exp_dir, '1_feat')

    os.makedirs(exp_dir, exist_ok=True)
    # feature_dir = (
    #     "%s/3_feature256" % (exp_dir)
    #     if version == "v1"
    #     else "%s/3_feature768" % (exp_dir)
    # )
    if not os.path.exists(feature_dir):
        yield "Please perform feature extraction first!"
        return
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        yield "Please perform feature extraction first!"
        return
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = ["%s,%s" % (big_npy.shape, n_ivf)]
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("building")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        os.path.join(exp_dir, f'{_name}_trained.index')
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        os.path.join(exp_dir, f'{_name}_added.index')
    )
    infos.append(
        f"Successfully built index，{_name}_added.index"
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


def cancel_train():
    global training
    if training:
        training = False
        return 'Training cancelled.'
    return 'Not cancelled, was not training.'


def download_base_models(version_sr):
    repo = 'lj1995/VoiceConversionWebUI'
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
    'crepe_hop_length': 128,
    'dataset': '',
    'save_epochs': 10,
    'batch_size': 6,
    'lr': '1e-4',
    'filter_radius': 3
}


def get_workspaces():
    os.makedirs(workspace_path, exist_ok=True)
    return os.listdir(workspace_path)
