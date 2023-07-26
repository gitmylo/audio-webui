import gc
import os.path

import numpy as np
import parselmouth
import torch
import pyworld
import torchcrepe
from scipy import signal
from torch import Tensor


def get_f0_crepe_computation(
        x,
        f0_min,
        f0_max,
        p_len,
        sr,
        hop_length=128,
        # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
        model="full",  # Either use crepe-tiny "tiny" or crepe "full". Default is full
):
    x = x.astype(np.float32)  # fixes the F.conv2D exception. We needed to convert double to float.
    x /= np.quantile(np.abs(x), 0.999)
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio = torch.from_numpy(x).to(torch_device, copy=True)
    audio = torch.unsqueeze(audio, dim=0)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True).detach()
    audio = audio.detach()
    # print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
    pitch: torch.Tensor = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        f0_min,
        f0_max,
        model,
        batch_size=hop_length * 2,
        device=torch_device,
        pad=True
    )
    p_len = p_len or x.shape[0] // hop_length
    # Resize the pitch for final f0
    source = np.array(pitch.squeeze(0).cpu().float().numpy())
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * p_len, len(source)) / p_len,
        np.arange(0, len(source)),
        source
    )
    f0 = np.nan_to_num(target)
    return f0  # Resized f0


def get_mangio_crepe_f0(x, f0_min, f0_max, p_len, sr, crepe_hop_length, model='full'):
    # print("Performing crepe pitch extraction. (EXPERIMENTAL)")
    # print("CREPE PITCH EXTRACTION HOP LENGTH: " + str(crepe_hop_length))
    x = x.astype(np.float32)
    x /= np.quantile(np.abs(x), 0.999)
    torch_device_index = 0
    torch_device = None
    if torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    audio = torch.from_numpy(x).to(torch_device, copy=True)
    audio = torch.unsqueeze(audio, dim=0)
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True).detach()
    audio = audio.detach()
    # print(
    #     "Initiating f0 Crepe Feature Extraction with an extraction_crepe_hop_length of: " +
    #     str(crepe_hop_length)
    # )
    # Pitch prediction for pitch extraction
    pitch: Tensor = torchcrepe.predict(
        audio,
        sr,
        crepe_hop_length,
        f0_min,
        f0_max,
        model,
        batch_size=crepe_hop_length * 2,
        device=torch_device,
        pad=True
    )
    p_len = p_len or x.shape[0] // crepe_hop_length
    # Resize the pitch
    source = np.array(pitch.squeeze(0).cpu().float().numpy())
    source[source < 0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source) * p_len, len(source)) / p_len,
        np.arange(0, len(source)),
        source
    )
    return np.nan_to_num(target)


def pitch_extract(f0_method, x, f0_min, f0_max, p_len, time_step, sr, window, crepe_hop_length, filter_radius=3):
    f0s = []
    f0 = np.zeros(p_len)
    for method in f0_method if isinstance(f0_method, list) else [f0_method]:
        if method == "pm":
            f0 = (
                parselmouth.Sound(x, sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif method in ['harvest', 'dio']:
            if method == 'harvest':
                f0, t = pyworld.harvest(
                    x.astype(np.double),
                    fs=sr,
                    f0_ceil=f0_max,
                    f0_floor=f0_min,
                    frame_period=10,
                )
            elif method == "dio":
                f0, t = pyworld.dio(
                    x.astype(np.double),
                    fs=sr,
                    f0_ceil=f0_max,
                    f0_floor=f0_min,
                    frame_period=10,
                )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, sr)
        elif method == "torchcrepe":
            f0 = get_f0_crepe_computation(x, f0_min, f0_max, p_len, sr, crepe_hop_length)
        elif method == "torchcrepe tiny":
            f0 = get_f0_crepe_computation(x, f0_min, f0_max, p_len, sr, crepe_hop_length, "tiny")
        elif method == "mangio-crepe":
            f0 = get_mangio_crepe_f0(x, f0_min, f0_max, p_len, sr, crepe_hop_length)
        elif method == "mangio-crepe tiny":
            f0 = get_mangio_crepe_f0(x, f0_min, f0_max, p_len, sr, crepe_hop_length, 'tiny')
        elif method == "rmvpe":
            rmvpe_model_path = os.path.join('data', 'models', 'rmvpe')
            rmvpe_model_file = os.path.join(rmvpe_model_path, 'rmvpe.pt')
            if not os.path.isfile(rmvpe_model_file):
                import huggingface_hub
                rmvpe_model_file = huggingface_hub.hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', local_dir=rmvpe_model_path, local_dir_use_symlinks=False)

            from webui.modules.implementations.rvc.rmvpe import RMVPE
            print("loading rmvpe model")
            model_rmvpe = RMVPE(rmvpe_model_file, is_half=True, device=None)
            f0 = model_rmvpe.infer_from_audio(x, thred=0.03)
            del model_rmvpe
            torch.cuda.empty_cache()
            gc.collect()
        f0s.append(f0)

    if not f0s:
        f0s = [f0]

    f0s_new = []
    for f0_val in f0s:
        _len = f0_val.shape[0]
        if _len == p_len:
            f0s_new.append(f0)
            continue
        if _len > p_len:
            f0 = f0[:p_len]
            f0s_new.append(f0)
            continue
        if _len < p_len:
            print('WARNING: len < p_len, skipping this f0')


    f0 = np.nanmedian(np.stack(f0s_new, axis=0), axis=0)

    if filter_radius >= 2:
        f0 = signal.medfilt(f0, filter_radius)

    return f0
