import numpy as np
import parselmouth
import torch
import torchaudio
import torchaudio.functional as F
import pyworld
import torchcrepe
from scipy import signal


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
    print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
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


def pitch_extract(f0_method, x, f0_min, f0_max, p_len, time_step, sr, window, crepe_hop_length):
    if f0_method == "pm":
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
    elif f0_method in ['harvest', 'dio']:
        if f0_method == 'harvest':
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10,
            )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, sr)
        f0 = signal.medfilt(f0, 3)
    elif f0_method == "torchcrepe":
        f0 = get_f0_crepe_computation(x, f0_min, f0_max, p_len, sr, crepe_hop_length)
    elif f0_method == "torchcrepe tiny":
        f0 = get_f0_crepe_computation(x, f0_min, f0_max, p_len, sr, crepe_hop_length, "tiny")
    return f0
