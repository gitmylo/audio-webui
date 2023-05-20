from typing import Union

from bark.api import *
from .bark_generation import generate_text_semantic_new, generate_coarse_new, generate_fine_new, codec_decode_new


def text_to_semantic_new(
    text: str,
    history_prompt: Union[str, dict] = None,
    temp: float = 0.7,
    silent: bool = False,
    allow_early_stop: bool = True
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        allow_early_stop: (Added in new) set to False to generate until the limit

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic_new(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True,
        allow_early_stop=allow_early_stop
    )
    return x_semantic


def semantic_to_waveform_new(
    semantic_tokens: np.ndarray,
    history_prompt: Union[str, dict] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
    skip_fine: bool = False,
    decode_on_cpu: bool = False
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt
        skip_fine: (Added in new) Skip converting coarse to fine
        decode_on_cpu: (Added in new) Move everything to cpu when decoding, useful for decoding huge audio files on medium vram

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse_new(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    if not skip_fine:
        fine_tokens = generate_fine_new(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )
    else:
        fine_tokens = coarse_tokens
    audio_arr = codec_decode_new(fine_tokens, decode_on_cpu)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def generate_audio_new(
    text: str,
    history_prompt: Optional[str] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
    skip_fine: bool = False,
    decode_on_cpu: bool = False,
    allow_early_stop: bool = True
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt
        skip_fine: (Added in new) Skip converting from coarse to fine
        decode_on_cpu: (Added in new) Decode on cpu
        allow_early_stop: (Added in new) Set to false to continue until the limit is reached

    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic_new(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
        allow_early_stop=allow_early_stop
    )
    out = semantic_to_waveform_new(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
        skip_fine=skip_fine,
        decode_on_cpu=decode_on_cpu
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr
