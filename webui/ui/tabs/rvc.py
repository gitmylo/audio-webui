import os

import torch.cuda
import torchaudio
import gradio
from webui.modules import util

from webui.modules.download import fill_models

flag_strings = ['denoise', 'denoise output', 'separate background']


def flatten_audio(audio_tensor: torch.Tensor | tuple[torch.Tensor, int] | tuple[int, torch.Tensor], add_batch=True):
    if isinstance(audio_tensor, tuple):
        if isinstance(audio_tensor[0], int):
            return audio_tensor[0], flatten_audio(audio_tensor[1])
        elif torch.is_tensor(audio_tensor[0]):
            return flatten_audio(audio_tensor[0]), audio_tensor[1]
    if audio_tensor.dtype == torch.int16:
        audio_tensor = audio_tensor.float() / 32767.0
    if audio_tensor.dtype == torch.int32:
        audio_tensor = audio_tensor.float() / 2147483647.0
    if len(audio_tensor.shape) == 2:
        if audio_tensor.shape[0] == 2:
            # audio_tensor = audio_tensor[0, :].div(2).add(audio_tensor[1, :].div(2))
            audio_tensor = audio_tensor.mean(0)
        elif audio_tensor.shape[1] == 2:
            # audio_tensor = audio_tensor[:, 0].div(2).add(audio_tensor[:, 1].div(2))
            audio_tensor = audio_tensor.mean(1)
        audio_tensor = audio_tensor.flatten()
    if add_batch:
        audio_tensor = audio_tensor.unsqueeze(0)
    return audio_tensor


def merge_and_match(x, y, sr):
    # import scipy.signal
    x = x / 2
    y = y / 2
    import torchaudio.functional as F
    y = F.resample(y, sr, int(sr * (x.shape[-1] / y.shape[-1])))
    if x.shape[0] > y.shape[0]:
        x = x[-y.shape[0]:]
    else:
        y = y[-x.shape[0]:]
    return x.add(y)


def get_models_installed():
    return [gradio.update(choices=fill_models('rvc')), gradio.update()]


def unload_rvc():
    import webui.modules.implementations.rvc.rvc as rvc
    rvc.unload_rvc()
    return [gradio.update(value=''), gradio.update(maximum=0, value=0, visible=False)]


def load_rvc(model):
    if not model:
        return unload_rvc()
    import webui.modules.implementations.rvc.rvc as rvc
    maximum = rvc.load_rvc(model)
    return [gradio.update(), gradio.update(maximum=maximum, value=0, visible=maximum > 0)]


def denoise(sr, audio):
    if not torch.is_tensor(audio):
        audio = torch.tensor(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    audio = audio.detach().cpu().numpy()
    import noisereduce.noisereduce as noisereduce
    audio = torch.tensor(noisereduce.reduce_noise(y=audio, sr=sr))
    return sr, audio


def gen(rvc_model_selected, speaker_id, pitch_extract, audio_in, up_key, index_rate, filter_radius, protect, crepe_hop_length, flag):
    background = None
    audio = None
    sr, audio_in = audio_in
    audio_tuple = (sr, torch.tensor(audio_in))

    audio_tuple = flatten_audio(audio_tuple)

    if 'separate background' in flag:
        if not torch.is_tensor(audio_tuple[1]):
            audio_tuple = (audio_tuple[0], torch.tensor(audio_tuple[1]).to(torch.float32))
        if len(audio_tuple[1].shape) != 1:
            audio_tuple = (audio_tuple[0], audio_tuple[1].flatten())
        import webui.modules.implementations.rvc.split_audio as split_audio
        foreground, background, sr = split_audio.split(*audio_tuple)
        audio_tuple = flatten_audio((sr, foreground))
        background = flatten_audio(background)
    if 'denoise' in flag:
        audio_tuple = denoise(*audio_tuple)

    if rvc_model_selected:
        print('Selected model', rvc_model_selected)
        if len(audio_tuple[1].shape) == 1:
            audio_tuple = (audio_tuple[0], audio_tuple[1].unsqueeze(0))
        torchaudio.save('speakeraudio.wav', audio_tuple[1], audio_tuple[0])

        import webui.modules.implementations.rvc.rvc as rvc
        rvc.load_rvc(rvc_model_selected)

        index_file = ''
        try:
            model_basedir = os.path.join('data', 'models', 'rvc', os.path.dirname(rvc_model_selected))
            index_files = [f for f in os.listdir(model_basedir) if f.endswith('.index')]
            if len(index_files) > 0:
                for f in index_files:
                    full_path = os.path.join(model_basedir, f)
                    if 'added' in f:
                        index_file = full_path
                if not index_file:
                    index_file = os.path.join(model_basedir, index_files[0])
        except:
            pass

        out1, out2 = rvc.vc_single(speaker_id, 'speakeraudio.wav', up_key, None, pitch_extract, index_file, '', index_rate, filter_radius, 0, 1, protect, crepe_hop_length)
        print(out1)
        audio_tuple = out2

    if background is not None and 'separate background' in flag:
        audio = audio_tuple[1] if torch.is_tensor(audio_tuple[1]) else torch.tensor(audio_tuple[1])
        audio_tuple = (audio_tuple[0], flatten_audio(audio, False))
        background = flatten_audio(background if torch.is_tensor(background) else torch.tensor(background), False)
        if audio_tuple[1].dtype == torch.int16:
            audio = audio_tuple[1]
            audio = audio.float() / 32767.0
            audio_tuple = (audio_tuple[0], audio)
        audio = audio_tuple[1]
        audio_tuple = (audio_tuple[0], merge_and_match(audio_tuple[1], background, audio_tuple[0]))

    if 'denoise output' in flag:
        audio_tuple = denoise(*audio_tuple)

    if torch.is_tensor(audio_tuple[1]):
        audio_tuple = (audio_tuple[0], audio_tuple[1].flatten().detach().cpu().numpy())

    sr = audio_tuple[0]

    audio = (sr, audio.detach().cpu().numpy()) if audio is not None else None
    background = (sr, background.detach().cpu().numpy()) if background is not None else None

    return [audio_tuple, util.make_waveform(audio_tuple), background, audio]


def rvc():
    with gradio.Row():
        with gradio.Column():
            use_microphone = gradio.Checkbox(label='Use microphone')
            audio_el = gradio.Audio(label='Audio input')
            from webui.ui.tabs.text_to_speech import to_rvc, audio_out
            from webui.ui.ui import tabs_el

            def to_rvc_func(audio):
                return gradio.update(selected='🗣▶🗣 RVC'), audio

            to_rvc.click(fn=to_rvc_func, inputs=audio_out, outputs=[tabs_el, audio_el])

            def update_audio_input(use_mic):
                return gradio.update(source='microphone' if use_mic else 'upload')
            use_microphone.change(fn=update_audio_input, inputs=use_microphone, outputs=audio_el)

            with gradio.Accordion('🗣 RVC'):
                with gradio.Row():
                    selected = gradio.Dropdown(get_models_installed()[0]['choices'], label='RVC Model')
                    with gradio.Column(elem_classes='smallsplit'):
                        refresh = gradio.Button('🔃', variant='tool secondary')
                        unload = gradio.Button('💣', variant='tool primary')
                speaker_id = gradio.Slider(value=0, step=1, maximum=0, visible=False, label='Speaker id', info='For multi speaker models, the speaker to use.')
                pitch_extract = gradio.CheckboxGroup(choices=["dio", "pm", "harvest", "torchcrepe", "torchcrepe tiny", "mangio-crepe", "mangio-crepe tiny", "rmvpe"], label='Pitch extraction', value='harvest', interactive=True, info='Default: dio. dio and pm are faster, harvest is slower but good. Crepe is good but uses GPU.')
                crepe_hop_length = gradio.Slider(visible=False, minimum=64, maximum=512, step=64, value=128, label='torchcrepe hop length', info='The length of the hops used for torchcrepe\'s crepe implementation')

                def update_crepe_hop_length_visible(pitch_mode: str):
                    return gradio.update(visible=any(['crepe' in v for v in pitch_mode]))

                pitch_extract.change(fn=update_crepe_hop_length_visible, inputs=pitch_extract, outputs=crepe_hop_length)

                refresh.click(fn=get_models_installed, outputs=[selected, speaker_id], show_progress=True)
                unload.click(fn=unload_rvc, outputs=[selected, speaker_id], show_progress=True)
                selected.select(fn=load_rvc, inputs=selected, outputs=[selected, speaker_id], show_progress=True)
                index_rate = gradio.Slider(0, 1, 0.88, step=0.01, label='Index rate for feature retrieval', info='Default: 0.88. Higher is more indexing, takes longer but could be better')
                filter_radius = gradio.Slider(0, 7, 3, step=1, label='Filter radius', info='Default: 3. Smooth out the pitches, should yield less voice cracks.')
                up_key = gradio.Number(value=0, label='Pitch offset', info='Default: 0. Shift the pitch up or down')
                protect = gradio.Slider(0, 0.5, 0.33, step=0.01, label='Protect amount', info='Default: 0.33. Avoid non voice sounds. Lower is more being ignored.')
            flags = gradio.Dropdown(flag_strings, label='Flags', info='Things to apply on the audio input/output', multiselect=True)
        with gradio.Column():
            with gradio.Row():
                generate = gradio.Button('Generate', variant='primary', elem_id='rvc-generate')
            with gradio.Row():
                audio_out = gradio.Audio(label='output audio', interactive=False)
            with gradio.Row():
                video_out = gradio.Video(label='output spectrogram video', interactive=False)
            with gradio.Row():
                audio_bg = gradio.Audio(label='background', interactive=False)
            with gradio.Row():
                audio_vocal = gradio.Audio(label='vocals', interactive=False)

        generate.click(fn=gen, inputs=[selected, speaker_id, pitch_extract, audio_el,
                                       up_key, index_rate, filter_radius, protect, crepe_hop_length, flags], outputs=[audio_out, video_out, audio_bg, audio_vocal])
