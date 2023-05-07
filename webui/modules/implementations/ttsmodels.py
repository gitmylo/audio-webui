import os.path

import gradio
import numpy as np
from bark import generation

import webui.modules.models as mod


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)


class BarkTTS(mod.TTSModelLoader):
    no_install = True

    @staticmethod
    def get_voices():
        found_prompts = []
        base_path = 'data/bark_custom_speakers/'
        for path, subdirs, files in os.walk(base_path):
            for name in files:
                if name.endswith('.npz'):
                    found_prompts.append(os.path.join(path, name)[len(base_path):-4])
        from webui.modules.implementations.patches.bark_generation import ALLOWED_PROMPTS
        return ['None'] + found_prompts + ALLOWED_PROMPTS

    @staticmethod
    def create_voice(file, transcript):
        from encodec.utils import convert_audio
        import torchaudio
        import torch
        import os
        import gradio
        import numpy as np
        file_path = file.name
        file_name = '.'.join(file_path.replace('\\', '/').split('/')[-1].split('.')[:-1])
        out_file = f'data/bark_custom_speakers/{file_name}.npz'

        use_gpu = not os.environ.get("BARK_FORCE_CPU", False)
        model = generation.load_codec_model(use_gpu=use_gpu)

        # Load and pre-process the audio waveform
        model = generation.load_codec_model(use_gpu=use_gpu)
        device = generation._grab_best_device(use_gpu)
        wav, sr = torchaudio.load(file_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0).to(device)

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

        # get seconds of audio
        seconds = wav.shape[-1] / model.sample_rate
        # generate semantic tokens

        codes_copy = codes.cpu().numpy().copy()

        semantic_tokens = generation.generate_text_semantic(transcript, max_gen_duration_s=seconds, top_k=50,
                                                            min_eos_p=0.2, top_p=.95, temp=0.7)

        # semantic_tokens = np.concatenate(all_semantic_tokens)
        np.savez(out_file,
                 coarse_prompt=codes_copy[:2, :],
                 fine_prompt=codes_copy,
                 semantic_prompt=semantic_tokens
                 )
        return file_name


    def _components(self, **quick_kwargs):
        def update_speaker(option):
            if option == 'File':
                speaker.hide = True
                refresh_speakers.hide = True
                speaker_file.hide = False
                speaker_file_transcript.hide = False
                return [gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True), gradio.update(visible=True)]
            else:
                speaker.hide = False
                refresh_speakers.hide = False
                speaker_file.hide = True
                speaker_file_transcript.hide = True
                return [gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False)]

        def update_voices():
            return gradio.update(choices=self.get_voices())

        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', **quick_kwargs)
        mode = gradio.Radio(['File', 'Upload'], label='Speaker from', value='File', **quick_kwargs)
        with gradio.Row():
            text_temp = gradio.Slider(0, 1, 0.7, step=0.05, label='Text temperature', **quick_kwargs)
            waveform_temp = gradio.Slider(0, 1, 0.7, step=0.05, label='Waveform temperature', **quick_kwargs)
        with gradio.Row():
            speaker = gradio.Dropdown(self.get_voices(), value='None', show_label=False, **quick_kwargs)
            refresh_speakers = gradio.Button('🔃', variant='tool secondary', **quick_kwargs)
        refresh_speakers.click(fn=update_voices, outputs=speaker)
        speaker_file = gradio.File(label='Speaker', file_types=['audio'], **quick_kwargs)
        speaker_file_transcript = gradio.Textbox(lines=1, label='Transcript', **quick_kwargs)
        speaker_file.hide = True  # Custom, auto hide speaker_file
        speaker_file_transcript.hide = True

        mode.select(fn=update_speaker, inputs=mode, outputs=[speaker, refresh_speakers, speaker_file, speaker_file_transcript])
        return [textbox, mode, text_temp, waveform_temp, speaker, speaker_file, speaker_file_transcript, refresh_speakers]

    model = 'suno/bark'

    def get_response(self, *inputs):
        textbox, mode, text_temp, waveform_temp, speaker, speaker_file, speaker_file_transcript, refresh_speakers = inputs
        _speaker = None
        if mode == 'File':
            _speaker = speaker if speaker != 'None' else None
        else:
            _speaker = self.create_voice(speaker_file, speaker_file_transcript)
        from bark.api import generate_audio
        from bark.generation import SAMPLE_RATE
        return SAMPLE_RATE, generate_audio(textbox, _speaker, text_temp, waveform_temp)

    def unload_model(self):
        from bark.generation import clean_models
        clean_models()


    def load_model(self):
        from bark.generation import preload_models
        preload_models()


elements = []


def init_elements():
    global elements
    elements = [BarkTTS()]


def all_tts() -> list[mod.TTSModelLoader]:
    if not elements:
        init_elements()
    return elements


def all_elements(in_dict):
    l = []
    for value in in_dict.values():
        l += value
    return l


def all_elements_dict():
    d = {}
    for tts in all_tts():
        d[tts.model] = tts.gradio_components()
    return d
