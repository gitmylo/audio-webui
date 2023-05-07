import os.path

import gradio

import webui.modules.models as mod


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
    def create_voice(file):
        pass

    def _components(self, **quick_kwargs):
        def update_speaker(option):
            if option == 'File':
                speaker.hide = True
                refresh_speakers.hide = True
                speaker_file.hide = False
                return [gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True)]
            else:
                speaker.hide = False
                refresh_speakers.hide = False
                speaker_file.hide = True
                return [gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False)]

        def update_voices():
            return gradio.update(choices=self.get_voices())

        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', **quick_kwargs)
        mode = gradio.Radio(['File', 'Upload'], value='File', **quick_kwargs)
        with gradio.Row():
            text_temp = gradio.Slider(0, 1, 0.7, step=0.05, label='Text temperature', **quick_kwargs)
            waveform_temp = gradio.Slider(0, 1, 0.7, step=0.05, label='Waveform temperature', **quick_kwargs)
        with gradio.Row():
            speaker = gradio.Dropdown(self.get_voices(), value='None', show_label=False, **quick_kwargs)
            refresh_speakers = gradio.Button('🔃', variant='tool secondary', **quick_kwargs)
        refresh_speakers.click(fn=update_voices, outputs=speaker)
        speaker_file = gradio.File(label='Speaker', file_types=['audio'], **quick_kwargs)
        speaker_file.hide = True  # Custom, auto hide speaker_file

        mode.select(fn=update_speaker, inputs=mode, outputs=[speaker, refresh_speakers, speaker_file])
        return [textbox, mode, text_temp, waveform_temp, speaker, speaker_file, refresh_speakers]

    model = 'suno/bark'

    def get_response(self, *inputs):
        textbox, mode, text_temp, waveform_temp, speaker, speaker_file, refresh_speakers = inputs
        _speaker = None
        if mode == 'File':
            _speaker = speaker if speaker != 'None' else None
        else:
            _speaker = self.create_voice(speaker_file)
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
