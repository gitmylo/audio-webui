import gradio

import webui.modules.models as mod


class BarkTTS(mod.TTSModelLoader):
    no_install = True

    def _components(self, **quick_kwargs):
        def update_speaker(option):
            if option == 'Upload':
                speaker.hide = True
                speaker_file.hide = False
                return [gradio.update(visible=True), gradio.update(visible=False)]
            else:
                speaker.hide = False
                speaker_file.hide = True
                return [gradio.update(visible=False), gradio.update(visible=True)]

        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', **quick_kwargs)
        mode = gradio.Radio(['File', 'Upload'], value='File', **quick_kwargs)
        with gradio.Row():
            text_temp = gradio.Slider(0, 1, 0.7, step=0.05, label='Text temperature', **quick_kwargs)
            waveform_temp = gradio.Slider(0, 1, 0.7, step=0.05, label='Waveform temperature', **quick_kwargs)

        speaker = gradio.Textbox(lines=1, label='Speaker', placeholder='Speaker goes here, or empty to let the AI guess', **quick_kwargs)
        speaker_file = gradio.File(label='Speaker', file_types=['audio'], **quick_kwargs)
        speaker_file.hide = True  # Custom, auto hide speaker_file

        mode.select(fn=update_speaker, inputs=mode, outputs=[speaker, speaker_file])
        return [textbox, mode, text_temp, waveform_temp, speaker, speaker_file]

    model = 'suno/bark'

    def get_response(self, *inputs):
        textbox, mode, text_temp, waveform_temp, speaker, speaker_file = inputs
        _speaker = None
        if mode == 'File':
            _speaker = speaker
        else:
            # _speaker = speaker_file
            pass
        from bark.api import generate_audio
        from bark.generation import SAMPLE_RATE
        return SAMPLE_RATE, generate_audio(textbox, speaker if speaker else None, text_temp, waveform_temp)

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
