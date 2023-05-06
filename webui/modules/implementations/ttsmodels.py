import gradio

import webui.modules.models as mod


class BarkTTS(mod.TTSModelLoader):
    def _components(self, **quick_kwargs):
        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', **quick_kwargs)
        return [textbox]

    from bark import SAMPLE_RATE, generate_audio, preload_models
    model = 'suno/bark'

    def get_response(self, *inputs):
        pass

    def unload_model(self):
        pass

    def load_model(self):
        return self.gradio_components()


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
