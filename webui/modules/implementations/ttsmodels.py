import gradio

import webui.modules.models as mod
import tts_monkeypatching as mp

mp.patch()  # Monkey patch suno


class BarkTTS(mod.TTSModelLoader):
    no_install = True

    def _components(self, **quick_kwargs):
        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', **quick_kwargs)
        speaker = gradio.Textbox(lines=1, label='Speaker', placeholder='Speaker goes here, or empty to let the AI guess', **quick_kwargs)
        return [textbox, speaker]

    from bark.api import generate_audio
    from bark.generation import preload_models, clean_models, SAMPLE_RATE
    model = 'suno/bark'

    def get_response(self, *inputs):
        textbox, speaker = inputs
        return BarkTTS.SAMPLE_RATE, BarkTTS.generate_audio(textbox, speaker if speaker else None)

    def unload_model(self):
        BarkTTS.clean_models()


    def load_model(self):
        BarkTTS.preload_models()
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
