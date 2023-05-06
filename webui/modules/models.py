import gc
import os

import gradio
import torch.cuda
from transformers import Pipeline

loaded_model = None


def choices():
    from .download import model_types
    return [_type + '/' + model for _type in model_types for model in get_installed_models(_type)]


def refresh_choices():
    return gradio.Dropdown.update('', choices())


def get_installed_models(model_type):
    _dir = f'data/models/{model_type}'
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
    found = []
    for model in [name for name in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, name))]:
        found.append(model)
    return found


class ModelLoader:
    def __init__(self, model_type):
        self.type = model_type
        self.pipeline: Pipeline = None

    def load_model(self, name):
        _dir = f'data/models/{self.type}/{name}'
        self.pipeline = self._load_internal(_dir)

    def _load_internal(self, path):
        return Pipeline.from_pretrained(task=self.type, model=path)

    def unload_model(self, name):
        del self.pipeline
        if not self.pipeline.device == 'cpu':
            torch.cuda.empty_cache()
        gc.collect()

    def get_loaded_model(self):
        return self.pipeline

    def get_response(self, *inputs):
        raise NotImplementedError('Not implemented, please implement this method.')


def all_tts():
    import webui.modules.implementations as impl
    return impl.tts.all_tts()


def all_tts_models():
    return [model.model for model in all_tts()]


class TTSModelLoader(ModelLoader):
    def get_response(self, *inputs):
        raise NotImplementedError('Not implemented, please implement this method.')

    model: str
    trigger: str

    def __init__(self):
        super().__init__('text-to-speech')
        self.trigger = self.model.replace('/', '--')

    def load_model(self):
        raise NotImplementedError('Not implemented, please implement this method.')

    def unload_model(self, name):
        raise NotImplementedError('Not implemented, please implement this method.')

    @staticmethod
    def from_model(model_path):
        pass
