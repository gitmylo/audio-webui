import gc
import os

import torch.cuda
from transformers import pipeline, PretrainedConfig, Pipeline

from .download import model_types
loaded_model = None


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
        self.pipeline: Pipeline = None
        self.type = model_type

    def load_model(self, name):
        _dir = f'data/models/{self.type}/{name}'
        self.pipeline = self._load_internal(_dir)

    def _load_internal(self, path):
        return pipeline(task=self.type, model=path)

    def unload_model(self, name):
        del self.pipeline
        if not self.pipeline.device == 'cpu':
            torch.cuda.empty_cache()
        gc.collect()

    def get_loaded_model(self):
        return self.pipeline
