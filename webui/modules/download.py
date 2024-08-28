import os.path

import gradio
import huggingface_hub
import webui.modules.models as mod

model_types = ['text-to-speech', 'audio-to-audio', 'rvc']


class AutoModel:
    def __init__(self, repo_id, model_type):
        self.repo_id = repo_id
        self.model_type = model_type

    def __str__(self):
        return self.repo_id


def get_rvc_models():
    path = os.path.join('data', 'models', 'rvc')
    output = []
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        if os.path.isdir(f_path):
            for f2 in os.listdir(f_path):
                if f2.endswith('.pth') and f2 not in ['f0D40k.pth', 'f0G40k.pth', 'f0D48k.pth', 'f0G48k.pth']:
                    output.append(os.path.join(f, f2))
        # Don't allow files anymore, it's bugged.
        # elif os.path.isfile(f_path):
        #     if f.endswith('.pth') and f not in ['f0D40k.pth', 'f0G40k.pth', 'f0D48k.pth', 'f0G48k.pth']:
        #         output.append(f)
    return output


def fill_models(model_type: str):
    if model_type == 'text-to-speech':
        return [m for m in mod.all_tts() if not m.no_install]
    if model_type == 'rvc':
        return get_rvc_models()
    return [model.id for model in
            huggingface_hub.list_models(task=model_type, sort='downloads')]


def get_file_name(repo_id: str):
    return repo_id.replace('/', '--')