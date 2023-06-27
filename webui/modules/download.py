import os.path

import gradio
import huggingface_hub
import webui.modules.models as mod

model_types = ['text-to-speech', 'automatic-speech-recognition', 'audio-to-audio', 'rvc']


class AutoModel:
    def __init__(self, repo_id, model_type):
        self.repo_id = repo_id
        self.model_type = model_type

    def __str__(self):
        return self.repo_id


def get_rvc_models():
    path = os.path.join('data', 'models', 'rvc')
    output = []
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
    return [model.modelId for model in
            huggingface_hub.list_models(filter=huggingface_hub.ModelFilter(task=model_type), sort='downloads')]


def get_file_name(repo_id: str):
    return repo_id.replace('/', '--')


def hub_download(repo_id: str, model_type: str):
    try:
        huggingface_hub.snapshot_download(repo_id, local_dir_use_symlinks=False,
                                          local_dir=f'data/models/{model_type}/{get_file_name(repo_id)}')
    except Exception as e:
        return [f'<p style="color: red;">{str(e)}</p>', gradio.Dropdown.update()]
    return [f"Successfully downloaded <a target='_blank' href='https://www.huggingface.co/{repo_id}'>{repo_id}</a>", mod.refresh_choices()]
