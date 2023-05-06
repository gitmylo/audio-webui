import gradio
import huggingface_hub

model_types = ['text-to-speech', 'automatic-speech-recognition', 'audio-to-audio']


class AutoModel:
    def __init__(self, repo_id, model_type):
        self.repo_id = repo_id
        self.model_type = model_type

    def __str__(self):
        return self.repo_id


def fill_models(model_type: str):
    return [model.modelId for model in
            huggingface_hub.list_models(filter=huggingface_hub.ModelFilter(task=model_type), sort='downloads')]


def get_file_name(repo_id: str):
    return repo_id.replace('/', '--')


def choices():
    import webui.modules.models as mod  # Avoid circular import
    return [_type + '/' + model for _type in model_types for model in mod.get_installed_models(_type)]


def refresh_choices():
    return gradio.Dropdown.update('', choices())



def hub_download(repo_id: str, model_type: str):
    return [f'data/models/{model_type}/{get_file_name(repo_id)}', gradio.Dropdown.update()]
    try:
        huggingface_hub.snapshot_download(repo_id, local_dir_use_symlinks=False,
                                          local_dir=f'data/models/{model_type}/{get_file_name(repo_id)}')
    except Exception as e:
        return [f'<p style="color: red;">{str(e)}</p>', gradio.Dropdown.update()]
    return [f"Successfully downloaded <a target='_blank' href='https://www.huggingface.co/{repo_id}'>{repo_id}</a>", refresh_choices()]
