import shutil

import gradio
import huggingface_hub
import webui.modules.download as dl
import webui.modules.models as mod


def login_hf(token):
    try:
        huggingface_hub.login(token)
    except (ValueError, ImportError) as e:
        return False
    return True


def choices():
    return [_type + '/' + model for _type in dl.model_types for model in mod.get_installed_models(_type)]


def refresh_choices():
    return gradio.Dropdown.update('', choices())


def delete_model(model):
    shutil.rmtree(f'data/models/{model}')
    return refresh_choices()


def extra_tab():
    gradio.HTML('<h1>Huggingface</h1>')
    with gradio.Row():
        with gradio.Column():
            textbox = gradio.Textbox(placeholder='Huggingface token from https://huggingface.co/settings/tokens goes here',
                                     label='Huggingface token for private/gated models', lines=1)
            login = gradio.Button('Log in with this token', variant='primary')
            login.click(fn=login_hf, inputs=textbox, api_name='login_hf', show_progress=True)
        with gradio.Column():
            installed_models = gradio.Dropdown(choices(), label='Installed models')
            with gradio.Row():
                delete = gradio.Button('Delete model', variant='stop')
                refresh = gradio.Button('Refresh models', variant='primary')
            delete.click(fn=delete_model, inputs=installed_models, outputs=installed_models, show_progress=True, api_name='models/delete')
            refresh.click(fn=refresh_choices, outputs=installed_models, show_progress=True)

    with gradio.Row():
        for _type in dl.model_types:
            # models = dl.fill_models(_type)
            with gradio.Column():
                repo = gradio.Dropdown(label=f'{_type} models', allow_custom_value=True)
                with gradio.Row():
                    download = gradio.Button('Download model')
                    refresh = gradio.Button('Refresh models', variant='primary')
                model_result = gradio.HTML()
                download.click(lambda r: dl.hub_download(r, _type), inputs=[repo], outputs=[model_result, installed_models],
                               show_progress=True, api_name=f'models/{_type}/download')
                refresh.click(lambda: gradio.Dropdown.update(choices=dl.fill_models(_type)), outputs=repo)
