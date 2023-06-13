import shutil

import gradio
import huggingface_hub
import webui.modules.models as mod


def login_hf(token):
    try:
        huggingface_hub.login(token)
    except (ValueError, ImportError) as e:
        return False
    return True


def delete_model(model):
    shutil.rmtree(f'data/models/{model}')
    return mod.refresh_choices()


def extra_tab():
    gradio.Markdown('# 🤗 Huggingface')
    with gradio.Row():
        with gradio.Column():
            textbox = gradio.Textbox(placeholder='Huggingface token from https://huggingface.co/settings/tokens goes here',
                                     label='Huggingface token for private/gated models', lines=1, info='Put your key here if you\'re trying to load private or gated models.')
            login = gradio.Button('Log in with this token', variant='primary')
            login.click(fn=login_hf, inputs=textbox, api_name='login_hf', show_progress=True)
        with gradio.Column():
            installed_models = gradio.Dropdown(mod.choices(), label='Installed models')
            with gradio.Row():
                delete = gradio.Button('Delete model', variant='stop')
                refresh = gradio.Button('Refresh models', variant='primary')
            delete.click(fn=delete_model, inputs=installed_models, outputs=installed_models, show_progress=True, api_name='models/delete')
            refresh.click(fn=mod.refresh_choices, outputs=installed_models, show_progress=True)
