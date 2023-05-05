import gradio
import huggingface_hub
import webui.modules.download as dl


def login_hf(token):
    try:
        huggingface_hub.login(token)
    except (ValueError, ImportError) as e:
        return False
    return True


def extra_tab():
    gradio.HTML('<h1>Huggingface</h1>')
    with gradio.Row():
        textbox = gradio.Textbox(placeholder='Huggingface token from https://huggingface.co/settings/tokens goes here',
                                 label='Huggingface token for private/gated models')
        login = gradio.Button('Log in with this token')
        login.click(fn=login_hf, inputs=textbox, api_name='login_hf', show_progress=True)
    with gradio.Row():
        repo = gradio.Textbox(placeholder='microsoft/speecht5_tts', label='Model repo')
        model_type = gradio.Dropdown(dl.model_types, value=dl.model_types[0], label='Model type')
        download = gradio.Button('Download model')
    model_result = gradio.HTML()
    download.click(fn=dl.hub_download, inputs=[repo, model_type], outputs=model_result, api_name='download',
                   show_progress=True)
    with gradio.Row():
        for _type in dl.model_types:
            models = dl.fill_models(_type)
            with gradio.Column():
                repo = gradio.Dropdown(models, value=models[0], label=f'{_type} models')
                download = gradio.Button('Download model')
                model_result = gradio.HTML()
                download.click(fn=dl.hub_download, inputs=[repo, _type], outputs=model_result,
                               show_progress=True)
