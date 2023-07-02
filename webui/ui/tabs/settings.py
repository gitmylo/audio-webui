import json
import os.path
import shutil

import gradio
import huggingface_hub
import webui.modules.models as mod


class CustomSetting:
    def __init__(self, value):
        self.value = self.load_val(value)

    def create_ui(self, setting):
        raise NotImplementedError('UI for setting not implemented.')

    def load_val(self, val):
        raise NotImplementedError('Value loading not implemented.')

    def save_val(self):
        raise NotImplementedError('Value saving not implemented.')


config = {
    'tts_use_gpu': {
        'type': bool,
        'default': False,
        'readname': 'TTS: use gpu',
        'description': 'Use the GPU for tts',
        'el_kwargs': {}
    }
}


config_path = os.path.join('data', 'config.json')


def auto_value(val):
    if hasattr(val, 'save_val'):
        return val.save_val()
    return val


def save_config():
    global config
    output = {key: {k: auto_value(v) for k, v in zip(config[key].keys(), config[key].values()) if k == 'value'} for key, value in zip(config.keys(), config.values())}
    json.dump(output, open(config_path, 'w'))


def load_config():
    global config
    if os.path.isfile(config_path):
        config_dict = json.load(open(config_path))
        for k, v in zip(config.keys(), config.values()):
            if k in config_dict.keys():
                v['value'] = v['type'](config_dict[k])
    for k, v in zip(config.keys(), config.values()):
        if 'value' not in v.keys():
            v['value'] = v['default']
    return config


def change_value(name, value):
    global config
    config[name]['value'] = value
    save_config()  # Maybe add autosave as a setting instead of always on if the amount of settings becomes too much


def ui_for_setting(setting):
    if hasattr(setting['value'], 'create_ui'):
        return setting['value'].create_ui(setting)

    standard_kwargs = {
        'value': setting['value'],
        'label': setting['readname']
    }

    for kwarg, kwarg_val in zip(setting['el_kwargs'].keys(), setting['el_kwargs'].values()) if 'el_kwargs' in setting.keys() else []:
        standard_kwargs[kwarg] = kwarg_val

    withinfo = standard_kwargs.copy()
    withinfo['info'] = setting['description']

    match setting['type'].__name__:
        case 'bool':
            return gradio.Checkbox(**withinfo)
        case 'int' | 'float':
            num_type = 'number'
            if 'num_type' in setting.keys():
                num_type = setting['num_type']
            match num_type:
                case 'number':
                    return gradio.Number(**withinfo)
                case 'slider':
                    return gradio.Slider(**withinfo)
        case 'list':
            list_type = 'dropdown'
            if 'list_type' in setting.keys():
                list_type = setting['list_type']
            match list_type:
                case 'dropdown':
                    return gradio.Dropdown(choices=setting['choices'], **withinfo)
                case 'radio':
                    return gradio.Radio(choices=setting['choices'], **withinfo)
        case 'str':
            return gradio.Textbox(**withinfo)

    raise NotImplementedError('Setting type not implemented! Create a new setting type by extending CustomSetting.')


def login_hf(token):
    try:
        huggingface_hub.login(token)
    except (ValueError, ImportError) as e:
        return False
    return True


def delete_model(model):
    shutil.rmtree(f'data/models/{model}')
    return mod.refresh_choices()


def settings():
    load_config()
    save_config()
    for key, setting in zip(config.keys(), config.values()):
        elem = ui_for_setting(setting)
        if elem is not None:
            elem.change(fn=lambda v: change_value(key, v), inputs=elem)


def extra_tab():
    with gradio.Tabs():
        with gradio.Tab('Main'):
            settings()
        with gradio.Tab('Extra'):
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
