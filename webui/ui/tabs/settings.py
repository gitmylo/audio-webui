import json
import os.path
import shlex
import shutil
import subprocess

import gradio
import huggingface_hub
import webui.modules.models as mod
import webui.extensionlib.callbacks as cb
from setup_tools.os import is_windows


class CustomSetting:
    def __init__(self, value):
        self.value = self.load_val(value)

    def create_ui(self, name, setting):
        raise NotImplementedError('UI for setting not implemented.')

    def load_val(self, val):
        raise NotImplementedError('Value loading not implemented.')

    def save_val(self):
        raise NotImplementedError('Value saving not implemented.')


class BarkMix(CustomSetting):
    model_indexes = ['text', 'coarse', 'fine']
    valid_models = {
        'large': {
            'name': 'large',
            'large': True
        },
        'small': {
            'name': 'small',
            'large': False
        }
    }

    def create_ui(self, name, setting):
        gradio.Markdown('### Bark models')
        with gradio.Row():

            def create_model_el(model):
                dd = gradio.Dropdown(self.valid_models.keys(), value=setting['value'].value[model]['name'], label=model)

                def s(v):
                    self.value[model] = self.valid_models[v]
                    change_value(name)

                dd.change(fn=s, inputs=dd)


            for model in self.model_indexes:
                create_model_el(model)

    def load_val(self, val):
        val = val.split(':')
        if len(val) != 3:
            raise ValueError('Incorrect syntax for --bark-models-mix, use 3 characters')

        selected_models = []
        for char in val:
            if char not in self.valid_models.keys():
                raise ValueError(
                    f'An unknown model was specified for --bark-models-mix, Available models: {list(self.valid_models.keys())}')
            selected_models.append(self.valid_models[char])
        return {key: value for key, value in zip(self.model_indexes, selected_models)}

    def save_val(self):
        return ':'.join([self.value[idx]['name'] for idx in self.model_indexes])


config = {
    'bark_models_mix': {
        'tab': '🐶 Bark',
        'type': BarkMix,
        'default': BarkMix('large:large:large')
    },
    'bark_half': {
        'tab': '🐶 Bark',
        'type': bool,
        'default': False,
        'readname': 'Half precision',
        'description': 'Use half precision for Bark models, lower VRAM, slightly slower generations.'
    },
    'bark_offload_cpu': {
        'tab': '🐶 Bark',
        'type': bool,
        'default': True,
        'readname': 'Cpu offload',
        'description': 'Only load the currently needed bark model into the gpu\'s VRAM, keep unused models in RAM for quick loading.'
    },
    'bark_use_cpu': {
        'tab': '🐶 Bark',
        'type': bool,
        'default': False,
        'readname': 'Use cpu',
        'description': 'Do all processing on cpu, slow.'
    },
    # 'tts_use_gpu': {
    #     'tab': '🐸 Coqui TTS',
    #     'type': bool,
    #     'default': False,
    #     'readname': 'use gpu',
    #     'description': 'Use the GPU for TTS',
    #     'el_kwargs': {}  # Example
    # },
    'wav_type': {
        'type': str,
        'default': 'gradio',
        'choices': ['none', 'gradio', 'showwaves'],
        'readname': 'Waveform visualizer',
        'description': 'Pick a style to display the audio outputs as in a video.'
    }
}
settings_add = cb.get_manager('webui.settings')()
for settings_dict in settings_add:
    for k, v in zip(settings_dict.keys(), settings_dict.values()):
        if k not in config.keys():
            config[k] = v


config_path = os.path.join('data', 'config.json')


def get(name):
    val = config[name]['value']
    if isinstance(val, CustomSetting):
        return val.value
    return val


def auto_value(val):
    if hasattr(val, 'save_val'):
        return val.save_val()
    return val


def save_config():
    global config
    output = {key: [auto_value(v) for k, v in zip(config[key].keys(), config[key].values()) if k == 'value'][0] for key, value in zip(config.keys(), config.values())}
    json.dump(output, open(config_path, 'w'), indent=2)


def load_config():
    global config
    if os.path.isfile(config_path):
        config_dict = json.load(open(config_path))
        for k, v in zip(config.keys(), config.values()):
            if k in config_dict.keys():
                v['value'] = v['type'](config_dict[k])
    for k, v in zip(config.keys(), config.values()):
        v.setdefault('value', v['default'])
    return config


def change_value(name, value=None):
    global config
    if value is not None:
        config[name]['value'] = value
    if 'change_call' in config[name].keys():
        config[name].change_call()
    save_config()  # Maybe add autosave as a setting instead of always on if the amount of settings becomes too much


def ui_for_setting(name, setting):
    if hasattr(setting['value'], 'create_ui'):
        return setting['value'].create_ui(name, setting)

    standard_kwargs = {
        'value': setting.get('value', setting.get('default', None)),
        'label': setting.get('readname', None)
    }

    for kwarg, kwarg_val in zip(setting['el_kwargs'].keys(), setting['el_kwargs'].values()) if 'el_kwargs' in setting.keys() else []:
        standard_kwargs[kwarg] = kwarg_val

    withinfo = standard_kwargs.copy()
    withinfo['info'] = setting.get('description', None)

    typename = setting['type'].__name__
    match typename:
        case 'bool':
            return gradio.Checkbox(**withinfo)
        case 'int' | 'float':
            num_type = 'number'
            if 'num_type' in setting.keys():
                num_type = setting['num_type']
            step_val = setting.get('step', 1 if typename == 'int' else None)
            match num_type:
                case 'number':
                    return gradio.Number(precision=step_val, **withinfo)
                case 'slider':
                    return gradio.Slider(step=step_val, **withinfo)
        case 'list':
            list_type = 'dropdown'
            if 'list_type' in setting.keys():
                list_type = setting['list_type']
            match list_type:
                case 'dropdown':
                    return gradio.Dropdown(choices=setting['choices'], multiselect=True, **withinfo)
                case 'checkbox':
                    return gradio.CheckboxGroup(choices=setting['choices'], **withinfo)
        case 'str':
            if 'choices' in setting.keys():
                list_type = 'dropdown'
                if 'list_type' in setting.keys():
                    list_type = setting['list_type']
                match list_type:
                    case 'dropdown':
                        return gradio.Dropdown(choices=setting['choices'], **withinfo)
                    case 'radio':
                        return gradio.Radio(choices=setting['choices'], **withinfo)
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
    save_config()  # Save newly added values

    tab_config = {}
    other_tab = {}
    for key, setting in zip(config.keys(), config.values()):
        tab = setting.get('tab', '❓ other')
        if tab not in tab_config.keys():
            if tab == '❓ other':
                other_tab[key] = setting
            else:
                tab_config[tab] = {key: setting}
        else:
            tab_config[tab][key] = setting

    tab_config['❓ other'] = other_tab  # Ensure other is at end

    def change_setting(setting):
        def c(v):
            change_value(setting, v)
        return c

    with gradio.Tabs():
        for tab, key_dict in zip(tab_config.keys(), tab_config.values()):
            with gradio.Tab(tab):
                for key, setting in zip(key_dict.keys(), key_dict.values()):
                    elem = ui_for_setting(key, setting)
                    if elem is not None:
                        elem.change(fn=change_setting(key), inputs=elem)


def extensions_tab():
    with gradio.Tabs():
        with gradio.Tab('✅ Installed'):
            list_all_extensions()
        with gradio.Tab('⬇️ Install new extensions'):
            install_extensions_tab()


def list_all_extensions():
    import webui.extensionlib.extensionmanager as em

    with gradio.Row():
        check_updates = gradio.Button('Check for updates', variant='primary')
        update_selected = gradio.Button('Update selected', variant='primary')
        shutdown = gradio.Button('Shutdown audio webui')
        shutdown.click(fn=lambda: os._exit(0))

    gradio.Markdown('Extension managing is still work in progress, and will require restarts of the webui.')

    with gradio.Row():
        gradio.Markdown('## Name')
        gradio.Markdown('## Description')
        gradio.Markdown('## Author')
        gradio.Markdown('## Update if available')
        gradio.Markdown('## Enabled')

    updatelist = []

    def add_or_remove(name):
        def f(val):
            if val:
                updatelist.append(val)
            else:
                try:
                    updatelist.remove(val)
                except:
                    pass

        return f

    for e in em.states.values():
        with gradio.Row() as parent:
            gradio.Markdown(e.info['name'])
            gradio.Markdown(e.info['description'])
            gradio.Markdown(e.info['author'])
            with gradio.Row():
                e.update_el = [gradio.Markdown('Not checked'), gradio.Checkbox(False, label='Update', visible=False)]
                e.update_el[1].change(fn=add_or_remove(e.extname), inputs=e.update_el[1])
            enabled = gradio.Checkbox(e.enabled, label='Enabled')
            enabled.change(fn=e.set_enabled, inputs=enabled, outputs=enabled)

    def quick_update_return(val, l):
        if isinstance(val, str):
            l.append(gradio.update(visible=True, value=val))
            l.append(gradio.update(visible=False))
            return
        l.append(gradio.update(visible=False))
        l.append(gradio.update(visible=True, value=val))

    def check_for_updates():
        if em.git_ready():
            out_list = []
            for e in em.states.values():
                update_status = e.check_updates()
                match update_status:
                    case em.UpdateStatus.no_git:
                        if e.extname in updatelist:
                            updatelist.remove(e.extname)
                        quick_update_return('Git not installed', out_list)
                    case em.UpdateStatus.updated:
                        if e.extname in updatelist:
                            updatelist.remove(e.extname)
                        quick_update_return('Up to date', out_list)
                    case em.UpdateStatus.unmanaged:
                        if e.extname in updatelist:
                            updatelist.remove(e.extname)
                        quick_update_return('I had an issue with git', out_list)
                    case em.UpdateStatus.outdated:
                        if e.extname not in updatelist:
                            updatelist.append(e.extname)
                        quick_update_return(True, out_list)
            return out_list

        return quick_update_return('Git not installed') * len(em.states)

    all_els = []
    for e in em.states.values():
        all_els.append(e.update_el[0])
        all_els.append(e.update_el[1])

    check_updates.click(fn=check_for_updates, outputs=all_els)

    def update_exts():
        for e in updatelist:
            ext = em.states[e]
            ext.update()

    update_selected.click(fn=update_exts())


def install_extensions_tab():
    import webui.extensionlib.extensionmanager as em

    def install_extension(url):
        command = f'git clone {url}'
        command = command if is_windows() else shlex.split(command)
        out = subprocess.run(command, cwd=os.path.abspath(em.ext_folder))
        if out.returncode != 0:
            return '', f'Something went wrong with installing! Check output in console for details.'
        return '', f'Installed {url} (Not checked if successful, check console for status)'

    with gradio.Row():
        repo_url = gradio.Textbox(placeholder='https://www.github.com/user/repo', label='Git repo url', max_lines=1)
        download_button = gradio.Button('Install from url', variant='primary')
    markdown = gradio.Markdown()

    download_button.click(fn=install_extension, inputs=repo_url, outputs=[repo_url, markdown])


def extra_tab():
    with gradio.Tabs():
        with gradio.Tab('✅ Main'):
            settings()
        with gradio.Tab('🚀 Extensions'):
            extensions_tab()
        with gradio.Tab('➕ Extra'):
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
