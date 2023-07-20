import json
import os.path
import shlex
import subprocess
from enum import Enum

from setup_tools.os import is_windows

extension_states = os.path.join('data', 'extensions.json')
ext_folder = os.path.join('extensions')


def git_ready():
    cmd = 'git -v'
    cmd = cmd if is_windows() else shlex.split(cmd)
    result = subprocess.run(cmd).returncode
    return result == 0


class UpdateStatus(Enum):
    no_git = -1
    unmanaged = 0
    updated = 1
    outdated = 2


class Extension:
    def __init__(self, ext_name, load_states):
        self.enabled = (ext_name not in load_states.keys()) or load_states[ext_name]
        self.extname = ext_name
        self.path = os.path.join(ext_folder, ext_name)
        self.main_file = os.path.join(self.path, 'main.py')
        self.req_file = os.path.join(self.path, 'requirements.py')  # Optional
        self.style_file = os.path.join(self.path, 'style.py')
        self.js_file = os.path.join(self.path, 'scripts', 'script.js')
        self.git_dir = os.path.join(self.path, '.git')
        extinfo = os.path.join(self.path, 'extension.json')
        if os.path.isfile(extinfo):
            with open(extinfo, 'r', encoding='utf8') as info_file:
                self.info = json.load(info_file)
        else:
            raise FileNotFoundError(f'No extension.json file for {ext_name} extension.')

    def activate(self):
        if self.enabled and os.path.isfile(self.main_file):
            __import__(os.path.splitext(self.main_file)[0].replace(os.path.sep, '.'), fromlist=[''])

    def get_style_rules(self):
        if self.enabled and os.path.isfile(self.style_file):
            __import__(os.path.splitext(self.style_file)[0].replace(os.path.sep, '.'), fromlist=[''])

    def get_requirements(self):
        if self.enabled and os.path.isfile(self.req_file):
            return __import__(os.path.splitext(self.req_file)[0].replace(os.path.sep, '.'), fromlist=['']).requirements()
        return []

    def get_javascript(self) -> str | bool:
        if self.enabled and os.path.isfile(self.js_file):
            return self.js_file
        return False

    def set_enabled(self, new):
        self.enabled = new
        set_load_states()

    def check_updates(self) -> UpdateStatus:
        if not os.path.isdir(self.git_dir):
            return UpdateStatus.unmanaged
        return UpdateStatus.updated  # No check yet, TODO: add check


def get_valid_extensions():
    return [e for e in os.listdir(ext_folder)
            if os.path.isdir(os.path.join(ext_folder, e))
            and os.path.isfile(os.path.join(ext_folder, e, 'extension.json'))]


states: dict[str, Extension] = {}


def set_load_states():
    s = {k: v.enabled for k, v in zip(states.keys(), states.values())}
    json.dump(s, open(extension_states, 'w', encoding='utf8'))


def get_load_states():
    if os.path.isfile(extension_states):
        return json.load(open(extension_states, 'r', encoding='utf8'))
    return {}


register_callbacks = [
    'webui.init',
    'webui.settings'
]


def init_extensions():
    # Register default callbacks
    from webui.extensionlib.callbacks import register_new as register
    for cb in register_callbacks:
        register(cb)

    # Load enabled extensions
    s = get_load_states()
    exts = get_valid_extensions()
    print(f'Found extensions: {", ".join(exts)}')
    for ext in exts:
        states[ext] = Extension(ext, s)


def get_scripts() -> list[str]:
    out = []
    for script in [e.get_javascript() for e in states.values()]:
        if script:
            out.append(script)
    return out


def get_requirements():
    out = []
    for req in [e.get_requirements() for e in states.values()]:
        if req:
            out += req
    return out
