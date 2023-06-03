from webui import args  # Will show help message if needed

from install import ensure_installed

print('Checking installs and venv')

ensure_installed()  # Installs missing packages

from webui.modules.implementations.tts_monkeypatching import patch as patch1
patch1()

import torch
print('Launching, cuda available:', torch.cuda.is_available())

from webui.webui import launch_webui

launch_webui()
