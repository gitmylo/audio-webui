from webui import args  # Will show help message if needed
import os
# Set custom default huggingface download path
os.environ['HF_HOME'] = os.getenv('HF_HOME', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'unclassified'))
os.environ['MUSICGEN_ROOT'] = os.getenv('MUSICGEN_ROOT', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'musicgen'))

from autodebug.prelaunch import prelaunch_checks
from autodebug import autodebug

try:
    print('Checking installs and venv + autodebug checks')

    prelaunch_checks()

    print('Preparing')

    from webui.modules.implementations.tts_monkeypatching import patch as patch1
    patch1()

    import torch
    print('Launching, cuda available:', torch.cuda.is_available())

    from webui.webui import launch_webui

    launch_webui()

except Exception as e:
    autodebug.catcher(e)
