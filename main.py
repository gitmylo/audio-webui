from webui.args import args  # Will show help message if needed
import os
# Set custom default huggingface download path
if not args.no_data_cache:
    cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'unclassified')
    if os.environ.get("MODELS_ROOT"):
        workspace_root = os.environ.get("MODELS_ROOT", 'data')
        cache_path = os.path.join(workspace_root, 'models', 'unclassified')
        os.makedirs(cache_path, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = os.getenv('XDG_CACHE_HOME', cache_path)
    os.environ['MUSICGEN_ROOT'] = os.getenv('MUSICGEN_ROOT', cache_path)
    os.environ['HF_HOME'] = os.getenv('HF_HOME', cache_path)

from autodebug.prelaunch import prelaunch_checks
from autodebug import autodebug

try:
    print('Checking installs and venv + autodebug checks')

    prelaunch_checks()

    print('Activating extensions')
    import webui.extensionlib.extensionmanager as em
    for e in em.states.values():
        e.activate()

    print('Preparing')
    from webui.modules.implementations.tts_monkeypatching import patch as patch1
    patch1()

    from webui.modules.implementations.gradio_monkeypatching import patch as patch2
    patch2()

    import torch
    print('Launching, cuda available:', torch.cuda.is_available())


    from webui.webui import launch_webui

    launch_webui()

except Exception as e:
    autodebug.catcher(e)
