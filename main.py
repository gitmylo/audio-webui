from webui.args import args  # Will show help message if needed
import os
# Set custom default huggingface download path
if not args.no_data_cache:
    os.environ['XDG_CACHE_HOME'] = os.getenv('XDG_CACHE_HOME', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'unclassified'))
    os.environ['HF_HOME'] = os.getenv('HF_HOME', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'unclassified'))
    os.environ['MUSICGEN_ROOT'] = os.getenv('MUSICGEN_ROOT', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'musicgen'))
    os.environ['HF_HUB_CACHE'] = os.getenv('HF_HUB_CACHE', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'models', 'hf_cache'))  # Experimental, due to some people being unable to install from this variable missing, set a default here.

# Set custom gradio temp dir
os.environ['GRADIO_TEMP_DIR'] = os.getenv('GRADIO_TEMP_DIR', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'temp'))

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

    # from webui.modules.implementations.gradio_monkeypatching import patch as patch2
    # patch2()
    #
    from webui.modules.implementations.huggingface_hub_monkeypatching import patch as patch3
    patch3()

    import torch
    print('Launching, cuda available:', torch.cuda.is_available())


    from webui.webui import launch_webui

    launch_webui()

except Exception as e:
    autodebug.catcher(e)
