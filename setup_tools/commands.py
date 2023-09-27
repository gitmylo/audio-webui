import sys
from os import system

from setup_tools.os import is_windows


def get_python():
    return sys.executable


def run_command(command: list[tuple[str, str]] | str, args='', show_output=True):
    extra = (' >nul' if is_windows() else ' >/dev/null') if not show_output else ''
    if not isinstance(command, str):
        commandstr = '&&'.join([' '.join(cmd) + extra for cmd in command])
    else:
        commandstr = f'{command} {args}' + extra
    
    # Ensure bin/bash shell
    if not is_windows():
        commandstr = f'/usr/bin/env bash -c "{commandstr}"'
        
    system(commandstr)


def run_pip(package, show_output=False):
    run_command('pip', f'install {package}', show_output)
