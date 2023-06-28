import sys
import traceback

import setup_tools.os as oscheck
import setup_tools.commands as commands


class AutoDebugException(OSError):
    def __init__(self, message):
        super(AutoDebugException, self).__init__(message)

    def action(self):
        pass


class WrongPythonVersionException(AutoDebugException):
    def action(self):
        if oscheck.is_windows():
            print('Do you want to download the python 3.10 installer?\nWhen installing, make sure py launcher is selected as well.')
            response = input('Y/n: ').upper()
            if not response:
                response = 'Y'
            response = response[0]
            if response == 'Y':
                commands.run_command('start', 'https://www.python.org/downloads/release/python-31011/')  # Last python 3.10 version with windows installer
            elif response == 'N':
                print('If you ever want to download it, the link is: https://www.python.org/downloads/release/python-31011/')
        else:
            print('Please use a package manager to install python 3.10. For example: `apt install python3.10` on debian.')
        input()


class InstallFailException(AutoDebugException):
    def __init__(self, exit_code, stdout, stderr):
        super(InstallFailException, self).__init__('Install failed!')
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def action(self):
        print(f'STDOUT:\n{self.stdout}\n\n\n\nSTDERR:\n{self.stderr}\n\n')
        print('Please read the error above carefully. It might tell you to install visual C++ build tools, it might tell you something else.\nIf you are unsure, please create an issue at https://github.com/gitmylo/audio-webui/issues.')
        print('When creating an issue, please include your full autodebug message.')
        print(f'Exit code: {self.exit_code}')
        input()


def print_banner():
    print('''
    ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
    █░▄▄▀██░██░█▄▄░▄▄██░▄▄▄░██░▄▄▀██░▄▄▄██░▄▄▀██░██░██░▄▄░██
    █░▀▀░██░██░███░████░███░██░██░██░▄▄▄██░▄▄▀██░██░██░█▀▀██
    █░██░██▄▀▀▄███░████░▀▀▀░██░▀▀░██░▀▀▀██░▀▀░██▄▀▀▄██░▀▀▄██
    ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀''')


def catcher(e: Exception):
    if isinstance(e, AutoDebugException):
        print_banner()
        print(e)
        e.action()
    elif isinstance(e, ImportError):
        if 'No module named \'pywintypes\''.casefold() in str(e).casefold():
            print('Install finished, pywintypes missing, relaunching script.')
            commands.run_command(sys.executable, sys.argv[0])
            return
        traceback.print_exception(e)
        print(e)
        print('Your install might have failed to install one of the requirements, are you missing a package?')
        print('Depending on the error message that was given during install, you might need to install visual C++ build tools.')
        print('Or read common issues at https://github.com/gitmylo/audio-webui/wiki/common-issues')
        input()
    else:
        raise e
