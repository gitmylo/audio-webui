import setup_tools.os as oscheck
import setup_tools.commands as commands


class AutoDebugException(OSError):
    def __init__(self, message):
        super(AutoDebugException, self).__init__(message)

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


class WrongPythonVersionException(AutoDebugException):
    pass


def catcher(e: Exception):
    if isinstance(e, AutoDebugException):
        print('''
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
█░▄▄▀██░██░█▄▄░▄▄██░▄▄▄░██░▄▄▀██░▄▄▄██░▄▄▀██░██░██░▄▄░██
█░▀▀░██░██░███░████░███░██░██░██░▄▄▄██░▄▄▀██░██░██░█▀▀██
█░██░██▄▀▀▄███░████░▀▀▀░██░▀▀░██░▀▀▀██░▀▀░██▄▀▀▄██░▀▀▄██
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀''')
        print(e)
        e.action()
    else:
        raise e
