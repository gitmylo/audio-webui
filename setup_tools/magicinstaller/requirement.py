import subprocess
import sys
import time

from setup_tools.os import is_windows
from threading import Thread


class Requirement:
    def __init__(self):
        self.running = False

    def install_or_upgrade_if_needed(self):
        if not self.is_installed() or not self.is_right_version():
            self.post_install(self.install())


    def post_install(self, install_output: tuple[int, str, str]):
        exit_code, stdout, stderr = install_output
        if exit_code != 0:
            print('Exit code as not 0.')

    def is_right_version(self):
        raise NotImplementedError('Not implemented')

    def is_installed(self):
        raise NotImplementedError('Not implemented')

    def install_check(self, package_name: str) -> bool:
        try:
            __import__(package_name)
            return True
        except:
            return False

    @staticmethod
    def loading_thread(status_dict, name):
        idx = 0
        load_symbols = ['|', '/', '-', '\\']
        while status_dict['running']:
            curr_symbol = load_symbols[idx % len(load_symbols)]
            idx += 1
            print(f'\rInstalling {name} {curr_symbol}', end='')
            time.sleep(0.25)

    def install_pip(self, command, name='package') -> tuple[int, str, str]:
        status_dict = {
            'running': True
        }
        thread = Thread(target=self.loading_thread, args=[status_dict, name], daemon=True)
        thread.start()
        result = subprocess.run(f'{sys.executable} -m pip install {command}', capture_output=True, text=True)
        status_dict['running'] = False
        while thread.is_alive():
            time.sleep(0.1)
        print()
        return result.returncode, result.stdout, result.stderr

    def is_windows(self) -> bool:
        return is_windows()

    def install(self) -> tuple[int, str, str]:
        raise NotImplementedError('Not implemented')
