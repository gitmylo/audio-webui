import re
import shlex
import subprocess
import sys
import time
from enum import Enum

import webui.args
from autodebug.autodebug import InstallFailException
from setup_tools.os import is_windows
from threading import Thread

valid_last: list[tuple[str, str]] = None


class CompareAction(Enum):
    LT = -2
    LEQ = -1
    EQ = 0
    GEQ = 1
    GT = 2


class Requirement:
    def __init__(self):
        self.running = False

    def install_or_upgrade_if_needed(self):
        if not self.is_installed() or not self.is_right_version():
            self.post_install(self.install())

    def post_install(self, install_output: tuple[int, str, str]):
        exit_code, stdout, stderr = install_output
        if exit_code != 0:
            raise InstallFailException(exit_code, stdout, stderr)

    def is_right_version(self):
        raise NotImplementedError('Not implemented')

    def is_installed(self):
        raise NotImplementedError('Not implemented')

    def install_check(self, package_name: str) -> bool:
        return self.get_package_version(package_name) is not False

    @staticmethod
    def loading_thread(status_dict, name):
        idx = 0
        load_symbols = ['|', '/', '-', '\\']
        while status_dict['running']:
            curr_symbol = load_symbols[idx % len(load_symbols)]
            idx += 1
            print(f'\rInstalling {name} {curr_symbol}', end='')
            time.sleep(0.25)
        print(f'\rInstalled {name}!             ' if status_dict[
            'success'] else f'\rFailed to install {name}. Check AutoDebug output.')

    def install_pip(self, command, name=None) -> tuple[int, str, str]:
        global valid_last
        valid_last = None
        if not name:
            name = command
        status_dict = {
            'running': True
        }

        verbose = webui.args.args.verbose

        if not verbose:
            thread = Thread(target=self.loading_thread, args=[status_dict, name], daemon=True)
            thread.start()
        args = f'"{sys.executable}" -m pip install --upgrade {command}'
        args = args if self.is_windows() else shlex.split(args)
        result = subprocess.run(args, capture_output=not verbose, text=True)
        status_dict['success'] = result.returncode == 0
        status_dict['running'] = False
        if not verbose:
            while thread.is_alive():
                time.sleep(0.1)
        return result.returncode, result.stdout, result.stderr

    def is_windows(self) -> bool:
        return is_windows()

    def install(self) -> tuple[int, str, str]:
        raise NotImplementedError('Not implemented')

    def pip_freeze(self) -> list[tuple[str, str]]:
        global valid_last
        if valid_last:
            return valid_last
        args = f'"{sys.executable}" -m pip freeze'
        args = args if self.is_windows() else shlex.split(args)
        result = subprocess.run(args, capture_output=True, text=True)
        test_str = result.stdout
        out_list = []
        matches = re.finditer('^(.*)(?:==| @ )(.+)$', test_str, re.MULTILINE)
        for match in matches:
            out_list.append((match.group(1), match.group(2)))

        valid_last = out_list
        return out_list

    def get_package_version(self, name: str, freeze: dict[tuple[str, str]] | None = None) -> bool | str:
        if freeze is None:
            freeze = self.pip_freeze()
        for p_name, version in freeze:
            if name.casefold() == p_name.casefold():
                return version
        return False


class SimpleRequirement(Requirement):
    package_name: str

    def is_right_version(self):
        return True

    def is_installed(self):
        return self.install_check(self.package_name)

    def install(self) -> tuple[int, str, str]:
        return self.install_pip(self.package_name)


class SimpleRequirementInit(SimpleRequirement):
    def __init__(self, package_name, compare: CompareAction = None, version: str = None):
        super().__init__()
        self.package_name = package_name
        self.compare = compare
        self.version = version

    def is_right_version(self):
        if self.compare is None or self.version is None:
            return True
        from packaging import version
        version_obj = version.parse(self.get_package_version(self.package_name))
        version_target_obj = version.parse(self.version)
        match self.compare:
            case CompareAction.LT:
                return version_obj < version_target_obj
            case CompareAction.LEQ:
                return version_obj <= version_target_obj
            case CompareAction.EQ:
                return version_obj == version_target_obj
            case CompareAction.GEQ:
                return version_obj >= version_target_obj
            case CompareAction.GT:
                return version_obj > version_target_obj

            case _:
                return True

    def install(self) -> tuple[int, str, str]:
        if self.version is None:
            return self.install_pip(self.package_name)
        match self.compare:
            case CompareAction.LT:
                symbol = '<'
            case CompareAction.LEQ:
                symbol = '<='
            case CompareAction.EQ:
                symbol = '=='
            case CompareAction.GEQ:
                symbol = '>='
            case CompareAction.GT:
                symbol = '>'
            case _:
                symbol = '=='

        return self.install_pip(f'{self.package_name}{symbol}{self.version}', self.package_name)


class SimpleGitRequirement(SimpleRequirement):
    def __init__(self, package_name, repo, check_version=False):
        super().__init__()
        self.package_name = package_name
        self.repo = repo
        self.check_version = check_version

    def is_right_version(self):
        if not self.check_version:
            return True
        return self.get_package_version(self.package_name) == self.repo

    def install(self) -> tuple[int, str, str]:
        return self.install_pip(self.repo, self.package_name)
