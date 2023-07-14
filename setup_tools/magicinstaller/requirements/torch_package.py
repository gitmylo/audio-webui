from setup_tools.magicinstaller.requirement import Requirement


class Torch(Requirement):
    def is_right_version(self):
        ver = self.get_package_version('torch')
        if ver:
            # Check if a CUDA version is installed
            return ver.startswith('2') and ('+cu' in ver if self.is_windows() else True)
        return False

    def is_installed(self):
        return self.install_check('torch')

    def install(self):
        if self.is_windows():
            return self.install_pip('torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117', 'PyTorch')
        else:
            return self.install_pip('torch torchvision torchaudio', 'PyTorch')
