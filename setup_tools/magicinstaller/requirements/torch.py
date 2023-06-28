from setup_tools.magicinstaller.requirement import Requirement


class Torch(Requirement):
    def is_right_version(self):
        try:
            import torch.version
            ver = torch.version.__version__
            return ver.startswith('2') and '+cu' in ver  # Check if a CUDA version is installed
        except:
            return False

    def is_installed(self):
        return self.install_check('torch')

    def install(self):
        if self.is_windows():
            return self.install_pip('--upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117', 'PyTorch')
        else:
            return self.install_pip('--upgrade torch torchvision torchaudio', 'PyTorch')
