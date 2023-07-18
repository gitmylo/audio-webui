from setup_tools.magicinstaller.requirement import SimpleRequirement


class Praat(SimpleRequirement):
    package_name = 'praat-parselmouth'

    def is_right_version(self):
        from packaging import version
        return version.parse(self.get_package_version(self.package_name)) >= version.parse('0.4.2')

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('praat-parselmouth>=0.4.2', 'praat-parselmouth')


class PyWorld(SimpleRequirement):
    package_name = 'pyworld'

    def is_right_version(self):
        from packaging import version
        return version.parse(self.get_package_version(self.package_name)) >= version.parse('0.3.2')

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('pyworld>=0.3.2 --no-build-isolation', 'pyworld')


class FaissCpu(SimpleRequirement):
    package_name = 'faiss-cpu'

    def is_right_version(self):
        from packaging import version
        return version.parse(self.get_package_version(self.package_name)) == version.parse('1.7.3')

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('faiss-cpu==1.7.3', 'faiss')


class TorchCrepe(SimpleRequirement):
    package_name = 'torchcrepe'

    def is_right_version(self):
        from packaging import version
        return version.parse(self.get_package_version(self.package_name)) == version.parse('0.0.20')

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('torchcrepe==0.0.20', 'torchcrepe')


class FfmpegPython(SimpleRequirement):
    package_name = 'ffmpeg-python'


class NoiseReduce(SimpleRequirement):
    package_name = 'noisereduce'


class LibRosa(SimpleRequirement):
    package_name = 'librosa'


class Demucs(SimpleRequirement):
    package_name = 'demucs'

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('git+https://github.com/facebookresearch/demucs#egg=demucs', 'demucs')
