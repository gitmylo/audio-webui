from setup_tools.magicinstaller.requirement import Requirement, SimpleRequirement


class AudioLM(Requirement):
    def is_right_version(self):
        return self.get_package_version('audiolm-pytorch') == '1.1.4'

    def is_installed(self):
        return self.install_check('audiolm-pytorch')

    def install(self) -> tuple[int, str, str]:
        return self.install_pip('audiolm-pytorch==1.1.4', 'audiolm')


class JobLib(SimpleRequirement):
    package_name = 'joblib'


class FairSeq(SimpleRequirement):
    package_name = 'fairseq'
