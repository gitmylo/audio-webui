import os.path
import urllib.request

from webui.modules.download import hub_download


class HuBERTManager:
    @staticmethod
    def make_sure_hubert_installed(download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt',
                                   file_name: str = 'hubert.pt'):
        app_root = os.getenv('APP_ROOT', None)
        if app_root is None:
            raise ValueError('APP_ROOT not set')
        install_dir = os.path.join(app_root, 'data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, file_name)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT base model')
            urllib.request.urlretrieve(download_url, install_file)
            print('Downloaded HuBERT')
        return install_file

    @staticmethod
    def make_sure_tokenizer_installed(model: str = 'quantifier_hubert_base_ls960_14.pth',
                                      repo: str = 'GitMylo/bark-voice-cloning', local_file: str = 'tokenizer.pth'):
        install_dir = 'data/models/hubert'
        app_root = os.getenv('APP_ROOT', None)
        if app_root is None:
            raise ValueError('APP_ROOT not set')
        install_file = os.path.join(app_root, install_dir, local_file)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT custom tokenizer')
            install_file = hub_download(repo, model,
                                        local_dir=install_dir)
            print('Downloaded tokenizer')
        return install_file

    @staticmethod
    def make_sure_hubert_rvc_installed(model: str = 'hubert_base.pt', repo: str = 'lj1995/VoiceConversionWebUI',
                                       local_file: str = 'hubert_rvc.pt'):
        app_root = os.getenv('APP_ROOT', None)
        if app_root is None:
            raise ValueError('APP_ROOT not set')
        install_dir = 'data/models/hubert'
        install_file = os.path.join(app_root, install_dir, local_file)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT for RVC')
            install_file = hub_download(repo, model,
                                        local_dir=install_dir)
            print('Downloaded HuBERT for RVC')
        return install_file
