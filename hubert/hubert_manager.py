import os.path
import shutil
import urllib.request

import huggingface_hub


class HuBERTManager:
    @staticmethod
    def make_sure_hubert_installed():
        install_dir = os.path.join('data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.mkdir(install_dir)
        install_file = os.path.join(install_dir, 'hubert.pt')
        download_url = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt'
        if not os.path.isfile(install_file):
            print('Downloading HuBERT small model')
            urllib.request.urlretrieve(download_url, install_file)
            print('Downloaded HuBERT')


    @staticmethod
    def make_sure_tokenizer_installed():
        install_dir = os.path.join('data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.mkdir(install_dir)
        install_file = os.path.join(install_dir, 'tokenizer.pth')
        repo = 'GitMylo/bark-voice-cloning'
        file = 'quantifier_hubert_base_ls960_14.pth'
        if not os.path.isfile(install_file):
            print('Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, file, local_dir=install_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(install_dir, file), install_file)
            print('Downloaded tokenizer')
