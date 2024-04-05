import json
import os
from typing import List, Union

import requests
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


def download_from_url(url: str, filename: str, local_dir: str) -> str:
    """
    Download a file from a URL using TQDM to show the progress.
    :param url: The URL to download the file from.
    :param filename: The name of the file to save.
    :param local_dir: The directory to save the file in.
    :return: The path to the downloaded file.
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    if not os.path.isfile(local_path):
        response = requests.get(url, stream=True)

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    return local_path


def get_model_path(
        model_url: str,
        model_name: str = None,
        model_type: str = None,
        single_file: bool = False,
        single_file_name: str = None,
        save_file_name: str = None,
        allow_patterns: Union[str, List[str]] = None,
        ignore_patterns: Union[str, List[str]] = None) -> str:
    """
    Get the model path from the model URL
    :param model_url: The URL of the model on the HF Hub
    :param model_name: The directory to store the model in the models folder, defaults to the model/creator name
    :param model_type: The type of model to download - this will be inserted into the model path
    :param single_file: Whether the model is a single file
    :param single_file_name: The name of the single file to download
    :param save_file_name: The name of the file to save
    :param allow_patterns: The patterns to allow for file downloads
    :param ignore_patterns: The patterns to ignore for file downloads
    :return: The model path
    """
    calls_file_path = os.path.join(os.path.dirname(__file__), 'all_models.json')
    try:
        with open(calls_file_path, 'r') as calls_file:
            model_calls = json.load(calls_file)
    except Exception:
        model_calls = {}

    # Check if the model_url is already logged, if not, log the call parameters
    model_key = model_url
    if single_file and single_file_name:
        model_key = f"{model_key}||{single_file_name}"
    if model_key not in model_calls:
        model_calls[model_key] = {
            'model_name': model_name,
            'model_type': model_type,
            'single_file': single_file,
            'single_file_name': single_file_name,
            'save_file_name': save_file_name,
            'allow_patterns': allow_patterns,
            'ignore_patterns': ignore_patterns
        }
        with open(calls_file_path, 'w') as calls_file:
            json.dump(model_calls, calls_file, indent=4)

    if "http" in model_url:
        model_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
        if model_type is not None:
            model_dir = os.path.join(model_dir, model_type)
        if model_name is not None:
            model_dir = os.path.join(model_dir, model_name)
    else:
        model_dev = model_url.split('/')[0]
        model_name = model_name or model_url.split('/')[1]
        if model_type is not None:
            model_dir = os.path.join(os.path.dirname(__file__), 'data', 'models', model_type, model_dev, model_name)
        else:
            model_dir = os.path.join(os.path.dirname(__file__), 'data', 'models', model_dev, model_name)
    if single_file and single_file_name:
        model_path = os.path.join(model_dir, single_file_name)
        if save_file_name:
            model_path = os.path.join(model_dir, save_file_name)
        do_download = not os.path.isfile(model_path)
    else:
        do_download = not os.path.exists(model_dir)
        model_path = model_dir

    # If the model doesn't exist, us HF Hub to download it
    if do_download:
        try:
            if single_file and single_file_name:
                print(f"Downloading {single_file_name} from {model_url}")
                if "http" in model_url:
                    dl_path = download_from_url(model_url, filename=single_file_name, local_dir=model_path)
                else:
                    dl_path = hf_hub_download(model_url, filename=single_file_name, local_dir=model_path,
                                              local_dir_use_symlinks=False)
                if dl_path != model_path:
                    temp_name = os.path.join(model_dir, f"{single_file_name}.tmp")
                    os.rename(dl_path, temp_name)
                    # If the dirname of dl_path is empty, remove it
                    if not os.listdir(os.path.dirname(dl_path)):
                        os.rmdir(os.path.dirname(dl_path))
                    os.rename(temp_name, model_path)
            else:
                print(f"Downloading model from {model_url}")
                snapshot_download(model_url, local_dir=model_path, local_dir_use_symlinks=False,
                                  allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
        except Exception as e:
            raise Exception(f"Failed to download model from {model_url}: {e}")

    return model_path
