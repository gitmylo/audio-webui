import huggingface_hub.file_download as fd


original_check_disk_space = fd._check_disk_space


def _check_disk_space(expected_size, target_dir):
    """Calls the original, but with a try-catch to fix some crashes"""
    try:
        original_check_disk_space(expected_size, target_dir)
    except Exception as _:
        pass


def patch():
    fd._check_disk_space = _check_disk_space
