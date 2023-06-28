import os.path
import re

import pytube
from pytube.exceptions import RegexMatchError


# Monkeypatched function to fix youtube downloads for now. (from https://github.com/pytube/pytube/pull/1691/files)
def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

        :param str js:
            The contents of the base.js asset file.
        :rtype: str
        :returns:
            The name of the function used to compute the throttling parameter.
        """
    function_patterns = [
        # https://github.com/ytdl-org/youtube-dl/issues/29326#issuecomment-865985377
        # https://github.com/yt-dlp/yt-dlp/commit/48416bc4a8f1d5ff07d5977659cb8ece7640dcd8
        # var Bpa = [iha];
        # ...
        # a.C && (b = a.get("n")) && (b = Bpa[0](b), a.set("n", b),
        # Bpa.length || iha("")) }};
        # In the above case, `iha` is the relevant function name
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&.*?\|\|\s*([a-z]+)',
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
    ]
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                        nfunc=re.escape(function_match.group(1))),
                    js
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(
        caller="get_throttling_function_name", pattern="multiple"
    )


pytube.cipher.get_throttling_function_name = get_throttling_function_name


def download_audio(url_type, url):
    if url_type == 'youtube':
        yt = pytube.YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path="./downloads/yt")
        filename, _ = os.path.splitext(out_file)
        ffmpeg_out_file = f'{filename}.wav'
        import ffmpeg
        (
            ffmpeg
            .input(out_file)
            .output(ffmpeg_out_file)
            .run(overwrite_output=1)
        )
        return ffmpeg_out_file
    return ''
