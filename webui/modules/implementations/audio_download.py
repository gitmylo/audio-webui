import os.path
import pytube


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
