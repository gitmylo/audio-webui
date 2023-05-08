import numpy as np
import gradio
import torch
import torchaudio
from bark.generation import SAMPLE_RATE, load_codec_model, codec_decode
from encodec import EncodecModel
from encodec.utils import convert_audio

model: EncodecModel = load_codec_model()


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    # Modified to support in64
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    print('Converting', data.dtype)
    if data.dtype in [np.float64, np.float32, np.float16]:
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int64:
        data = data / 4295229444
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data


def file_to_audio(file):
    if file.name.endswith('.npz'):
        html = '<h1>Result</h1>'
        try:
            data = np.load(file.name)
            for dpart in data.keys():
                data_content = data[dpart]
                html += f'File name: "{dpart}"<br>' \
                        f'Shape: {data_content.shape}<br>' \
                        f'Dtype: {data_content.dtype}'
                html += '<br><br>'
            audio_arr = codec_decode(data['fine_prompt'])
        except Exception as e:
            return gradio.update(), f'<h1 style="color: red;">Error</h1>{str(e)}'
        return (SAMPLE_RATE, audio_arr), html
    elif file.name.endswith('.wav'):
        wav, sr = torchaudio.load(file.name)
        wav_pre_convert_shape = wav.shape
        wav = convert_audio(wav, sr, SAMPLE_RATE, model.channels)
        wav_post_convert_shape = wav.shape
        wav = wav.unsqueeze(0).to('cuda')
        wav_unsqueezed_shape = wav.shape
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
        codes_shape = codes.shape

        seconds = wav.shape[-1] / model.sample_rate

        # codes = codes.cpu().numpy()
        return (SAMPLE_RATE, wav.cpu().squeeze().numpy()), f'Seconds: {seconds}<br>' \
                                                           f'Pre convert shape: {wav_pre_convert_shape}<br>' \
                                                           f'Post convert shape: {wav_post_convert_shape}<br>' \
                                                           f'Wav unsqueezed shape: {wav_unsqueezed_shape}<br>' \
                                                           f'Codes shape: {codes_shape}<br>'


gradio.interface.Interface(fn=file_to_audio, inputs='file', outputs=['audio', 'html']).launch()
