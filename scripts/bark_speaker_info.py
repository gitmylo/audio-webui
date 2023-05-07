import numpy as np
import gradio


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
    html = '<h1>Result</h1>'
    try:
        data = np.load(file.name)
        for dpart in data.keys():
            data_content = data[dpart]
            html += f'File name: "{dpart}"<br>' \
                    f'Shape: {data_content.shape}<br>' \
                    f'Dtype: {data_content.dtype}'
            html += '<br><br>'
    except Exception as e:
        return f'<h1 style="color: red;">Error</h1>{e}'
    return html


gradio.interface.Interface(fn=file_to_audio, inputs='file', outputs='html').launch()
