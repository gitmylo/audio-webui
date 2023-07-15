import gradio
import webui.modules.implementations.whisper as w


def whisper():
    with gradio.Row():
        with gradio.Row():
            selected = gradio.Dropdown(w.get_official_models(), value='base', label='Model')
            with gradio.Column(elem_classes='smallsplit'):
                load = gradio.Button('🚀', variant='tool secondary')
                unload = gradio.Button('💣', variant='tool primary')

            def load_model(model):
                return w.load(model)
        with gradio.Row():
            transcribe = gradio.Button('Transcribe', variant='primary', elem_id='whisper-transcribe')
    with gradio.Row():
        with gradio.Column():
            audio = gradio.Audio(label='Audio to transcribe', elem_id='whisper-audio-in')
            audios = gradio.Files(label='Batch input', file_types=['audio'], elem_id='whisper-batch-in')
        with gradio.Column():
            output = gradio.TextArea(label='Transcript', elem_id='whisper-audio-out')
            outputs = gradio.Files(label='Batch output', file_types=['audio'], elem_id='whisper-batch-out')

    unload.click(fn=w.unload, outputs=output, show_progress=True)
    load.click(fn=load_model, inputs=selected, outputs=output, show_progress=True)

    transcribe.click(fn=w.transcribe, inputs=[audio, audios], outputs=[output, outputs])
