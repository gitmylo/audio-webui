import gradio
import webui.modules.implementations.audioldm as aldm


def generate(prompt, negative, duration, steps, cfg):
    output = aldm.generate(prompt, negative, steps, duration, cfg)
    if isinstance(output, str):
        return None, None, output
    else:
        return output, gradio.make_waveform(output), 'Successfully generated audio.'


def audioldm_tab():
    with gradio.Row():
        with gradio.Column():
            with gradio.Row():
                load_button = gradio.Button('Load model')
                status_box = gradio.Textbox('Unloaded AudioLDM.', show_label=False)
                unload_button = gradio.Button('Unload model', visible=False)

                def load():
                    aldm.create_model()
                    if aldm.is_loaded():
                        return 'Loaded AudioLDM.', gradio.update(visible=False), gradio.update(visible=True)
                    else:
                        return 'Failed loading AudioLDM.', gradio.update(visible=True), gradio.update(visible=False)

                def unload():
                    aldm.delete_model()
                    if aldm.is_loaded():
                        return 'Failed to unload AudioLDM.', gradio.update(visible=False), gradio.update(visible=True)
                    else:
                        return 'Unloaded AudioLDM.', gradio.update(visible=True), gradio.update(visible=False)

                load_button.click(fn=load, outputs=[status_box, load_button, unload_button], show_progress=True)
                unload_button.click(fn=unload, outputs=[status_box, load_button, unload_button], show_progress=True)
            prompt = gradio.TextArea(label='Prompt', info='Put the audio you want here.', placeholder='Techno music with a strong, upbeat tempo and high melodic riffs')
            neg_prompt = gradio.TextArea(label='Negative prompt', info='Put things to avoid generating here.', placeholder='low bitrate, low quality, bad quality')
            duration = gradio.Number(5, label='Duration (s)', info='Duration for the generation in seconds.')
            with gradio.Accordion('Extra options', open=False):
                with gradio.Row():
                    cfg = gradio.Slider(1, 20, 2.5, step=0.01, label='CFG scale', info='Default: 2.5. How much should the prompt affect the audio on every step?')
                    steps = gradio.Slider(1, 150, 10, step=1, label='Steps', info='Default: 10. How many diffusion steps should be performed?')

        with gradio.Column():
            gen_button = gradio.Button('Generate', variant='primary')
            audio_out = gradio.Audio(label='Generated audio')
            video_out = gradio.Video(label='Waveform video')
            text_out = gradio.Textbox(label='Result')

        gen_button.click(fn=generate, inputs=[prompt, neg_prompt, duration, steps, cfg], outputs=[audio_out, video_out, text_out])
