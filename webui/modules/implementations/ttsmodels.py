import gc
import os.path
import tempfile

import gradio
import numpy
import numpy as np
import scipy.io.wavfile

import webui.modules.models as mod
from webui.modules.implementations.patches.bark_custom_voices import wav_to_semantics, generate_fine_from_wav, \
    generate_course_history


class BarkTTS(mod.TTSModelLoader):
    no_install = True

    @staticmethod
    def get_voices():
        found_prompts = []
        base_path = 'data/bark_custom_speakers/'
        for path, subdirs, files in os.walk(base_path):
            for name in files:
                if name.endswith('.npz'):
                    found_prompts.append(os.path.join(path, name)[len(base_path):-4])
        from webui.modules.implementations.patches.bark_generation import ALLOWED_PROMPTS
        return ['None'] + found_prompts + ALLOWED_PROMPTS

    @staticmethod
    def create_voice(file):
        file_name = '.'.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        out_file = f'data/bark_custom_speakers/{file_name}.npz'

        semantic_prompt = wav_to_semantics(file)
        fine_prompt = generate_fine_from_wav(file)
        coarse_prompt = generate_course_history(fine_prompt)


        np.savez(out_file,
                 semantic_prompt=semantic_prompt,
                 fine_prompt=fine_prompt,
                 coarse_prompt=coarse_prompt
                 )
        return file_name

    def _components(self, **quick_kwargs):
        def update_speaker(option):
            if option == 'File':
                speaker.hide = True
                refresh_speakers.hide = True
                speaker_file.hide = False
                speaker_name.hide = False
                return [gradio.update(visible=False), gradio.update(visible=False), gradio.update(visible=True), gradio.update(visible=True)]
            else:
                speaker.hide = False
                refresh_speakers.hide = False
                speaker_file.hide = True
                speaker_name.hide = True
                return [gradio.update(visible=True), gradio.update(visible=True), gradio.update(visible=False), gradio.update(visible=False)]

        def update_input(option):
            if option == 'Text':
                textbox.hide = False
                audio_upload.hide = True
                return [gradio.update(visible=False), gradio.update(visible=True)]
            else:
                textbox.hide = True
                audio_upload.hide = False
                return [gradio.update(visible=True), gradio.update(visible=False)]

        def update_voices():
            return gradio.update(choices=self.get_voices())

        clone_guide = gradio.Markdown('''
        ## Long form generations
        Split your long form generations with newlines (enter), every line will be generated individually, but as a continuation of the last.

        Empty lines at the start and end will be skipped.

        ## When cloning a voice:
        * The speaker will be saved in the data/bark_custom_speakers directory.
        * The "file" output contains a different speaker. This is for saving speakers created through random generation. Or continued cloning.

        ## Cloning guide (short edition)
        * Clear spoken, no noise, no music.
        * Ends after a short pause for best results.
                ''', visible=False)

        input_type = gradio.Radio(['Text', 'File'], label='Input type', value='Text', **quick_kwargs)
        textbox = gradio.Textbox(lines=7, label='Input', placeholder='Text to speak goes here', **quick_kwargs)
        gen_prefix = gradio.Textbox(label='Generation prefix', info='Add this text before every generated chunk, better for keeping emotions.', **quick_kwargs)
        audio_upload = gradio.File(label='Words to speak', file_types=['audio'], **quick_kwargs)
        audio_upload.hide = True
        with gradio.Row(visible=False) as temps:
            text_temp = gradio.Slider(0.05, 1, 0.7, step=0.05, label='Text temperature', **quick_kwargs)
            waveform_temp = gradio.Slider(0.05, 1, 0.7, step=0.05, label='Waveform temperature', **quick_kwargs)
        mode = gradio.Radio(['File', 'Upload'], label='Speaker from', value='File', **quick_kwargs)
        with gradio.Row(visible=False) as speakers:
            speaker = gradio.Dropdown(self.get_voices(), value='None', show_label=False, **quick_kwargs)
            refresh_speakers = gradio.Button('🔃', variant='tool secondary', **quick_kwargs)
        refresh_speakers.click(fn=update_voices, outputs=speaker)
        speaker_name = gradio.Textbox(label='Speaker name', info='The name to save the speaker as, random if empty', **quick_kwargs)
        speaker_file = gradio.Audio(label='Speaker', **quick_kwargs)
        # speaker_file_transcript = gradio.Textbox(lines=1, label='Transcript', **quick_kwargs)
        speaker_name.hide = True
        speaker_file.hide = True  # Custom, auto hide speaker_file
        # speaker_file_transcript.hide = True

        keep_generating = gradio.Checkbox(label='Keep it up (keep generating)', value=False, **quick_kwargs)
        min_eos_p = gradio.Slider(0.05, 1, 0.2, step=0.05, label='min end of audio probability', info='Lower values cause the generation to stop sooner, higher values make it do more, 1 is about the same as keep generating being on.', **quick_kwargs)

        mode.select(fn=update_speaker, inputs=mode, outputs=[speaker, refresh_speakers, speaker_file, speaker_name])
        input_type.select(fn=update_input, inputs=input_type, outputs=[textbox, audio_upload])
        return [textbox, gen_prefix, audio_upload, input_type, mode, text_temp, waveform_temp,
                speaker, speaker_name, speaker_file, refresh_speakers, keep_generating, clone_guide, temps, speakers, min_eos_p]

    model = 'suno/bark'

    def get_response(self, *inputs):
        textbox, gen_prefix, audio_upload, input_type, mode, text_temp, waveform_temp, speaker,\
            speaker_name, speaker_file, refresh_speakers, keep_generating, clone_guide, min_eos_p = inputs
        _speaker = None
        if mode == 'File':
            _speaker = speaker if speaker != 'None' else None
        else:
            speaker_sr, speaker_wav = speaker_file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            if speaker_name:
                temp_file.name = os.path.join(os.path.dirname(temp_file.name), speaker_name + '.wav')
            scipy.io.wavfile.write(temp_file.name, speaker_sr, speaker_wav)
            _speaker = self.create_voice(temp_file.name)
        from webui.modules.implementations.patches.bark_api import generate_audio_new, semantic_to_waveform_new
        from bark.generation import SAMPLE_RATE
        if input_type == 'Text':
            history_prompt, audio = generate_audio_new(textbox, _speaker, text_temp, waveform_temp, output_full=True,
                                                       allow_early_stop=not keep_generating, min_eos_p=min_eos_p,
                                                       gen_prefix=gen_prefix)
        else:
            semantics = wav_to_semantics(audio_upload.name).numpy()
            history_prompt, audio = semantic_to_waveform_new(semantics, _speaker, waveform_temp, output_full=True)
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.name = temp.name.replace(temp.name.replace('\\', '/').split('/')[-1], 'speaker.npz')
        numpy.savez(temp.name, **history_prompt)
        return (SAMPLE_RATE, audio), temp.name

    def unload_model(self):
        # from bark.generation import clean_models
        # clean_models()

        # Temp fix while i wait for https://github.com/suno-ai/bark/pull/356
        import bark.generation as bark_gen
        model_keys = list(bark_gen.models.keys())
        for k in model_keys:
            if k in bark_gen.models:
                del bark_gen.models[k]
        bark_gen._clear_cuda_cache()
        gc.collect()

    def load_model(self):
        from bark.generation import preload_models
        from webui.args import args
        cpu = args.bark_use_cpu
        gpu = not cpu
        low_vram = args.bark_low_vram
        preload_models(
            text_use_gpu=gpu,
            fine_use_gpu=gpu,
            coarse_use_gpu=gpu,
            codec_use_gpu=gpu,
            fine_use_small=low_vram,
            coarse_use_small=low_vram,
            text_use_small=low_vram
        )


elements = []


def init_elements():
    global elements
    elements = [BarkTTS()]


def all_tts() -> list[mod.TTSModelLoader]:
    if not elements:
        init_elements()
    return elements


def all_elements(in_dict):
    l = []
    for value in in_dict.values():
        l += value
    return l


def all_elements_dict():
    d = {}
    for tts in all_tts():
        d[tts.model] = tts.gradio_components()
    return d
