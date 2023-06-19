import gc
import os.path
import tempfile

import gradio
import numpy
import numpy as np
import scipy.io.wavfile
import torch.cuda
from TTS.api import TTS

import webui.modules.models as mod
from webui.args import args
from webui.modules import util
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

    def load_model(self, progress=gradio.Progress()):
        from webui.modules.implementations.patches.bark_generation import preload_models_new
        from webui.args import args
        cpu = args.bark_use_cpu
        gpu = not cpu
        low_vram = args.bark_low_vram
        preload_models_new(
            text_use_gpu=gpu,
            fine_use_gpu=gpu,
            coarse_use_gpu=gpu,
            codec_use_gpu=gpu,
            fine_use_small=low_vram,
            coarse_use_small=low_vram,
            text_use_small=low_vram,
            progress=progress
        )


class CoquiTTS(mod.TTSModelLoader):
    no_install = True
    model = 'Coqui TTS'

    current_model: TTS = None
    current_model_name: str = None

    def load_model(self, progress=gradio.Progress()):
        pass

    def unload_model(self):
        self.current_model_name = None
        self.current_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def tts_speakers(self):
        if self.current_model is None:
            return gradio.update(choices=[]), gradio.update(choices=[])
        speakers = list(
            dict.fromkeys([speaker.strip() for speaker in self.current_model.speakers])) if self.current_model.is_multi_speaker else []
        languages = list(dict.fromkeys(self.current_model.languages)) if self.current_model.is_multi_lingual else []
        return gradio.update(choices=speakers), gradio.update(choices=languages)

    def _components(self, **quick_kwargs):
        with gradio.Row(visible=False) as r1:
            selected_tts = gradio.Dropdown(TTS.list_models(), label='TTS model', info='The TTS model to use for text-to-speech',
                                           allow_custom_value=True, **quick_kwargs)
            selected_tts_unload = gradio.Button('💣', variant='primary tool offset--10', **quick_kwargs)

        with gradio.Row(visible=False) as r2:
            speaker_tts = gradio.Dropdown(self.tts_speakers()[0]['choices'], label='TTS speaker',
                                          info='The speaker to use for the TTS model, only for multi speaker models.', **quick_kwargs)
            speaker_tts_refresh = gradio.Button('🔃', variant='primary tool offset--10', **quick_kwargs)

        with gradio.Row(visible=False) as r3:
            lang_tts = gradio.Dropdown(self.tts_speakers()[1]['choices'], label='TTS language',
                                       info='The language to use for the TTS model, only for multilingual models.', **quick_kwargs)
            lang_tts_refresh = gradio.Button('🔃', variant='primary tool offset--10', **quick_kwargs)

        speaker_tts_refresh.click(fn=self.tts_speakers, outputs=[speaker_tts, lang_tts])
        lang_tts_refresh.click(fn=self.tts_speakers, outputs=[speaker_tts, lang_tts])

        def load_tts(model):
            if self.current_model_name != model:
                unload_tts()
                self.current_model_name = model
                self.current_model = TTS(model, gpu=True if torch.cuda.is_available() and args.tts_use_gpu else False)
            return gradio.update(value=model), *self.tts_speakers()

        def unload_tts():
            if self.current_model is not None:
                self.current_model = None
                self.current_model_name = None
                gc.collect()
                torch.cuda.empty_cache()
            return gradio.update(value=''), *self.tts_speakers()

        selected_tts_unload.click(fn=unload_tts, outputs=[selected_tts, speaker_tts, lang_tts])
        selected_tts.select(fn=load_tts, inputs=selected_tts, outputs=[selected_tts, speaker_tts, lang_tts])

        text_input = gradio.TextArea(label='Text to speech text',
                                     info='Text to speech text if no audio file is used as input.', **quick_kwargs)

        return selected_tts, selected_tts_unload, speaker_tts, speaker_tts_refresh, lang_tts, lang_tts_refresh, text_input, r1, r2, r3



    def get_response(self, *inputs):
        selected_tts, selected_tts_unload, speaker_tts, speaker_tts_refresh, lang_tts, lang_tts_refresh, text_input = inputs
        if self.current_model_name != selected_tts:
            if self.current_model is not None:
                self.current_model = None
                self.current_model_name = None
                gc.collect()
                torch.cuda.empty_cache()
            self.current_model_name = selected_tts
            self.current_model = TTS(selected_tts, gpu=True if torch.cuda.is_available() and args.tts_use_gpu else False)
        audio = np.array(self.current_model.tts(text_input, speaker_tts if self.current_model.is_multi_speaker else None, lang_tts if self.current_model.is_multi_lingual else None))
        audio_tuple = (self.current_model.synthesizer.output_sample_rate, audio)
        return audio_tuple, None


elements = []


def init_elements():
    global elements
    elements = [BarkTTS(), CoquiTTS()]


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
