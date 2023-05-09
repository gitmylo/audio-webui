import importlib
from typing import List, Dict

from TTS.api import *
from TTS.utils.synthesizer import *
from TTS.utils.generic_utils import find_module


class Synthesizer_new(object):
    def __init__(
        self,
        tts_checkpoint: str = "",
        tts_config_path: str = "",
        tts_speakers_file: str = "",
        tts_languages_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        vc_checkpoint: str = "",
        vc_config: str = "",
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str, optional): path to the tts model file.
            tts_config_path (str, optional): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            vc_checkpoint (str, optional): path to the voice conversion model file. Defaults to `""`,
            vc_config (str, optional): path to the voice conversion config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.tts_languages_file = tts_languages_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.vc_checkpoint = vc_checkpoint
        self.vc_config = vc_config
        self.use_cuda = use_cuda

        self.tts_model = None
        self.vocoder_model = None
        self.vc_model = None
        self.speaker_manager = None
        self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        if tts_checkpoint:
            self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
            self.output_sample_rate = self.tts_config.audio["sample_rate"]

        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

        if vc_checkpoint:
            self._load_vc(vc_checkpoint, vc_config, use_cuda)
            if 'output_sample_rate' in self.vc_config.audio.keys():
                self.output_sample_rate = self.vc_config.audio["output_sample_rate"]
            else:
                self.output_sample_rate = self.vc_config.audio['sample_rate']

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_vc(self, vc_checkpoint: str, vc_config_path: str, use_cuda: bool) -> None:
        """Load the voice conversion model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.

        Args:
            vc_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.vc_config = load_config(vc_config_path)
        self.vc_model = setup_vc_model_new(config=self.vc_config)
        self.vc_model.load_checkpoint(self.vc_config, vc_checkpoint)
        if use_cuda:
            self.vc_model.cuda()

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        """Load the TTS model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")

        self.tts_model = setup_tts_model(config=self.tts_config)

        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)

    def _set_speaker_encoder_paths_from_tts_config(self):
        """Set the encoder paths from the tts model config for models with speaker encoders."""
        if hasattr(self.tts_config, "model_args") and hasattr(
            self.tts_config.model_args, "speaker_encoder_config_path"
        ):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        """Load the vocoder model.

        1. Load the vocoder config.
        2. Init the AudioProcessor for the vocoder.
        3. Init the vocoder model from the config.
        4. Move the model to the GPU if CUDA is enabled.

        Args:
            model_file (str): path to the model checkpoint.
            model_config (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def split_into_sentences(self, text) -> List[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.

        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate)

    def voice_conversion(self, source_wav: str, target_wav: str) -> List[int]:
        if self.vc_config.base_model.lower() == 'vits':
            target_audio = self.vc_model.wav_to_spec(target_wav,
                                                     self.vc_model.config.audio.fft_size,
                                                     self.vc_model.config.audio.hop_length,
                                                     self.vc_model.config.audio.win_length)
            output_wav, _, _ = self.vc_model.inference_voice_conversion(reference_wav=source_wav, speaker_id=, )
        else:
            output_wav = self.vc_model.voice_conversion(source_wav, target_wav)
        return output_wav

    def tts(
        self,
        text: str = "",
        speaker_name: str = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        reference_wav=None,
        reference_speaker_name=None,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): spekaer id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): spekaer id of reference waveform. Defaults to None.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if not text and not reference_wav:
            raise ValueError(
                "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
            )

        if text:
            sens = self.split_into_sentences(text)
            print(" > Text splitted to sentences.")
            print(sens)

        # handle multi-speaker
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
            # handle Neon models with single speaker.
            if len(self.tts_model.speaker_manager.name_to_id) == 1:
                speaker_id = list(self.tts_model.speaker_manager.name_to_id.values())[0]

            elif speaker_name and isinstance(speaker_name, str):
                if self.tts_config.use_d_vector_file:
                    # get the average speaker embedding from the saved d_vectors.
                    speaker_embedding = self.tts_model.speaker_manager.get_mean_embedding(
                        speaker_name, num_samples=None, randomize=False
                    )
                    speaker_embedding = np.array(speaker_embedding)[None, :]  # [1 x embedding_dim]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]

            elif not speaker_name and not speaker_wav:
                raise ValueError(
                    " [!] Look like you use a multi-speaker model. "
                    "You need to define either a `speaker_name` or a `speaker_wav` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_name:
                raise ValueError(
                    f" [!] Missing speakers.json file path for selecting speaker {speaker_name}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        # handle multi-lingual
        language_id = None
        if self.tts_languages_file or (
            hasattr(self.tts_model, "language_manager") and self.tts_model.language_manager is not None
        ):
            if len(self.tts_model.language_manager.name_to_id) == 1:
                language_id = list(self.tts_model.language_manager.name_to_id.values())[0]

            elif language_name and isinstance(language_name, str):
                try:
                    language_id = self.tts_model.language_manager.name_to_id[language_name]
                except KeyError as e:
                    raise ValueError(
                        f" [!] Looks like you use a multi-lingual model. "
                        f"Language {language_name} is not in the available languages: "
                        f"{self.tts_model.language_manager.name_to_id.keys()}."
                    ) from e

            elif not language_name:
                raise ValueError(
                    " [!] Look like you use a multi-lingual model. "
                    "You need to define either a `language_name` or a `style_wav` to use a multi-lingual model."
                )

            else:
                raise ValueError(
                    f" [!] Missing language_ids.json file path for selecting language {language_name}."
                    "Define path for language_ids.json if it is a multi-lingual model or remove defined language idx. "
                )

        # compute a new d_vector from the given clip.
        if speaker_wav is not None:
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

        use_gl = self.vocoder_model is None

        if not reference_wav:
            for sen in sens:
                # synthesize voice
                outputs = synthesis(
                    model=self.tts_model,
                    text=sen,
                    CONFIG=self.tts_config,
                    use_cuda=self.use_cuda,
                    speaker_id=speaker_id,
                    style_wav=style_wav,
                    style_text=style_text,
                    use_griffin_lim=use_gl,
                    d_vector=speaker_embedding,
                    language_id=language_id,
                )
                waveform = outputs["wav"]
                mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
                if not use_gl:
                    # denormalize tts output based on tts audio config
                    mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                    device_type = "cuda" if self.use_cuda else "cpu"
                    # renormalize spectrogram based on vocoder config
                    vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    # compute scale factor for possible sample rate mismatch
                    scale_factor = [
                        1,
                        self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                    ]
                    if scale_factor[1] != 1:
                        print(" > interpolating tts model output.")
                        vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                    else:
                        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                    # run vocoder model
                    # [1, T, C]
                    waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
                if self.use_cuda and not use_gl:
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()

                # trim silence
                if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
                    waveform = trim_silence(waveform, self.tts_model.ap)

                wavs += list(waveform)
                wavs += [0] * 10000
        else:
            # get the speaker embedding or speaker id for the reference wav file
            reference_speaker_embedding = None
            reference_speaker_id = None
            if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
                if reference_speaker_name and isinstance(reference_speaker_name, str):
                    if self.tts_config.use_d_vector_file:
                        # get the speaker embedding from the saved d_vectors.
                        reference_speaker_embedding = self.tts_model.speaker_manager.get_embeddings_by_name(
                            reference_speaker_name
                        )[0]
                        reference_speaker_embedding = np.array(reference_speaker_embedding)[
                            None, :
                        ]  # [1 x embedding_dim]
                    else:
                        # get speaker idx from the speaker name
                        reference_speaker_id = self.tts_model.speaker_manager.name_to_id[reference_speaker_name]
                else:
                    reference_speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(
                        reference_wav
                    )
            outputs = transfer_voice(
                model=self.tts_model,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                reference_wav=reference_wav,
                speaker_id=speaker_id,
                d_vector=speaker_embedding,
                use_griffin_lim=use_gl,
                reference_speaker_id=reference_speaker_id,
                reference_d_vector=reference_speaker_embedding,
            )
            waveform = outputs
            if not use_gl:
                mel_postnet_spec = outputs[0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            if self.use_cuda:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            wavs = waveform.squeeze()

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs



class TTS:
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        config_path: str = None,
        vocoder_path: str = None,
        vocoder_config_path: str = None,
        progress_bar: bool = True,
        gpu=False,
        force_voice_convert=False
    ):
        """🐸TTS python interface that allows to load and use the released models.

        Example with a multi-speaker model:
            >>> from TTS.api import TTS
            >>> tts = TTS(TTS.list_models()[0])
            >>> wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
            >>> tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

        Example with a single-speaker model:
            >>> tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example loading a model from a path:
            >>> tts = TTS(model_path="/path/to/checkpoint_100000.pth", config_path="/path/to/config.json", progress_bar=False, gpu=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example voice cloning with YourTTS in English, French and Portuguese:
            >>> tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
            >>> tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="thisisit.wav")
            >>> tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr", file_path="thisisit.wav")
            >>> tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt", file_path="thisisit.wav")

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to None.
            config_path (str, optional): Path to the model config. Defaults to None.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
            progress_bar (bool, optional): Whether to pring a progress bar while downloading a model. Defaults to True.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar, verbose=False)

        self.synthesizer = None
        self.voice_converter = None
        self.csapi = None
        self.model_name = None

        if model_name is not None:
            if ("tts_models" in model_name or "coqui_studio" in model_name) and not force_voice_convert:
                self.load_tts_model_by_name(model_name, gpu)
            elif "voice_conversion_models" in model_name or force_voice_convert:
                self.load_vc_model_by_name(model_name, gpu)

        if model_path:
            self.load_tts_model_by_path(
                model_path, config_path, vocoder_path=vocoder_path, vocoder_config=vocoder_config_path, gpu=gpu
            )

    @property
    def models(self):
        return self.manager.list_tts_models()

    @property
    def is_multi_speaker(self):
        if hasattr(self.synthesizer.tts_model, "speaker_manager") and self.synthesizer.tts_model.speaker_manager:
            return self.synthesizer.tts_model.speaker_manager.num_speakers > 1
        return False

    @property
    def is_coqui_studio(self):
        if self.model_name is None:
            return False
        return "coqui_studio" in self.model_name

    @property
    def is_multi_lingual(self):
        if hasattr(self.synthesizer.tts_model, "language_manager") and self.synthesizer.tts_model.language_manager:
            return self.synthesizer.tts_model.language_manager.num_languages > 1
        return False

    @property
    def speakers(self):
        if not self.is_multi_speaker:
            return None
        return self.synthesizer.tts_model.speaker_manager.speaker_names

    @property
    def languages(self):
        if not self.is_multi_lingual:
            return None
        return self.synthesizer.tts_model.language_manager.language_names

    @staticmethod
    def get_models_file_path():
        import TTS.api as a
        return Path(a.__file__).parent / ".models.json"

    @staticmethod
    def list_models():
        try:
            csapi = CS_API()
            models = csapi.list_speakers_as_tts_models()
        except ValueError as e:
            print(e)
            models = []
        manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False)
        return manager.list_tts_models() + models

    def download_model_by_name(self, model_name: str):
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if model_item.get("default_vocoder") is None:
            return model_path, config_path, None, None
        vocoder_path, vocoder_config_path, _ = self.manager.download_model(model_item["default_vocoder"])
        return model_path, config_path, vocoder_path, vocoder_config_path

    def load_vc_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the voice conversion models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.model_name = model_name
        model_path, config_path, _, _ = self.download_model_by_name(model_name)
        self.voice_converter = Synthesizer_new(vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

    def load_tts_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.

        TODO: Add tests
        """
        self.synthesizer = None
        self.csapi = None
        self.model_name = model_name

        if "coqui_studio" in model_name:
            self.csapi = CS_API()
        else:
            model_path, config_path, vocoder_path, vocoder_config_path = self.download_model_by_name(model_name)

            # init synthesizer
            # None values are fetch from the model
            self.synthesizer = Synthesizer_new(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                tts_speakers_file=None,
                tts_languages_file=None,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=vocoder_config_path,
                encoder_checkpoint=None,
                encoder_config=None,
                use_cuda=gpu,
            )

    def load_tts_model_by_path(
        self, model_path: str, config_path: str, vocoder_path: str = None, vocoder_config: str = None, gpu: bool = False
    ):
        """Load a model from a path.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the model config.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config (str, optional): Path to the vocoder config. Defaults to None.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """

        self.synthesizer = Synthesizer_new(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config,
            encoder_checkpoint=None,
            encoder_config=None,
            use_cuda=gpu,
        )

    def _check_arguments(
        self,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
    ) -> None:
        """Check if the arguments are valid for the model."""
        if not self.is_coqui_studio:
            # check for the coqui tts models
            if self.is_multi_speaker and (speaker is None and speaker_wav is None):
                raise ValueError("Model is multi-speaker but no `speaker` is provided.")
            if self.is_multi_lingual and language is None:
                raise ValueError("Model is multi-lingual but no `language` is provided.")
            if not self.is_multi_speaker and speaker is not None:
                raise ValueError("Model is not multi-speaker but `speaker` is provided.")
            if not self.is_multi_lingual and language is not None:
                raise ValueError("Model is not multi-lingual but `language` is provided.")
            if not emotion is None and not speed is None:
                raise ValueError("Emotion and speed can only be used with Coqui Studio models.")
        else:
            if emotion is None:
                emotion = "Neutral"
            if speed is None:
                speed = 1.0
            # check for the studio models
            if speaker_wav is not None:
                raise ValueError("Coqui Studio models do not support `speaker_wav` argument.")
            if speaker is not None:
                raise ValueError("Coqui Studio models do not support `speaker` argument.")
            if language is not None and language != "en":
                raise ValueError("Coqui Studio models currently support only `language=en` argument.")
            if emotion not in ["Neutral", "Happy", "Sad", "Angry", "Dull"]:
                raise ValueError(f"Emotion - `{emotion}` - must be one of `Neutral`, `Happy`, `Sad`, `Angry`, `Dull`.")

    def tts_coqui_studio(
        self,
        text: str,
        speaker_name: str = None,
        language: str = None,
        emotion: str = "Neutral",
        speed: float = 1.0,
        file_path: str = None,
    ) -> Union[np.ndarray, str]:
        """Convert text to speech using Coqui Studio models. Use `CS_API` class if you are only interested in the API.

        Args:
            text (str):
                Input text to synthesize.
            speaker_name (str, optional):
                Speaker name from Coqui Studio. Defaults to None.
            language (str, optional):
                Language code. Coqui Studio currently supports only English. Defaults to None.
            emotion (str, optional):
                Emotion of the speaker. One of "Neutral", "Happy", "Sad", "Angry", "Dull". Defaults to "Neutral".
            speed (float, optional):
                Speed of the speech. Defaults to 1.0.
            file_path (str, optional):
                Path to save the output file. When None it returns the `np.ndarray` of waveform. Defaults to None.

        Returns:
            Union[np.ndarray, str]: Waveform of the synthesized speech or path to the output file.
        """
        speaker_name = self.model_name.split("/")[2]
        if file_path is not None:
            return self.csapi.tts_to_file(
                text=text,
                speaker_name=speaker_name,
                language=language,
                speed=speed,
                emotion=emotion,
                file_path=file_path,
            )[0]
        return self.csapi.tts(text=text, speaker_name=speaker_name, language=language, speed=speed, emotion=emotion)[0]

    def tts(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. If None, Studio models use "Neutral". Defaults to None.
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0 and 2.0. If None, Studio models use 1.0.
                Defaults to None.
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, emotion=emotion, speed=speed)
        if self.csapi is not None:
            return self.tts_coqui_studio(
                text=text, speaker_name=speaker, language=language, emotion=emotion, speed=speed
            )

        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = "Neutral",
        speed: float = 1.0,
        file_path: str = "output.wav",
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. Defaults to "Neutral".
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0.0 and 2.0. Defaults to None.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav)

        if self.csapi is not None:
            return self.tts_coqui_studio(
                text=text, speaker_name=speaker, language=language, emotion=emotion, speed=speed, file_path=file_path
            )
        wav = self.tts(text=text, speaker=speaker, language=language, speaker_wav=speaker_wav)
        self.synthesizer.save_wav(wav=wav, path=file_path)
        return file_path

    def voice_conversion(
        self,
        source_wav: str,
        target_wav: str,
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker.

        Args:``
            source_wav (str):
                Path to the source wav file.
            target_wav (str):`
                Path to the target wav file.
        """
        wav = self.voice_converter.voice_conversion(source_wav=source_wav, target_wav=target_wav)
        return wav

    def voice_conversion_to_file(
        self,
        source_wav: str,
        target_wav: str,
        file_path: str = "output.wav",
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker.

        Args:
            source_wav (str):
                Path to the source wav file.
            target_wav (str):
                Path to the target wav file.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
        """
        wav = self.voice_conversion(source_wav=source_wav, target_wav=target_wav)
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
        return file_path

    def tts_with_vc(self, text: str, language: str = None, speaker_wav: str = None):
        """Convert text to speech with voice conversion.

        It combines tts with voice conversion to fake voice cloning.

        - Convert text to speech with tts.
        - Convert the output wav to target speaker with voice conversion.

        Args:
            text (str):
                Input text to synthesize.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            # Lazy code... save it to a temp file to resample it while reading it for VC
            self.tts_to_file(text=text, speaker=None, language=language, file_path=fp.name)
        if self.voice_converter is None:
            self.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
        wav = self.voice_converter.voice_conversion(source_wav=fp.name, target_wav=speaker_wav)
        return wav

    def tts_with_vc_to_file(
        self, text: str, language: str = None, speaker_wav: str = None, file_path: str = "output.wav"
    ):
        """Convert text to speech with voice conversion and save to file.

        Check `tts_with_vc` for more details.

        Args:
            text (str):
                Input text to synthesize.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
        """
        wav = self.tts_with_vc(text=text, language=language, speaker_wav=speaker_wav)
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)


def setup_vc_model_new(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "BaseVC":
    print(" > Using model: {}".format(config.model))
    # fetch the right model implementation.
    if "model" in config and config["model"].lower() == "freevc":
        MyModel = importlib.import_module("TTS.vc.models.freevc").FreeVC
        model = MyModel.init_from_config(config, samples)
    else:
        if "base_model" in config and config["base_model"] is not None:
            MyModel = find_module("TTS.tts.models", config.base_model.lower())
        else:
            MyModel = find_module("TTS.tts.models", config.model.lower())
        model = MyModel.init_from_config(config, samples)
    return model
