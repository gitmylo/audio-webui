from .no_colab_package import NoColabRequirement
from .packaging_package import Packaging
from .torch_package import Torch
from .huggingface_package import Transformers, diffusers
from .audio2numpy_package import AudioToNumpy
from .bark_package import Bark, SoundFileOrSox
from .audiolm_package import AudioLM, JobLib, FairSeq
from .rvc_package import Praat, PyWorld, FaissCpu, TorchCrepe, FfmpegPython, NoiseReduce, LibRosa, Demucs
from .tts_package import TTS
from .pytube_package import PyTube
from .whisper_package import Whisper
from .audiocraft_package import AudioCraft
from setup_tools.magicinstaller.requirement import SimpleRequirementInit, CompareAction

requirements = [
    Packaging(),  # Allows for version checks

    TTS(),

    # SimpleRequirementInit('numpy', CompareAction.EQ, '1.23.5'),
    NoColabRequirement('numpy', CompareAction.EQ, '1.23.5'),  # Don't install this one when in google colab

    Torch(),

    Transformers(),
    diffusers(),  # This one's a function
    SimpleRequirementInit('gradio', CompareAction.EQ, '3.35.2'),
    SimpleRequirementInit('huggingface-hub', CompareAction.EQ, '0.17.1'),  # TODO: remove this once huggingface-hub downloads work again on windows (FileNotFoundError: [WinError 3] The system cannot find the path specified, #137)

    AudioToNumpy(),

    Bark(),
    SoundFileOrSox(),

    AudioLM(),
    JobLib(),
    FairSeq(),

    Praat(),
    PyWorld(),
    FaissCpu(),
    TorchCrepe(),
    FfmpegPython(),
    NoiseReduce(),
    LibRosa(),
    Demucs(),

    PyTube(),

    Whisper(),

    AudioCraft(),

    SimpleRequirementInit('beartype', CompareAction.EQ, '0.15.0')  # Overwrite version of beartype which broke things.
]
