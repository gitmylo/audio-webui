from .no_colab_package import NoColabRequirement
from .packaging_package import Packaging
from .torch_package import Torch
from .huggingface_package import Transformers, diffusers
from .audio2numpy_package import AudioToNumpy
from .bark_package import Bark, SoundFileOrSox
from .audiolm_package import AudioLM, JobLib, FairSeq
from .rvc_package import Praat, PyWorld, FaissCpu, TorchCrepe, FfmpegPython, NoiseReduce, LibRosa, Demucs
# from .tts_package import TTS
from .pytube_package import PyTube
from .whisper_package import Whisper
from setup_tools.magicinstaller.requirement import SimpleRequirementInit, CompareAction

requirements = [
    Packaging(),  # Allows for version checks

    # TTS(),
    SimpleRequirementInit('wheel'),

    # SimpleRequirementInit('numpy', CompareAction.EQ, '1.23.5'),
    NoColabRequirement('numpy', CompareAction.EQ, '1.23.5'),  # Don't install this one when in google colab

    Torch(),

    Transformers(),
    diffusers(),  # This one's a function
    SimpleRequirementInit('gradio', CompareAction.EQ, '3.49.0'),
    SimpleRequirementInit('huggingface-hub', CompareAction.EQ, '0.19.4'),
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

    SimpleRequirementInit('audiocraft', CompareAction.GEQ, '1.0.0'),

    SimpleRequirementInit('beartype', CompareAction.EQ, '0.15.0')  # Overwrite version of beartype which broke things.
]
