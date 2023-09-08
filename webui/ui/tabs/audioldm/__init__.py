import gradio
from .audioldm import audioldm_tab
from .audioldm2 import audioldm2_tab


def create_tab():
    with gradio.Tabs():
        with gradio.Tab("AudioLDM 1 🎵"):
            audioldm_tab()
        with gradio.Tab("AudioLDM 2 🎶"):
            audioldm2_tab()
