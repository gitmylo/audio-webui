import gradio
from webui.ui.tabs.training.rvc import train_rvc


def training_tab():
    with gradio.Tabs():
        with gradio.Tab('🧬 RVC'):
            train_rvc()
