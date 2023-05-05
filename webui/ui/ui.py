import gradio as gr
from .tabs import *


def create_ui() -> gr.Blocks:

    css = """
    .gradio-container {
        max-width: calc(100% - 100px) !important;
    }
    
    .tabitem {
        height: calc(100vh - 100px) !important;
        overflow: auto;
    }
    """

    tabs = [
        ('extra', extra_tab)
    ]
    with gr.Blocks(theme='gradio/soft', title='Audio WebUI', css=css) as webui:
        with gr.Tabs() as _tabs:
            for name, content in tabs:
                with gr.Tab(name):
                    content()
    return webui
