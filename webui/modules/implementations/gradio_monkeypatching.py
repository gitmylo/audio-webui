from typing import Literal, Callable

import gradio
import numpy as np


class Audio(gradio.Audio):
    def __init__(
            self,
            value: str | tuple[int, np.ndarray] | Callable | None = None,
            *,
            source: str = "upload",
            type: str = "numpy",
            label: str | None = None,
            every: float | None = None,
            show_label: bool = True,
            container: bool = True,
            scale: int | None = None,
            min_width: int = 160,
            interactive: bool | None = None,
            visible: bool = True,
            streaming: bool = False,
            elem_id: str | None = None,
            elem_classes: list[str] | str | None = None,
            format: Literal["wav", "mp3"] = "wav",
            autoplay: bool = False,
            **kwargs,
    ):
        super().__init__(value, source=source, type=type, label=label, every=every, show_label=show_label,
                         container=container, scale=scale, min_width=min_width, interactive=interactive,
                         visible=visible, streaming=streaming, elem_id=elem_id, elem_classes=elem_classes,
                         format=format, autoplay=autoplay, **kwargs)
        self.change(fn=lambda a: a, inputs=self, outputs=self)


def patch():
    print('Monkeypatching gradio')
    gradio.Audio = Audio

