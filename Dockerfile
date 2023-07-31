FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL Version="1.0"
LABEL Maintainer="Nicolas Gargaud <jacen92@gmail.com>"
LABEL Description="audio webui"

ARG USER_NAME="dev"
ARG USER_UID="1000"
ENV USER_NAME=${USER_NAME}
ENV USER_UID=${USER_UID}
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DATASET_ROOT="/datasets"
ENV MODELS_ROOT="/models"
ENV PATH=/home/${USER_NAME}/.local/bin:$PATH
ENV TERM xterm
ENV UI_PORT 7860

RUN apt-get update \
    && apt-get install -y \
      apt-utils sudo nano git wget \
      build-essential make gcc g++ ffmpeg \
      python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN adduser --disabled-password --gecos "" ${USER_NAME} \
    && usermod -aG sudo ${USER_NAME} && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && echo "${USER_NAME}:${USER_NAME}" | chpasswd \
    && usermod -u ${USER_UID} ${USER_NAME} \
    && mkdir -p ${DATASET_ROOT} && mkdir -p ${MODELS_ROOT} && mkdir -p /home/dev/.local/share/tts \
    && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

RUN sudo -u ${USER_NAME} pip install --user audio2numpy \
                                            diffusers \
                                            faiss-cpu==1.7.3 \
                                            ffmpeg-python>=0.2.0 \
                                            git+https://github.com/suno-ai/bark.git@6921c9139a97d0364208407191c92ec265ef6759 \
                                            git+https://github.com/facebookresearch/demucs#egg=demucs \
                                            git+https://github.com/facebookresearch/audiocraft.git \
                                            gradio \
                                            joblib \
                                            librosa \
                                            noisereduce \
                                            openai-whisper \
                                            praat-parselmouth>=0.4.2 \
                                            pyworld>=0.3.2 \
                                            pytube \
                                            sox \
                                            torch==2.0.1 \
                                            torchaudio \
                                            torchcrepe \
                                            torchvision \
                                            transformers \
                                            TTS

# install afterward to avoid failure
RUN sudo -u ${USER_NAME} pip install --user fairseq audiolm-pytorch tensorboard tensorboardX
COPY --chown=${USER_NAME}:${USER_NAME} . /opt/app/

USER ${USER_NAME}
EXPOSE ${UI_PORT}
WORKDIR /opt/app
VOLUME ["${DATASET_ROOT}", "${MODELS_ROOT}", "/home/dev/.local/share/tts"]
CMD python3 "./main.py" --listen --port ${UI_PORT} --skip-install --skip-venv
