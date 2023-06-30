# <img alt="logo" height="25" src="assets/logo.png" width="25"/> Audio Webui <img alt="logo" height="25" src="assets/logo.png" width="25"/>

[![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/t/gitmylo/audio-webui)](https://github.com/gitmylo/audio-webui/commits/master)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/gitmylo/audio-webui)](https://github.com/gitmylo/audio-webui/graphs/contributors)
[![GitHub all releases](https://img.shields.io/github/downloads/gitmylo/audio-webui/total?label=installer%20downloads)](https://github.com/gitmylo/audio-webui/releases/tag/Installers)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/gitmylo?label=github+sponsors+supporters)](https://github.com/sponsors/gitmylo)

[![Discord](https://img.shields.io/discord/1118525872882843711?style=flat&label=discord)](https://discord.gg/NB86C3Szkg)


## ❗❗ Please read ❗❗
This code works on python 3.10 (lower versions don't support "|" type annotations, and i believe 3.11 doesn't have support for the TTS library currently).

You also need to have [Git](https://git-scm.com/downloads) installed, you might already have it, run `git --version` in a console/terminal to see if you already have it installed.

[Common issues](https://github.com/gitmylo/audio-webui/wiki/common-issues)

<!-- TOC -->
* [<img alt="logo" height="15" src="assets/logo.png" width="15"/> Audio Webui <img alt="logo" height="15" src="assets/logo.png" width="15"/>](#img-altlogo-height25-srcassetslogopng-width25-audio-webui-img-altlogo-height25-srcassetslogopng-width25)
  * [❗❗ Please read ❗❗](#-please-read-)
  * [👍 NEW: Automatic installers](#-new--automatic-installers)
  * [💬 NEW: Discord server](#-new--discord-server)
  * [💻 Local install (Manual)](#-local-install--manual-)
    * [🔽 Downloading](#-downloading)
    * [📦 Installing](#-installing)
    * [🔼 Updating](#-updating)
    * [🏃‍ Running](#-running)
  * [💻 Google colab notebook](#-google-colab-notebook)
  * [💻 Command line flags](#-command-line-flags)
  * [✨ Current goals and features ✨](#-current-goals-and-features-)
  * [More readme](#more-readme)
<!-- TOC -->

## 👍 NEW: Automatic installers
[Automatic installers! (Download)](https://github.com/gitmylo/audio-webui/releases/tag/Installers)
1. Put the installer in a folder
2. Run the installer for your operating system.
3. Now run the webui's install script. Follow the steps at [📦 Installing](#-installing)

## 💻 Local install (Manual)
### 🔽 Downloading
It is recommended to use git to download the webui, using git allows for easy updating.

To download using git, run `git clone https://github.com/gitmylo/audio-webui` in a console/terminal

### 📦 Installing
Installation is done automatically in a venv when `run.bat` or `run.sh` is ran without the `--skip-install` flag.

Alternatively, run `install.bat` or `install.sh` to just install, and nothing else. To install with install.bat in a custom environment which is currently active. Do `install.bat --skip-venv` or `install.sh --skip-venv`.

### 🔼 Updating
To update,  
run `update.bat` on windows, `update.sh` on linux/macos  
OR run `git pull` in the folder your webui is installed in.

### 🏃‍ Running
Running should be as simple as running `run.bat` or `run.sh` depending on your OS.
If you want to run with custom command line flags, copy `run_user_example.(bat/sh)` and put whatever flags you want on every run in there. recommended flags are already in the example. (skip install and cpu offload)
Everything should get installed automatically.

This has not been tested beyond 2 of my pcs.
If there's an issue with running, please create an [issue](https://github.com/gitmylo/audio-webui/issues)

## 💻 Google colab notebook
[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitmylo/audio-webui/blob/master/audio_webui_colab.ipynb) [![Open in github](https://img.shields.io/badge/Github-Open%20file-green)](audio_webui_colab.ipynb)

## 💻 Command line flags

| Name               | Args                             | Short      | Usage                 | Description                                                                                                            |
|--------------------|----------------------------------|------------|-----------------------|------------------------------------------------------------------------------------------------------------------------|
| --skip-install     | [None]                           | -si        | -si                   | Skip installing packages                                                                                               |
| --skip-venv        | [None]                           | -sv        | -sv                   | Skip creating/activating venv, also skips install. (for advanced users)                                                |
| --no-data-cache    | [None]                           | [None]     | --no-data-cache       | Don't change the default dir for huggingface_hub models. (This might fix some models not loading)                      |
| --bark-low-vram    | [None]                           | [None]     | --bark-low-vram       | Use low vram for bark                                                                                                  |
| --bark-cpu-offload | [None]                           | [None]     | --bark-cpu-offload    | Use cpu offloading to save vram while still running on gpu                                                             |
| --bark-use-cpu     | [None]                           | [None]     | --bark-use-cpu        | Use cpu for bark                                                                                                       |
| --bark-half        | [None]                           | [None]     | --bark-half           | Use half precision for bark models. (This uses less VRAM) (Experimental)                                               |
| --tts-use-gpu      | [None]                           | [None]     | --tts-use-gpu         | Use your GPU for TTS with the TTS library                                                                              |
| --share            | [None]                           | -s         | -s                    | Share the gradio instance publicly                                                                                     |
| --username         | username (str)                   | -u, --user | -u username           | Set the username for gradio                                                                                            |
| --password         | password (str)                   | -p, --pass | -p password           | Set the password for gradio                                                                                            |
| --theme            | theme (str)                      | [None]     | --theme "gradio/soft" | Set the theme for gradio                                                                                               |
| --listen           | [None]                           | -l         | -l                    | Listen a server, allowing other devices within your local network to access the server. (or outside if port forwarded) |
| --port             | port (int)                       | [None]     | --port 12345          | Set a custom port to listen on, by default a port is picked automatically                                              |
| --wav-type         | type any of: [gradio, showwaves] | [None]     | --wav-type showwaves  | Change the visualizers for creating a video from audio.                                                                |


## ✨ Current goals and features ✨
moved to [wiki](https://github.com/gitmylo/audio-webui/wiki/Features)

## More readme
* 🐶 Bark info
* 😎 RVC info

[Link](https://github.com/gitmylo/audio-webui/wiki/info)
