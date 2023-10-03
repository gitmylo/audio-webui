# <img alt="logo" height="25" src="assets/logo.png" width="25"/> Audio Webui <img alt="logo" height="25" src="assets/logo.png" width="25"/>


https://github.com/gitmylo/audio-webui/assets/36931363/c285b4dc-63cf-4b1c-895d-9723a2cbf91e


[![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/t/gitmylo/audio-webui)](https://github.com/gitmylo/audio-webui/commits/master)
[![GitHub contributors](https://img.shields.io/github/contributors-anon/gitmylo/audio-webui)](https://github.com/gitmylo/audio-webui/graphs/contributors)
[![GitHub all releases](https://img.shields.io/github/downloads/gitmylo/audio-webui/total?label=installer%20downloads)](https://github.com/gitmylo/audio-webui/releases/tag/Installers)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/gitmylo?label=github+sponsors+supporters)](https://github.com/sponsors/gitmylo)

[![Discord](https://img.shields.io/discord/1118525872882843711?style=flat&label=discord)](https://discord.gg/NB86C3Szkg)


## ❗❗ Please read ❗❗
This code works using Python 3.10 (lower versions don't support union types, and I believe version 3.11 doesn't have support for the current TTS library at the moment).

You also need to have [Git](https://git-scm.com/downloads) installed, you might already have it, to see if you do, run `git --version` in a console/terminal.

Some features require [FFmpeg](http://ffmpeg.org/) to be installed.

On Windows, you need to have [Build Tools for Visual Studio](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed.

[Common issues](https://github.com/gitmylo/audio-webui/blob/master/readme/common_issues.md)

<!-- TOC -->
### Content
* [<img alt="logo" height="15" src="assets/logo.png" width="15"/> Audio Webui <img alt="logo" height="15" src="assets/logo.png" width="15"/>](#img-altlogo-height25-srcassetslogopng-width25-audio-webui-img-altlogo-height25-srcassetslogopng-width25)
  * [❗ Please read](#-please-read-)
  * [📰 Latest features](#-latest-features)
  * [👍 Automatic installers](#-automatic-installers)
  * [📦 Docker](#-docker)
  * [💻 Local install (Manual)](#-local-install-manual)
    * [🔽 Downloading](#-downloading)
    * [📦 Installing](#-installing)
    * [🔼 Updating](#-updating)
    * [🏃‍ Running](#-running)
  * [💻 Google Colab notebook](#-google-colab-notebook)
  * [💻 Common command line flags](#-common-command-line-flags)
  * [✨ Current goals and features](#-current-goals-and-features-)
  * [➕ More](#-more)
<!-- TOC -->

## 📰 Latest features
* [🧩 Extensions](readme/extensions/index.md)

## 👍 Automatic installers
[Automatic installers! (Download)](https://github.com/gitmylo/audio-webui/releases/tag/Installers)
1. Put the installer in a folder
2. Run the installer for your operating system.
3. Now run the webui's install script. Follow the steps at [📦 Installing](#-installing)

## 📦 Docker
<details>
<summary>Links to community audio-webui docker projects</summary>

* https://github.com/LajaSoft/audio-webui-docker (Docker compose which downloads jacen92's fork)
* https://github.com/jacen92/audio-webui-docker (Fork of audio-webui which includes docker compose)

</details>
Note: The docker repositories are not maintained by me. And docker related issues should go to the docker repositories.  
If an issue is related to audio-webui directly, create the issue here. Unless a fix has already been made.

## 💻 Local install (Manual)
### 🔽 Downloading
It is recommended to use git to download the webui, using git allows for easy updating.

To download using git, run `git clone https://github.com/gitmylo/audio-webui` in a console/terminal

### 📦 Installing
Installation is done automatically in a venv when you run `run.bat` or `run.sh` (.bat on Windows, .sh on Linux/MacOS).

### 🔼 Updating
To update,  
run `update.bat` on windows, `update.sh` on linux/macos  
OR run `git pull` in the folder your webui is installed in.

### 🏃‍ Running
Running should be as simple as running `run.bat` or `run.sh` depending on your OS.
Everything should get installed automatically.

If there's an issue with running, please create an [issue](https://github.com/gitmylo/audio-webui/issues)

## 💻 Google Colab notebook
[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitmylo/audio-webui/blob/master/audio_webui_colab.ipynb) [![Open in github](https://img.shields.io/badge/Open%20in%20GitHub-green?logo=github&labelColor=555
)](audio_webui_colab.ipynb)

## 💻 Common command line flags

| Name            | Args                                   | Short      | Usage                 | Description                                                                                                            |
|-----------------|----------------------------------------|------------|-----------------------|------------------------------------------------------------------------------------------------------------------------|
| --skip-install  | [None]                                 | -si        | -si                   | Skip installing packages                                                                                               |
| --skip-venv     | [None]                                 | -sv        | -sv                   | Skip creating/activating venv, also skips install. (for advanced users)                                                |
| --no-data-cache | [None]                                 | [None]     | --no-data-cache       | Don't change the default dir for huggingface_hub models. (This might fix some models not loading)                      |
| --launch        | [None]                                 | [None]     | --launch              | Automatically open the webui in your browser once it launches.                                                         |
| --share         | [None]                                 | -s         | -s                    | Share the gradio instance publicly                                                                                     |
| --username      | username (str)                         | -u, --user | -u username           | Set the username for gradio                                                                                            |
| --password      | password (str)                         | -p, --pass | -p password           | Set the password for gradio                                                                                            |
| --theme         | theme (str)                            | [None]     | --theme "gradio/soft" | Set the theme for gradio                                                                                               |
| --listen        | [None]                                 | -l         | -l                    | Listen a server, allowing other devices within your local network to access the server. (or outside if port forwarded) |
| --port          | port (int)                             | [None]     | --port 12345          | Set a custom port to listen on, by default a port is picked automatically                                              |

## ✨ Current goals and features
Moved to [a separate readme](readme/features.md)

## ➕ More
There are additional resources for individual topics not covered in this readme. You can view them in [this list](readme/readme.md).
