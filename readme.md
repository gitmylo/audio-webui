# Readme 💀x7

## Running
Running should be as simple as running `run.bat` or `run.sh` depending on your OS.
Everything should get installed automatically.

This has not been tested beyond 2 of my pcs.
If there's an issue with running, please create an [issue](https://github.com/gitmylo/audio-webui/issues)

## Command line flags

| Name               | Args     | Short      | Usage                 | Description                                                |
|--------------------|----------|------------|-----------------------|------------------------------------------------------------|
| --skip-install     | [None]   | -si        | -si                   | Skip installing packages                                   |
| --bark-low-vram    | [None]   | [None]     | --bark-low-vram       | Use low vram for bark                                      |
| --bark-cpu-offload | [None]   | [None]     | --bark-cpu-offload    | Use cpu offloading to save vram while still running on gpu |
| --bark-use-cpu     | [None]   | [None]     | --bark-use-cpu        | Use cpu for bark                                           |
| --share            | [None]   | -s         | -s                    | Share the gradio instance publicly                         |
| --username         | username | -u, --user | -u username           | Set the username for gradio                                |
| --password         | password | -p, --pass | -p password           | Set the password for gradio                                |
| --theme            | theme    | [None]     | --theme "gradio/soft" | Set the theme for gradio                                   |


## Current goals
* [x] Text-to-speech
  * [x] Bark
    * [x] Speech generation
    * [x] Voice cloning
      * [x] Basic voice cloning
      * [x] Accurate voice cloning
    * [x] Disable stopping token option to let the AI decide how it wants to continue
* [ ] Audio-to-audio
* [ ] Automatic-speech-recognition

## More readme
[Link](readme/readme.md)
