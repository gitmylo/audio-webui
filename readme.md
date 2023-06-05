# Readme 💀x7

## Please read
This code works on python 3.10, i have not tested it on other versions. Some older versions will have issues.

## Running
Running should be as simple as running `run.bat` or `run.sh` depending on your OS.
If you want to run with custom command line flags, copy `run_user_example.(bat/sh)` and put whatever flags you want on every run in there. recommended flags are already in the example. (skip install and cpu offload)
Everything should get installed automatically.

This has not been tested beyond 2 of my pcs.
If there's an issue with running, please create an [issue](https://github.com/gitmylo/audio-webui/issues)

## Command line flags

| Name                       | Args           | Short      | Usage                      | Description                                                                                                            |
|----------------------------|----------------|------------|----------------------------|------------------------------------------------------------------------------------------------------------------------|
| --skip-install             | [None]         | -si        | -si                        | Skip installing packages                                                                                               |
| --skip-venv                | [None]         | -sv        | -sv                        | Skip creating/activating venv, also skips install. (for advanced users)                                                |
| --bark-low-vram            | [None]         | [None]     | --bark-low-vram            | Use low vram for bark                                                                                                  |
| --bark-cpu-offload         | [None]         | [None]     | --bark-cpu-offload         | Use cpu offloading to save vram while still running on gpu                                                             |
| --bark-use-cpu             | [None]         | [None]     | --bark-use-cpu             | Use cpu for bark                                                                                                       |
| --bark-cloning-large-model | [None]         | [None]     | --bark-cloning-large-model | Use the larger voice cloning model. (It hasn't been tested as much yet)                                                |
| --share                    | [None]         | -s         | -s                         | Share the gradio instance publicly                                                                                     |
| --username                 | username (str) | -u, --user | -u username                | Set the username for gradio                                                                                            |
| --password                 | password (str) | -p, --pass | -p password                | Set the password for gradio                                                                                            |
| --theme                    | theme (str)    | [None]     | --theme "gradio/soft"      | Set the theme for gradio                                                                                               |
| --listen                   | [None]         | -l         | -l                         | Listen a server, allowing other devices within your local network to access the server. (or outside if port forwarded) |
| --port                     | port (int)     | [None]     | --port 12345               | Set a custom port to listen on, by default a port is picked automatically                                              |


## Current goals and features
* [x] Text-to-audio
  * [x] Text-to-speech
    * [x] Bark
      * [x] Speech generation
      * [x] Voice cloning
        * [x] Basic voice cloning
        * [x] Accurate voice cloning
      * [x] Disable stopping token option to let the AI decide how it wants to continue
  * [x] AudioLDM text-to-audio generation
* [x] Audio-to-audio
  * [x] Bark audio-to-audio
  * [x] RVC (retrieval based voice conversion)
* [ ] Automatic-speech-recognition
  * [ ] Whisper speech recognition

## More readme
[Link](readme/readme.md)
