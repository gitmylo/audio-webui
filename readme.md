# Readme

## Command line flags

| Name            | Args     | Short      | Usage                 | Description                                    |
|-----------------|----------|------------|-----------------------|------------------------------------------------|
| --skip-install  | [None]   | -si        | -si                   | Skip installing packages                       |
| --bark-low-vram | [None]   | [None]     | --bark-low-vram       | Use low vram for bark                          |
| --bark-use-cpu  | [None]   | [None]     | --bark-use-cpu        | Use cpu for bark                               |
| --tts-cpu       | [None]   | [None]     | --tts-use-cpu         | [Currently deprecated] Use cpu for tts library |
| --share         | [None]   | -s         | -s                    | Share the gradio instance publicly             |
| --username      | username | -u, --user | -u username           | Set the username for gradio                    |
| --password      | password | -p, --pass | -p password           | Set the password for gradio                    |
| --theme         | theme    | [None]     | --theme "gradio/soft" | Set the theme for gradio                       |


## Current goals
* [x] Text-to-speech
  * [x] Bark
    * [x] Speech generation
    * [x] Voice cloning
      * [x] Basic voice cloning
      * [ ] Accurate voice cloning
    * [ ] Unlimited length generation using NLTK
    * [ ] Disable stopping token option to let the AI decide how it wants to continue
* [ ] Audio-to-audio
* [ ] Automatic-speech-recognition

## More readme
[Link](readme/readme.md)