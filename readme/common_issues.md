# ❗ Common issues ❗
## ❌ Error messages

| Error                                                                        | Solution                                                                                           |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| TypeError: 'type' object is not subscriptable                                | You're on python < 3.10, Use [Python 3.10](https://www.python.org/downloads/release/python-31012/) |
| TypeError: unsupported operand type(s) for : 'types.GenericAlias' and 'type' | You're on python < 3.10, Use [Python 3.10](https://www.python.org/downloads/release/python-31012/) |
| I cannot install TTS, no matching version found                              | You're on python > 3.10, Use [Python 3.10](https://www.python.org/downloads/release/python-31012/) |
| Please install tensorboardX: pip install tensorboardX                        | You can ignore this warning as it is not a requirement.                                            |


## 🔍 Missing models

| Issue                            | Solution                                                                                                      |
|----------------------------------|---------------------------------------------------------------------------------------------------------------|
| Where does the HuBERT model go?  | All HuBERT models download automatically to `data/models/hubert`                                              |
| RVC models don't get downloaded. | RVC models have to manually be downloaded and installed, unzip them into a folder inside of `data/models/rvc` |

## ❗ Other issues

| Issue                                                        | Solution                                                                                                                                                                                                                                                                                                      |
|--------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Launching takes a long time, it's always installing packages | Launch with the `--skip-install` flag.                                                                                                                                                                                                                                                                        |
| I want to host the app over my local network                 | Launch with the `-l`/`--listen` flag.                                                                                                                                                                                                                                                                         |
| I want to use a different port                               | Launch with the `--port xxxx` flag where `xxxx` is the port you want to use.                                                                                                                                                                                                                                  |
| I'm running out of VRAM (CUDA Out Of Memory error)           | Launch with `--bark-cpu-offload` for much lower vram usage at the cost of a small speed decrease, use `--bark-low-vram` to reduce vram usage and increase speed at the cost of quality. And use `--bark-use-cpu` to fully run the model on the CPU, with no VRAM usage, likely at the cost of a lot of speed. |
| Models aren't loading. The loading stays frozen.             | Something's wrong with the file path of audio-webui (Too long?) Launch with the `--no-data-cache`                                                                                                                                                                                                             |
