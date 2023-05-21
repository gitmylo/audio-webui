# Run the webui with skip install and bark cpu offloading
echo Starting webui, not installing packages. If you want to install packages. Run run.bat or install.bat instead.
source run.sh -si --bark-cpu-offload "$@"