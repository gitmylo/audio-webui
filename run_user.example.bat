@echo off
REM Run the webui with skip install and bark cpu offloading
echo Starting webui, not installing packages. If you want to install packages. Run run.bat or install.bat instead.
call run.bat -si --bark-cpu-offload %*