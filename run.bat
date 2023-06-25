@echo off
where py -V > nul 2> nul
if %errorlevel%==0 (
    call run_py.bat %*
) else (
    echo WARNING: Unable to run py command, you probably do not have py launcher installed, make sure you have the python.org version installed, not the windows store version.
    echo Launching anyways, running python command.
    call run_python.bat %*
)
