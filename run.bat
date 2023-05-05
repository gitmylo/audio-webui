@echo off
where py -V > nul 2> nul
if %errorlevel%==0 (
    call run_py.bat
) else (
    call run_python.bat
)
pause