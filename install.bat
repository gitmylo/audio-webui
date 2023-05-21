@echo off
where py -V > nul 2> nul
if %errorlevel%==0 (
    py -3.10 install.py
) else (
    python install.py
)
