#!/usr/bin/env bash

# Unable to check the testing here. As i do not have a linux system, and not enough storage on C: to install wsl

if which python3.10 >/dev/null; then
  python3.10 main.py "$@"
else
  echo 'WARNING: python3.10 command was not found, attempting with python command, this could fail.'
  python main.py "$@"
fi

read -n1 -r -p "Press any key to exit..." key
