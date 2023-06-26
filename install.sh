if ! type "python3.10" > /dev/null; then
  python3.10 install.py "$@"
else
  echo 'WARNING: python3.10 command was not found, attempting with python command, this could fail.'
  python install.py "$@"
fi

read -n1 -r -p "Press any key to exit..." key
