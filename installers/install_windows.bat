@echo off
mkdir "audio-webui"
cd "audio-webui"
git init
git remote add origin https://github.com/gitmylo/audio-webui
git fetch
git reset origin/master  # Required when the versioned files existed in path before "git init" of this repo.
git checkout -t origin/master -f