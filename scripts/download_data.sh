#!/usr/bin/env bash
# Download datasets for the ARC-AGI-2 solver.
#  - Official ARC-AGI-2 (public GitHub, no creds) is cloned by scaffold; re-clone here if needed.
#  - NVARC artifacts require Kaggle creds at ~/.kaggle/kaggle.json.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/arc-agi-2 data/nvarc

# 1) Official ARC-AGI-2 tasks (public)
if [ ! -d data/arc-agi-2/training ]; then
  echo ">> Cloning official ARC-AGI-2 data from GitHub"
  tmp=$(mktemp -d)
  git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git "$tmp"
  cp -r "$tmp/data/training" "$tmp/data/evaluation" data/arc-agi-2/
  rm -rf "$tmp"
fi

# 2) NVARC artifacts (needs Kaggle credentials)
if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "!! No Kaggle credentials found (~/.kaggle/kaggle.json)."
  echo "   Create an API token at https://www.kaggle.com/settings -> 'Create New Token',"
  echo "   then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  echo "   Skipping NVARC downloads for now."
  exit 0
fi

# Start small (disk is tight): synthetic 103k first, augmented (3.2M) only on demand.
for ds in nvarc-synthetic-puzzles nvarc-artifacts-puzzles; do
  echo ">> Downloading sorokin/$ds"
  kaggle datasets download -d "sorokin/$ds" -p data/nvarc --unzip
done
echo ">> (Skipping nvarc-augmented-puzzles by default — large. Pull on demand:)"
echo "   kaggle datasets download -d sorokin/nvarc-augmented-puzzles -p data/nvarc --unzip"
