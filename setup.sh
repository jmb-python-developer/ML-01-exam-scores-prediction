#!/usr/bin/env bash
# Sets up the project's virtual environment and installs requirements.
# Run with: source setup.sh   (so the activation persists in your shell)

VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd "$SCRIPT_DIR" > /dev/null || return 1

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

popd > /dev/null

echo "Done. Virtual environment active: $VIRTUAL_ENV"

if ! (return 0 2>/dev/null); then
    echo
    echo "Note: this script was executed, not sourced, so the venv activation"
    echo "won't persist in your shell. Re-run it as: source setup.sh"
fi
