#!/bin/bash
# Launcher for voice-dictation. Double-clickable / callable from Terminal.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python dictation_gui.py
