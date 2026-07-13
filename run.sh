#!/bin/bash
# Launcher for voice-dictation. Double-clickable / callable from Terminal.
cd "$(dirname "$0")" || exit 1
# Same accessory mode the .app bundle uses: without it the app keeps a Dock
# icon and a slot in the Cmd+Tab switcher instead of living in the menu bar.
export VD_ACCESSORY="${VD_ACCESSORY:-1}"
exec .venv/bin/python dictation_gui.py
