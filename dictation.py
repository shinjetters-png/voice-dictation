#!/usr/bin/env python3
"""
voice-dictation — a superwhisper-style local dictation tool for macOS (Apple Silicon).

Flow:  hotkey -> record mic -> transcribe locally with mlx-whisper -> paste at cursor.

Runs as a menu-bar app (rumps). Config lives in config.json next to this file.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")

# Icons for the menu-bar title
ICON_IDLE = "🎙️"
ICON_REC = "🔴"
ICON_WORK = "⏳"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def play_sound(name):
    """Play a built-in macOS system sound without blocking."""
    path = f"/System/Library/Sounds/{name}.aiff"
    if os.path.exists(path):
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def set_clipboard(text):
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))


def paste_at_cursor():
    """Simulate Cmd+V into the frontmost app via System Events."""
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class Recorder:
    """Captures mono float32 audio from the default input device into a buffer."""

    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.frames = queue.Queue()
        self.stream = None
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.put(indata.copy())

    def start(self):
        # drain any leftover frames
        while not self.frames.empty():
            self.frames.get_nowait()
        self.recording = True
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()

    def stop(self):
        self.recording = False
        chunks = []
        while not self.frames.empty():
            chunks.append(self.frames.get_nowait())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0).flatten()


class DictationApp(rumps.App):
    def __init__(self, config):
        super().__init__(ICON_IDLE, quit_button=None)
        self.config = config
        self.recorder = Recorder(config["samplerate"])
        self.busy = False
        self._transcribe = None  # lazily-loaded mlx_whisper.transcribe

        self.status_item = rumps.MenuItem("準備中…")
        self.menu = [
            self.status_item,
            None,
            rumps.MenuItem("設定を再読み込み", callback=self.reload_config),
            rumps.MenuItem("終了", callback=self.quit_app),
        ]

        # Warm up the model in the background so the first dictation is fast.
        threading.Thread(target=self._warm_up, daemon=True).start()
        # Start listening for the hotkey.
        threading.Thread(target=self._listen, daemon=True).start()

    # ---- model ----------------------------------------------------------
    def _warm_up(self):
        self._set_status("モデル読み込み中…")
        import mlx_whisper  # heavy import, do it off the main thread

        self._transcribe = mlx_whisper.transcribe
        # Run once on silence to trigger the model download / compile.
        try:
            self._transcribe(
                np.zeros(16000, dtype=np.float32),
                path_or_hf_repo=self.config["model"],
            )
        except Exception as e:
            self._set_status(f"モデル読込エラー: {e}")
            return
        self._set_status("待機中（ホットキーで開始）")

    # ---- hotkey ---------------------------------------------------------
    def _listen(self):
        mode = self.config.get("mode", "hold")
        if mode == "hold":
            key_name = self.config.get("hold_key", "alt_r")
            target = getattr(keyboard.Key, key_name, None)
            if target is None:
                self._set_status(f"不明なキー: {key_name}")
                return

            def on_press(key):
                if key == target and not self.recorder.recording and not self.busy:
                    self._start_recording()

            def on_release(key):
                if key == target and self.recorder.recording:
                    self._stop_and_transcribe()

            with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
                l.join()
        else:
            combo = self.config.get("toggle_hotkey", "<ctrl>+<alt>+d")
            with keyboard.GlobalHotKeys({combo: self._toggle}) as l:
                l.join()

    def _toggle(self):
        if self.busy:
            return
        if self.recorder.recording:
            self._stop_and_transcribe()
        else:
            self._start_recording()

    # ---- record / transcribe -------------------------------------------
    def _start_recording(self):
        if self._transcribe is None:
            self._set_status("モデル準備中… 少しお待ちを")
            return
        self.recorder.start()
        self.title = ICON_REC
        self._set_status("録音中…")
        if self.config.get("play_sounds", True):
            play_sound("Tink")

    def _stop_and_transcribe(self):
        audio = self.recorder.stop()
        self.busy = True
        self.title = ICON_WORK
        self._set_status("文字起こし中…")
        threading.Thread(target=self._do_transcribe, args=(audio,), daemon=True).start()

    def _do_transcribe(self, audio):
        try:
            if audio.size < self.config["samplerate"] * 0.3:
                self._set_status("録音が短すぎます")
                return
            lang = self.config.get("language") or None
            kwargs = {"path_or_hf_repo": self.config["model"]}
            if lang:
                kwargs["language"] = lang
            prompt = self.config.get("initial_prompt")
            if prompt:
                kwargs["initial_prompt"] = prompt

            result = self._transcribe(audio, **kwargs)
            text = (result.get("text") or "").strip()

            if not text:
                self._set_status("（無音でした）")
                return

            set_clipboard(text)
            if self.config.get("paste", True):
                time.sleep(0.15)  # let the hotkey keys fully release
                paste_at_cursor()
            if self.config.get("play_sounds", True):
                play_sound("Pop")
            preview = text if len(text) <= 40 else text[:40] + "…"
            self._set_status(f"✓ {preview}")
        except Exception as e:
            self._set_status(f"エラー: {e}")
        finally:
            self.busy = False
            self.title = ICON_IDLE

    # ---- menu / status --------------------------------------------------
    def _set_status(self, msg):
        self.status_item.title = msg

    def reload_config(self, _):
        self.config = load_config()
        self._set_status("設定を再読み込みしました（ホットキーは再起動で反映）")

    def quit_app(self, _):
        rumps.quit_application()


def main():
    config = load_config()
    DictationApp(config).run()


if __name__ == "__main__":
    main()
