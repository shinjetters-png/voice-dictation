#!/usr/bin/env python3
"""
voice-dictation (GUI) — a superwhisper-style local dictation app for macOS.

Menu-bar app: press a hotkey to record, transcribe locally with mlx-whisper, and
paste at the cursor. Has a modern sidebar UI with a custom-word dictionary that
biases recognition toward your own vocabulary.
"""

import json
import os
import queue
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd
from AppKit import (
    NSEvent,
    NSEventMaskKeyDown,
    NSEventMaskFlagsChanged,
    NSEventModifierFlagShift,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSEventModifierFlagCommand,
)
from Foundation import NSObject
from PySide6.QtCore import Qt, QObject, Signal, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
DICT_PATH = os.path.join(HERE, "dictionary.json")

MODELS = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-large-v3",
    "kaiinui/kotoba-whisper-v2.0-mlx",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
    "mlx-community/whisper-medium",
    "mlx-community/whisper-small",
    "mlx-community/whisper-tiny",
]


def model_backend(model_id):
    """Qwen3-ASR runs on mlx-audio; everything else is a Whisper-family
    model that mlx-whisper can load directly."""
    return "qwen" if "qwen3-asr" in model_id.lower() else "whisper"
HOLD_KEYS = ["cmd_r", "alt_r", "ctrl_r", "shift_r", "cmd_l", "alt_l", "ctrl_l", "shift_l"]

# Device-independent modifier flag for each hold-key family — used by the
# watchdog to poll whether the modifier is still held, since a missed
# flagsChanged event otherwise leaves the recorder (and the mic) stuck on
# forever. Note: CGEventSourceKeyState does NOT work here (always False for
# this keyboard), so the check must go through NSEvent.modifierFlags().
HOLD_FLAG_FAMILY = {
    "cmd": NSEventModifierFlagCommand,
    "alt": NSEventModifierFlagOption,
    "ctrl": NSEventModifierFlagControl,
    "shift": NSEventModifierFlagShift,
}

# Device-dependent modifier bits in NSEvent.modifierFlags() — tells left/right apart.
HOLD_DEVICE_MASKS = {
    "ctrl_l": 0x00000001,
    "shift_l": 0x00000002,
    "shift_r": 0x00000004,
    "cmd_l": 0x00000008,
    "cmd_r": 0x00000010,
    "alt_l": 0x00000020,
    "alt_r": 0x00000040,
    "ctrl_r": 0x00002000,
}

# Seed dictionary with a few sample proper nouns — replace with your own in
# the 辞書 tab (stored in dictionary.json, which stays local).
DEFAULT_WORDS = [
    "Claude", "ChatGPT", "Obsidian",
]

# Post-transcription replacements: [wrong, right]. Unlike DEFAULT_WORDS (which
# only *bias* Whisper via initial_prompt), these are applied deterministically
# to the transcribed text, so they always win.
DEFAULT_REPLACEMENTS = []

STYLE = """
QWidget { font-size: 13px; color: #232733; }
#root { background: #f4f5f8; }
#sidebar { background: #eceef3; }
#logoName { font-size: 15px; font-weight: 700; color: #1f2430; }
#logoVer  { font-size: 10px; color: #9aa0ae; }
QPushButton#nav {
    text-align: left; padding: 9px 14px; border: none; border-radius: 10px;
    background: transparent; color: #4b5060;
}
QPushButton#nav:hover   { background: #e2e5ec; }
QPushButton#nav:checked { background: #ffffff; color: #4f46e5; font-weight: 600; }
#pageTitle { font-size: 22px; font-weight: 700; color: #1f2430; }
#pageSub   { font-size: 12px; color: #8a90a0; }
#sectionTitle { font-size: 14px; font-weight: 700; color: #1f2430; }
#countPill { font-size: 12px; color: #8a90a0; }
#card    { background: #ffffff; border: 1px solid #e6e8ee; border-radius: 14px; }
#dictRow { background: transparent; border-bottom: 1px solid #eef0f4; }
QLineEdit {
    border: 1px solid #dfe2ea; border-radius: 10px; padding: 9px 12px; background: #ffffff;
}
QLineEdit:focus { border: 1px solid #8b8ff5; }
QComboBox { border: 1px solid #dfe2ea; border-radius: 10px; padding: 7px 10px; background: #ffffff; }
QTextEdit { border: 1px solid #e6e8ee; border-radius: 12px; padding: 10px; background: #ffffff; }
QPushButton#primary {
    border: none; border-radius: 11px; padding: 10px 18px; color: white; font-weight: 600;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5457e6, stop:1 #7c4fe0);
}
QPushButton#ghost {
    border: 1px solid #dfe2ea; border-radius: 10px; padding: 8px 14px;
    background: #ffffff; color: #3b4050;
}
QPushButton#ghost:hover { background: #f2f3f7; }
QPushButton#linkBtn { border: none; background: transparent; color: #b3b8c4; }
QPushButton#linkBtn:hover { color: #ef4444; }
QCheckBox { spacing: 8px; }
QStackedWidget { background: #f4f5f8; }
QScrollArea { border: none; background: #ffffff; }
#dictContainer { background: #ffffff; }
QLabel#word { font-size: 14px; color: #2a2f3a; }
#recordBtn {
    border: none; border-radius: 14px; padding: 16px; color: white; font-size: 15px; font-weight: 700;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
}
#recordBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5457e6, stop:1 #7c4fe0);
}
"""


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def hold_key_is_down(name):
    """Poll whether the hold key's modifier family (e.g. any Control key) is
    currently held, via the live event-stream state. Returns True/False, or
    None if the state cannot be determined (the caller must not act on None).
    Device-independent on purpose: it cannot tell left from right, but for the
    watchdog "the modifier is still held somewhere" is exactly what we need."""
    try:
        flag = HOLD_FLAG_FAMILY.get(name.split("_")[0])
        if flag is None:
            return None
        return bool(int(NSEvent.modifierFlags()) & flag)
    except Exception:
        return None


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_dict():
    """Returns (words, replacements). Older dictionary.json files carry only
    "words" — they load fine with an empty replacement list."""
    if os.path.exists(DICT_PATH):
        try:
            with open(DICT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            repls = [list(p) for p in data.get("replacements", []) if len(p) == 2]
            return data.get("words", []), repls
        except Exception:
            return [], []
    save_dict(DEFAULT_WORDS, DEFAULT_REPLACEMENTS)
    return list(DEFAULT_WORDS), [list(p) for p in DEFAULT_REPLACEMENTS]


def save_dict(words, replacements):
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"words": words, "replacements": replacements},
            f, ensure_ascii=False, indent=2,
        )


def play_sound(name):
    path = f"/System/Library/Sounds/{name}.aiff"
    if os.path.exists(path):
        subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def set_clipboard(text):
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))


def paste_at_cursor():
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_dot_icon(color):
    """A coloured status circle for the menu bar / window (green/red/yellow)."""
    from PySide6.QtGui import QPainter, QBrush, QPen

    size = 36
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor(color)))
    p.setPen(QPen(QColor(0, 0, 0, 70), 2))
    p.drawEllipse(5, 5, size - 10, size - 10)
    p.end()
    return QIcon(pm)


class Recorder:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.frames = queue.Queue()
        self.stream = None
        self.recording = False
        # True while a stream teardown is stuck inside CoreAudio (seen with
        # Bluetooth devices, e.g. soundcore multipoint). Cleared by the
        # teardown thread if/when CoreAudio recovers.
        self.wedged = False

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.put(indata.copy())

    def _open_stream(self):
        return sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32", callback=self._callback
        )

    def start(self):
        # A wedged teardown means CoreAudio's HAL mutex is still held; opening
        # or stopping another stream from here would block on that same mutex
        # and freeze the GUI. Refuse until the teardown thread reports back.
        if self.wedged:
            raise RuntimeError(
                "マイクの停止処理が終わっていません。ヘッドホンの電源を入れ直してからお試しください"
            )
        while not self.frames.empty():
            self.frames.get_nowait()
        # Close any stray stream left behind by an aborted session so the mic
        # can never leak (a leaked stream keeps the orange indicator on).
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        # Open a fresh stream each time so the mic is released between recordings
        # (otherwise macOS keeps showing the orange "mic in use" indicator).
        try:
            self.stream = self._open_stream()
        except sd.PortAudioError:
            # PortAudio enumerates devices once at init; if the default input
            # changed since launch (e.g. Bluetooth headphones connected), the
            # cached device id is stale and open fails with '!obj'. Re-init to
            # re-enumerate, then retry once.
            sd._terminate()
            sd._initialize()
            self.stream = self._open_stream()
        self.stream.start()
        # Only flag as recording once the stream is actually live — if start()
        # raises we must not be left in a phantom "recording" state.
        self.recording = True

    def stop(self, timeout=5.0):
        self.recording = False
        stream, self.stream = self.stream, None
        if stream is not None:
            # PortAudio's stop/close can deadlock forever inside CoreAudio
            # (HALB_Mutex) when a Bluetooth input device wedges, so it runs on
            # a sacrificial thread. On timeout the stream is abandoned — the
            # captured frames below are still returned, so the dictation
            # survives even when the device does not.
            def _teardown():
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                finally:
                    self.wedged = False

            self.wedged = True
            t = threading.Thread(target=_teardown, daemon=True)
            t.start()
            t.join(timeout)
        chunks = []
        while not self.frames.empty():
            chunks.append(self.frames.get_nowait())
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks, axis=0).flatten()


class Bridge(QObject):
    status = Signal(str)
    state = Signal(str)          # "idle" | "recording" | "working"
    transcript = Signal(str)
    hotkey_start = Signal()
    hotkey_stop = Signal()
    hotkey_toggle = Signal()


class MenuHandler(NSObject):
    """Target for the native NSStatusItem menu items."""

    def showWindow_(self, sender):
        if getattr(self, "open_cb", None):
            self.open_cb()

    def quitApp_(self, sender):
        if getattr(self, "quit_cb", None):
            self.quit_cb()


STATE_EMOJI = {"idle": "🎙", "recording": "🔴", "working": "🟡"}

# Strong, process-lifetime references to Cocoa menu-bar objects so macOS never
# releases them (a released status item flashes on screen then vanishes).
_RETAIN = []


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        # New performance settings remain backward-compatible with older
        # config.json files from the initial public release.
        self.config.setdefault("transcribe_temperatures", [0.0, 0.2])
        self.config.setdefault("rewarm_enabled", True)
        self.config.setdefault("rewarm_after_seconds", 600)
        self.dict_words, self.dict_repls = load_dict()
        self.recorder = Recorder(self.config["samplerate"])
        self.busy = False
        self._transcribe = None
        self._qwen_model = None
        self._qwen_model_id = None
        self._model_ready = False
        self._model_lock = threading.Lock()
        self._last_model_use_at = 0.0
        self._rewarm_thread = None
        self._stop_requested_at = None
        self.bridge = Bridge()
        self._monitors = []

        self.setWindowTitle("Voice Dictation")
        self.setObjectName("root")
        self.setMinimumSize(760, 500)
        self.setStyleSheet(STYLE)
        self._build_ui()

        self.bridge.status.connect(self.status_label.setText)
        self.bridge.state.connect(self._apply_state)
        self.bridge.transcript.connect(self._show_transcript)
        self.bridge.hotkey_start.connect(self._start_recording_hotkey)
        self.bridge.hotkey_stop.connect(self._stop_and_transcribe)
        self.bridge.hotkey_toggle.connect(self._toggle)

        # Watchdog: while recording, poll the real key state and enforce a max
        # duration, so a missed release event can never leave the mic stuck on.
        self._started_by_hotkey = False
        self._rec_started_at = 0.0
        self._release_seen = 0
        self._poll_ok = False
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(500)
        self._watchdog.timeout.connect(self._watchdog_tick)

        threading.Thread(target=self._warm_up, daemon=True).start()
        self._restart_listener()

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_home())
        self.stack.addWidget(self._page_dictionary())
        self.stack.addWidget(self._page_settings())
        root.addWidget(self.stack, 1)

        self.nav_group.button(0).setChecked(True)
        self.stack.setCurrentIndex(0)
        self._refresh_dict_list()
        self._refresh_repl_list()

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(196)
        v = QVBoxLayout(side)
        v.setContentsMargins(14, 18, 14, 18)
        v.setSpacing(6)

        logo = QHBoxLayout()
        dot = QLabel()
        dot.setPixmap(make_dot_icon("#6366f1").pixmap(26, 26))
        namebox = QVBoxLayout()
        namebox.setSpacing(0)
        nm = QLabel("Voice Dictation")
        nm.setObjectName("logoName")
        ver = QLabel("ローカル音声入力")
        ver.setObjectName("logoVer")
        namebox.addWidget(nm)
        namebox.addWidget(ver)
        logo.addWidget(dot)
        logo.addSpacing(4)
        logo.addLayout(namebox)
        logo.addStretch(1)
        v.addLayout(logo)
        v.addSpacing(16)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for i, label in enumerate(["ホーム", "辞書", "設定"]):
            b = QPushButton(label)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, idx=i: self.stack.setCurrentIndex(idx))
            self.nav_group.addButton(b, i)
            v.addWidget(b)

        v.addStretch(1)
        return side

    def _page_header(self, title, sub):
        box = QVBoxLayout()
        box.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("pageTitle")
        s = QLabel(sub)
        s.setObjectName("pageSub")
        box.addWidget(t)
        box.addWidget(s)
        return box

    def _page_home(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)
        v.addLayout(self._page_header("ホーム", "キーを押している間だけ録音し、その場に貼り付けます"))

        card = QFrame()
        card.setObjectName("card")
        cl = QHBoxLayout(card)
        cl.setContentsMargins(18, 16, 18, 16)
        self.state_dot = QLabel()
        self.state_dot.setPixmap(make_dot_icon("#34a853").pixmap(24, 24))
        self.status_label = QLabel("準備中…")
        f = QFont()
        f.setPointSize(15)
        self.status_label.setFont(f)
        cl.addWidget(self.state_dot)
        cl.addSpacing(6)
        cl.addWidget(self.status_label, 1)
        v.addWidget(card)

        self.record_btn = QPushButton("●  録音開始 / 停止")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self._toggle)
        v.addWidget(self.record_btn)

        self.transcript_box = QTextEdit()
        self.transcript_box.setPlaceholderText("ここに文字起こし結果が出ます")
        v.addWidget(self.transcript_box, 1)
        return page

    def _page_dictionary(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(14)

        head = QHBoxLayout()
        head.addLayout(self._page_header("辞書", "カスタムワードで認識を寄せ、置換で誤変換を確実に直します"))
        head.addStretch(1)
        self.dict_count = QLabel("0 単語")
        self.dict_count.setObjectName("countPill")
        head.addWidget(self.dict_count, 0, Qt.AlignBottom)
        v.addLayout(head)

        addrow = QHBoxLayout()
        addrow.setSpacing(8)
        self.dict_input = QLineEdit()
        self.dict_input.setPlaceholderText("単語を追加（固有名詞・専門用語など）")
        self.dict_input.returnPressed.connect(self._add_word)
        addbtn = QPushButton("＋ 追加")
        addbtn.setObjectName("primary")
        addbtn.clicked.connect(self._add_word)
        addrow.addWidget(self.dict_input, 1)
        addrow.addWidget(addbtn)
        v.addLayout(addrow)

        card = QFrame()
        card.setObjectName("card")
        cardv = QVBoxLayout(card)
        cardv.setContentsMargins(0, 4, 0, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setObjectName("dictContainer")
        self.dict_list_layout = QVBoxLayout(container)
        self.dict_list_layout.setContentsMargins(0, 0, 0, 0)
        self.dict_list_layout.setSpacing(0)
        scroll.setWidget(container)
        cardv.addWidget(scroll)
        v.addWidget(card, 1)

        # --- replacements: applied to the transcribed text, always win ------
        repl_head = QHBoxLayout()
        repl_title = QLabel("置換ルール（文字起こし後に自動修正）")
        repl_title.setObjectName("sectionTitle")
        repl_head.addWidget(repl_title)
        repl_head.addStretch(1)
        self.repl_count = QLabel("0 件")
        self.repl_count.setObjectName("countPill")
        repl_head.addWidget(self.repl_count, 0, Qt.AlignBottom)
        v.addLayout(repl_head)

        repl_row = QHBoxLayout()
        repl_row.setSpacing(8)
        self.repl_from = QLineEdit()
        self.repl_from.setPlaceholderText("誤変換される語（例: クロード）")
        self.repl_from.returnPressed.connect(lambda: self.repl_to.setFocus())
        arrow = QLabel("→")
        self.repl_to = QLineEdit()
        self.repl_to.setPlaceholderText("正しい表記（例: Claude）")
        self.repl_to.returnPressed.connect(self._add_repl)
        repl_btn = QPushButton("＋ 追加")
        repl_btn.setObjectName("primary")
        repl_btn.clicked.connect(self._add_repl)
        repl_row.addWidget(self.repl_from, 1)
        repl_row.addWidget(arrow)
        repl_row.addWidget(self.repl_to, 1)
        repl_row.addWidget(repl_btn)
        v.addLayout(repl_row)

        repl_card = QFrame()
        repl_card.setObjectName("card")
        repl_cardv = QVBoxLayout(repl_card)
        repl_cardv.setContentsMargins(0, 4, 0, 4)
        repl_scroll = QScrollArea()
        repl_scroll.setWidgetResizable(True)
        repl_container = QWidget()
        repl_container.setObjectName("dictContainer")
        self.repl_list_layout = QVBoxLayout(repl_container)
        self.repl_list_layout.setContentsMargins(0, 0, 0, 0)
        self.repl_list_layout.setSpacing(0)
        repl_scroll.setWidget(repl_container)
        repl_cardv.addWidget(repl_scroll)
        v.addWidget(repl_card, 1)
        return page

    def _page_settings(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)
        v.addLayout(self._page_header("設定", "モデル・入力方式・貼り付けの動作"))

        card = QFrame()
        card.setObjectName("card")
        form = QFormLayout(card)
        form.setContentsMargins(20, 18, 20, 18)
        form.setSpacing(12)

        self.model_cb = QComboBox()
        self.model_cb.addItems(MODELS)
        if self.config["model"] in MODELS:
            self.model_cb.setCurrentText(self.config["model"])
        form.addRow("モデル", self.model_cb)

        self.lang_cb = QComboBox()
        self.lang_cb.addItems(["ja（日本語）", "en（英語）", "自動判定"])
        self.lang_cb.setCurrentIndex({"ja": 0, "en": 1, "": 2}.get(self.config.get("language", "ja"), 0))
        form.addRow("言語", self.lang_cb)

        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["hold（押している間だけ録音）", "toggle（キーで開始・停止）"])
        self.mode_cb.setCurrentIndex(0 if self.config.get("mode", "hold") == "hold" else 1)
        form.addRow("方式", self.mode_cb)

        self.holdkey_cb = QComboBox()
        self.holdkey_cb.addItems(HOLD_KEYS)
        if self.config.get("hold_key", "ctrl_r") in HOLD_KEYS:
            self.holdkey_cb.setCurrentText(self.config["hold_key"])
        form.addRow("holdキー", self.holdkey_cb)

        self.toggle_edit = QLineEdit(self.config.get("toggle_hotkey", "<ctrl>+<alt>+d"))
        form.addRow("toggleキー", self.toggle_edit)

        self.paste_chk = QCheckBox("カーソル位置へ自動貼り付け")
        self.paste_chk.setChecked(self.config.get("paste", True))
        self.sound_chk = QCheckBox("効果音を鳴らす")
        self.sound_chk.setChecked(self.config.get("play_sounds", True))
        form.addRow(self.paste_chk)
        form.addRow(self.sound_chk)

        v.addWidget(card)

        save_btn = QPushButton("設定を保存して反映")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_settings)
        v.addWidget(save_btn, 0, Qt.AlignLeft)
        v.addStretch(1)
        return page

    # ---- dictionary -----------------------------------------------------
    def _refresh_dict_list(self):
        while self.dict_list_layout.count():
            item = self.dict_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for word in self.dict_words:
            row = QFrame()
            row.setObjectName("dictRow")
            h = QHBoxLayout(row)
            h.setContentsMargins(18, 13, 18, 13)
            lbl = QLabel(word)
            lbl.setObjectName("word")
            btn = QPushButton("削除")
            btn.setObjectName("linkBtn")
            btn.clicked.connect(lambda _=False, w=word: self._remove_word(w))
            h.addWidget(lbl)
            h.addStretch(1)
            h.addWidget(btn)
            self.dict_list_layout.addWidget(row)
        self.dict_list_layout.addStretch(1)
        self.dict_count.setText(f"{len(self.dict_words)} 単語")

    def _add_word(self):
        word = self.dict_input.text().strip()
        if word and word not in self.dict_words:
            self.dict_words.append(word)
            save_dict(self.dict_words, self.dict_repls)
            self._refresh_dict_list()
        self.dict_input.clear()

    def _remove_word(self, word):
        if word in self.dict_words:
            self.dict_words.remove(word)
            save_dict(self.dict_words, self.dict_repls)
            self._refresh_dict_list()

    def _refresh_repl_list(self):
        while self.repl_list_layout.count():
            item = self.repl_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for pair in self.dict_repls:
            row = QFrame()
            row.setObjectName("dictRow")
            h = QHBoxLayout(row)
            h.setContentsMargins(18, 13, 18, 13)
            lbl = QLabel(f"{pair[0]}　→　{pair[1]}")
            lbl.setObjectName("word")
            btn = QPushButton("削除")
            btn.setObjectName("linkBtn")
            btn.clicked.connect(lambda _=False, p=pair: self._remove_repl(p))
            h.addWidget(lbl)
            h.addStretch(1)
            h.addWidget(btn)
            self.repl_list_layout.addWidget(row)
        self.repl_list_layout.addStretch(1)
        self.repl_count.setText(f"{len(self.dict_repls)} 件")

    def _add_repl(self):
        src = self.repl_from.text().strip()
        dst = self.repl_to.text().strip()
        if not src or not dst or src == dst:
            return
        pair = [src, dst]
        if pair not in self.dict_repls:
            self.dict_repls.append(pair)
            save_dict(self.dict_words, self.dict_repls)
            self._refresh_repl_list()
        self.repl_from.clear()
        self.repl_to.clear()
        self.repl_from.setFocus()

    def _remove_repl(self, pair):
        if pair in self.dict_repls:
            self.dict_repls.remove(pair)
            save_dict(self.dict_words, self.dict_repls)
            self._refresh_repl_list()

    def _apply_replacements(self, text):
        for src, dst in self.dict_repls:
            if src and src in text:
                text = text.replace(src, dst)
                log(f"replacement: {src} -> {dst}")
        return text

    def _build_prompt(self):
        parts = list(self.dict_words)
        extra = self.config.get("initial_prompt", "")
        if extra:
            parts.append(extra)
        return "、".join(parts)

    # ---- state / transcript --------------------------------------------
    def _apply_state(self, state):
        colors = {"idle": "#34a853", "recording": "#ea4335", "working": "#fbbc04"}
        c = colors.get(state, "#9aa0a6")
        self.state_dot.setPixmap(make_dot_icon(c).pixmap(24, 24))
        btn = getattr(self, "_status_button", None)
        if btn is not None:
            img = getattr(self, "_status_images", {}).get(state)
            if img is not None:
                btn.setImage_(img)
            else:
                btn.setTitle_(STATE_EMOJI.get(state, "🎙"))

    def _show_transcript(self, text):
        self.transcript_box.setPlainText(text)

    # ---- model ----------------------------------------------------------
    def _warm_up(self):
        self.bridge.status.emit("モデル読み込み中…（初回はダウンロードします）")
        model_id = self.config["model"]
        try:
            started = time.perf_counter()
            if model_backend(model_id) == "qwen":
                log("warm-up: importing mlx_audio")
                from mlx_audio.stt.utils import load_model

                with self._model_lock:
                    if self._qwen_model_id != model_id:
                        self._qwen_model = None  # free the old one first
                        self._qwen_model_id = None
                        self._qwen_model = load_model(model_id)
                        self._qwen_model_id = model_id
                    self._model_ready = True
                    log("warm-up: running dummy transcription")
                    self._qwen_model.generate(
                        np.zeros(16000, dtype=np.float32),
                        language=self.config.get("language") or None,
                    )
                    self._last_model_use_at = time.monotonic()
            else:
                log("warm-up: importing mlx_whisper")
                import mlx_whisper

                self._transcribe = mlx_whisper.transcribe
                self._model_ready = True
                log("warm-up: running dummy transcription")
                with self._model_lock:
                    # A previously selected Qwen model is no longer needed —
                    # drop it so its weights don't sit in memory.
                    self._qwen_model = None
                    self._qwen_model_id = None
                    self._transcribe(
                        np.zeros(16000, dtype=np.float32),
                        path_or_hf_repo=model_id,
                        language=self.config.get("language") or None,
                        temperature=0.0,
                    )
                    self._last_model_use_at = time.monotonic()
            log(f"timing: initial_warmup={(time.perf_counter() - started) * 1000:.0f}ms")
        except Exception as e:
            log(f"warm-up failed: {e}")
            self.bridge.status.emit(f"モデル読込エラー: {e}")
            return
        log("warm-up: done")
        self.bridge.status.emit("待機中（ホットキーまたはボタンで開始）")

    def _maybe_rewarm(self):
        """Warm the model only after a long idle period, overlapping the work
        with the user's speech. There is no periodic background activity."""
        if not self.config.get("rewarm_enabled", True) or not self._model_ready:
            return
        threshold = max(0, float(self.config.get("rewarm_after_seconds", 600)))
        idle = time.monotonic() - self._last_model_use_at
        if self._last_model_use_at <= 0 or idle < threshold:
            return
        if self._rewarm_thread is not None and self._rewarm_thread.is_alive():
            return

        def run():
            started = time.perf_counter()
            log(f"re-warm: starting after {idle:.0f}s idle")
            try:
                with self._model_lock:
                    if model_backend(self.config["model"]) == "qwen":
                        if self._qwen_model is None:
                            return
                        self._qwen_model.generate(
                            np.zeros(16000, dtype=np.float32),
                            language=self.config.get("language") or None,
                        )
                    else:
                        self._transcribe(
                            np.zeros(16000, dtype=np.float32),
                            path_or_hf_repo=self.config["model"],
                            language=self.config.get("language") or None,
                            temperature=0.0,
                        )
                    self._last_model_use_at = time.monotonic()
                log(f"timing: rewarm={(time.perf_counter() - started) * 1000:.0f}ms")
            except Exception as e:
                # Re-warming is an optimization only. A failure must never
                # prevent the real recording from being transcribed.
                log(f"re-warm failed: {e}")

        self._rewarm_thread = threading.Thread(target=run, daemon=True)
        self._rewarm_thread.start()

    # ---- hotkey ---------------------------------------------------------
    def _remove_monitors(self):
        for m in self._monitors:
            try:
                NSEvent.removeMonitor_(m)
            except Exception:
                pass
        self._monitors = []

    def _add_monitors(self, mask, on_event):
        gm = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_event)

        def local(e):
            on_event(e)
            return e

        lm = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, local)
        self._monitors = [m for m in (gm, lm) if m is not None]

    @staticmethod
    def _parse_toggle(s):
        mask = 0
        key_char = "d"
        for tok in s.replace(" ", "").split("+"):
            t = tok.strip("<>").lower()
            if t in ("ctrl", "control"):
                mask |= NSEventModifierFlagControl
            elif t in ("alt", "option", "opt"):
                mask |= NSEventModifierFlagOption
            elif t in ("cmd", "command", "meta"):
                mask |= NSEventModifierFlagCommand
            elif t == "shift":
                mask |= NSEventModifierFlagShift
            elif t:
                key_char = t[-1:]
        return mask, key_char

    def _restart_listener(self):
        self._remove_monitors()
        mode = self.config.get("mode", "hold")
        if mode == "hold":
            mask = HOLD_DEVICE_MASKS.get(self.config.get("hold_key", "ctrl_r"))
            if mask is None:
                self.bridge.status.emit("不明なholdキーです")
                return

            def on_event(event):
                down = bool(int(event.modifierFlags()) & mask)
                if down and not self.recorder.recording and not self.busy:
                    self.bridge.hotkey_start.emit()
                elif not down and self.recorder.recording:
                    self.bridge.hotkey_stop.emit()

            self._add_monitors(NSEventMaskFlagsChanged, on_event)
        else:
            req_mask, key_char = self._parse_toggle(
                self.config.get("toggle_hotkey", "<ctrl>+<alt>+d")
            )

            def on_event(event):
                chars = (event.charactersIgnoringModifiers() or "").lower()
                flags = int(event.modifierFlags())
                if chars == key_char and (flags & req_mask) == req_mask:
                    self.bridge.hotkey_toggle.emit()

            self._add_monitors(NSEventMaskKeyDown, on_event)

    def _toggle(self):
        if self.busy:
            return
        if self.recorder.recording:
            self._stop_and_transcribe()
        else:
            self._start_recording()

    # ---- record / transcribe -------------------------------------------
    def _start_recording_hotkey(self):
        self._start_recording(via_hotkey=True)

    def _start_recording(self, via_hotkey=False):
        if not self._model_ready:
            self.bridge.status.emit("モデル準備中… 少しお待ちください")
            return
        try:
            self.recorder.start()
        except Exception as e:
            log(f"recorder start failed: {e}")
            self.bridge.status.emit(f"録音を開始できませんでした: {e}")
            self.bridge.state.emit("idle")
            return
        self._started_by_hotkey = via_hotkey
        self._rec_started_at = time.time()
        self._release_seen = 0
        # Self-calibrate the release sensor: at this instant the key IS down
        # (the press event just fired). If polling can't see that, it is not
        # trustworthy on this system — fall back to the max-duration cap only.
        self._poll_ok = False
        if via_hotkey and self.config.get("mode", "hold") == "hold":
            self._poll_ok = hold_key_is_down(self.config.get("hold_key", "ctrl_r")) is True
            if not self._poll_ok:
                log("watchdog: key-state polling unreliable here — release polling off, max-duration cap only")
        self._watchdog.start()
        log(f"recording started ({'hotkey' if via_hotkey else 'button'})")
        self.bridge.state.emit("recording")
        self.bridge.status.emit("録音中…")
        self._maybe_rewarm()
        if self.config.get("play_sounds", True):
            play_sound("Tink")

    def _watchdog_tick(self):
        if not self.recorder.recording:
            self._watchdog.stop()
            return
        elapsed = time.time() - self._rec_started_at
        if elapsed > self.config.get("max_record_seconds", 300):
            log(f"watchdog: max duration reached ({elapsed:.0f}s) — auto stop")
            self._stop_and_transcribe()
            return
        if self._poll_ok and self._started_by_hotkey and self.config.get("mode", "hold") == "hold":
            down = hold_key_is_down(self.config.get("hold_key", "ctrl_r"))
            # Require two consecutive "released" readings before acting, so a
            # single misread can never cut a live recording short.
            if down is False:
                self._release_seen += 1
                if self._release_seen >= 2:
                    log(f"watchdog: release event was missed ({elapsed:.0f}s in) — auto stop")
                    self._stop_and_transcribe()
            else:
                self._release_seen = 0

    def _stop_and_transcribe(self):
        self._watchdog.stop()
        self._stop_requested_at = time.perf_counter()
        # Flip the flags here so the hotkey/watchdog handlers immediately see
        # the recording as finished, but keep the actual stream teardown off
        # the main thread — it can block inside CoreAudio (see Recorder.stop)
        # and must never take the GUI down with it.
        self.recorder.recording = False
        self.busy = True
        self.bridge.state.emit("working")
        self.bridge.status.emit("文字起こし中…")
        threading.Thread(target=self._finish_and_transcribe, daemon=True).start()

    def _finish_and_transcribe(self):
        stop_started = time.perf_counter()
        audio = self.recorder.stop()
        stop_ms = (time.perf_counter() - stop_started) * 1000
        log(f"recording stopped ({audio.size / self.config['samplerate']:.1f}s of audio)")
        log(f"timing: recorder_stop={stop_ms:.0f}ms")
        if self.recorder.wedged:
            log("recorder: stream teardown timed out — audio kept, stream abandoned")
            self.bridge.status.emit(
                "マイクの停止に失敗しました（ヘッドホンの電源を入れ直すと復旧します）"
            )
        self._do_transcribe(audio)

    def _do_transcribe(self, audio):
        total_started = time.perf_counter()
        inference_ms = 0.0
        lock_wait_ms = 0.0
        postprocess_ms = 0.0
        clipboard_ms = 0.0
        paste_ms = 0.0
        try:
            # Below ~0.5s Whisper tends to hallucinate long text out of the
            # dictionary prompt and paste garbage — treat as an accidental tap.
            if audio.size < self.config["samplerate"] * 0.5:
                self.bridge.status.emit("録音が短すぎます")
                return
            # Silence gate: Whisper + initial_prompt hallucinates hundreds of
            # characters out of dead air, which would get pasted as garbage.
            peak = float(np.abs(audio).max())
            if peak < 0.015:
                log(f"silence gate: peak={peak:.4f} — skipping transcription")
                self.bridge.status.emit("（無音でした）")
                return
            model_id = self.config["model"]
            backend = model_backend(model_id)
            lang = self.config.get("language") or None
            prompt = self._build_prompt()
            temperatures = self.config.get("transcribe_temperatures", [0.0, 0.2])
            if isinstance(temperatures, (int, float)):
                temperatures = (float(temperatures),)
            else:
                temperatures = tuple(float(t) for t in temperatures) or (0.0, 0.2)

            if backend == "qwen":
                lock_started = time.perf_counter()
                with self._model_lock:
                    lock_wait_ms = (time.perf_counter() - lock_started) * 1000
                    if self._qwen_model is None or self._qwen_model_id != model_id:
                        raise RuntimeError("モデルを読み込み中です。少し待ってからもう一度お試しください")
                    inference_started = time.perf_counter()
                    # The dictionary goes in as Qwen3-ASR context biasing —
                    # unlike Whisper's initial_prompt it cannot leak into the
                    # output, it only steers how words are spelled. The
                    # 専門用語 label matters: without it the bias misses terms
                    # the raw word list alone would catch.
                    result = self._qwen_model.generate(
                        audio,
                        language=lang,
                        temperature=temperatures[0],
                        system_prompt=f"専門用語: {prompt}" if prompt else None,
                    )
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    self._last_model_use_at = time.monotonic()
                postprocess_started = time.perf_counter()
                text = (result.text or "").strip()
            else:
                kwargs = {"path_or_hf_repo": model_id}
                if lang:
                    kwargs["language"] = lang
                if prompt:
                    kwargs["initial_prompt"] = prompt
                kwargs["temperature"] = temperatures if len(temperatures) > 1 else temperatures[0]

                lock_started = time.perf_counter()
                with self._model_lock:
                    lock_wait_ms = (time.perf_counter() - lock_started) * 1000
                    inference_started = time.perf_counter()
                    result = self._transcribe(audio, **kwargs)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    self._last_model_use_at = time.monotonic()
                postprocess_started = time.perf_counter()
                # Hallucination filter (standard Whisper heuristic): drop segments
                # that are probably not speech, otherwise dead air + the dictionary
                # prompt turns into hundreds of characters of pasted garbage.
                segments = result.get("segments") or []
                if segments:
                    kept = [
                        s for s in segments
                        if not (
                            s.get("no_speech_prob", 0.0) > 0.6
                            and s.get("avg_logprob", 0.0) < -1.0
                        )
                    ]
                    if len(kept) < len(segments):
                        log(f"hallucination filter: dropped {len(segments) - len(kept)}/{len(segments)} segments")
                    text = "".join(s.get("text", "") for s in kept).strip()
                    used_temperatures = sorted({float(s.get("temperature") or 0.0) for s in segments})
                    log(f"decode: temperatures={used_temperatures}")
                else:
                    text = (result.get("text") or "").strip()
            if not text:
                self.bridge.status.emit("（無音でした）")
                return
            text = self._apply_replacements(text)
            postprocess_ms = (time.perf_counter() - postprocess_started) * 1000

            self.bridge.transcript.emit(text)
            clipboard_started = time.perf_counter()
            set_clipboard(text)
            clipboard_ms = (time.perf_counter() - clipboard_started) * 1000
            if self.config.get("paste", True):
                paste_started = time.perf_counter()
                time.sleep(0.15)
                paste_at_cursor()
                paste_ms = (time.perf_counter() - paste_started) * 1000
            if self.config.get("play_sounds", True):
                play_sound("Pop")
            self.bridge.status.emit("✓ 完了（クリップボードにも入っています）")
            log(f"transcribed {len(text)} chars")
        except Exception as e:
            log(f"transcribe error: {e}")
            self.bridge.status.emit(f"エラー: {e}")
        finally:
            total_ms = (time.perf_counter() - total_started) * 1000
            since_stop_ms = (
                (time.perf_counter() - self._stop_requested_at) * 1000
                if self._stop_requested_at is not None else total_ms
            )
            log(
                "timing: "
                f"model_lock_wait={lock_wait_ms:.0f}ms, "
                f"inference={inference_ms:.0f}ms, "
                f"postprocess={postprocess_ms:.0f}ms, "
                f"clipboard={clipboard_ms:.0f}ms, "
                f"paste={paste_ms:.0f}ms, "
                f"transcribe_total={total_ms:.0f}ms, "
                f"stop_to_done={since_stop_ms:.0f}ms"
            )
            self.busy = False
            self.bridge.state.emit("idle")

    # ---- settings -------------------------------------------------------
    def _save_settings(self):
        self.config["model"] = self.model_cb.currentText()
        self.config["language"] = ["ja", "en", ""][self.lang_cb.currentIndex()]
        self.config["mode"] = "hold" if self.mode_cb.currentIndex() == 0 else "toggle"
        self.config["hold_key"] = self.holdkey_cb.currentText()
        self.config["toggle_hotkey"] = self.toggle_edit.text().strip()
        self.config["paste"] = self.paste_chk.isChecked()
        self.config["play_sounds"] = self.sound_chk.isChecked()
        save_config(self.config)
        # Shut down any in-flight recording before swapping the recorder —
        # replacing it while a stream is open would leak the mic (orange dot
        # stays on and the app looks stuck on "recording").
        self._watchdog.stop()
        if self.recorder.recording:
            self.recorder.stop()
            self.bridge.state.emit("idle")
        self.recorder = Recorder(self.config["samplerate"])
        self._restart_listener()
        self.bridge.status.emit("設定を保存しました。モデル変更時は次回録音で読み込みます")
        threading.Thread(target=self._warm_up, daemon=True).start()

    def closeEvent(self, event):
        event.ignore()
        self.hide()


def _force_light_theme(app):
    """Keep the app light regardless of the system dark-mode setting."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f4f5f8"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f4f5f8"))
    pal.setColor(QPalette.Text, QColor("#232733"))
    pal.setColor(QPalette.WindowText, QColor("#232733"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor("#232733"))
    pal.setColor(QPalette.PlaceholderText, QColor("#9aa0ae"))
    pal.setColor(QPalette.Highlight, QColor("#6366f1"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    try:
        from AppKit import NSApplication, NSAppearance

        aqua = NSAppearance.appearanceNamed_("NSAppearanceNameAqua")
        NSApplication.sharedApplication().setAppearance_(aqua)
    except Exception:
        pass


def _make_status_item(win, open_window, quit_app):
    """Native macOS menu-bar item (reliable where QSystemTrayIcon is not)."""
    from AppKit import NSStatusBar, NSMenu, NSMenuItem, NSVariableStatusItemLength

    handler = MenuHandler.alloc().init()
    handler.open_cb = open_window
    handler.quit_cb = quit_app

    from AppKit import NSImage
    from Foundation import NSSize

    bar = NSStatusBar.systemStatusBar()
    item = bar.statusItemWithLength_(NSVariableStatusItemLength)
    try:
        item.setVisible_(True)
    except Exception:
        pass
    button = item.button()

    win._status_images = {}
    for state, fname in (
        ("idle", "menubar_idle.png"),
        ("recording", "menubar_recording.png"),
        ("working", "menubar_working.png"),
    ):
        img = NSImage.alloc().initWithContentsOfFile_(os.path.join(HERE, fname))
        if img is not None and img.isValid():
            img.setSize_(NSSize(18, 18))
            if state == "idle":
                img.setTemplate_(True)  # adapts: white on dark bar, black on light
            win._status_images[state] = img

    if win._status_images.get("idle") is not None:
        button.setImage_(win._status_images["idle"])
    else:
        button.setTitle_("🎙")
    button.setToolTip_("Voice Dictation")

    menu = NSMenu.alloc().init()
    mi_show = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "ウィンドウを表示", "showWindow:", ""
    )
    mi_show.setTarget_(handler)
    menu.addItem_(mi_show)
    menu.addItem_(NSMenuItem.separatorItem())
    mi_quit = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("終了", "quitApp:", "q")
    mi_quit.setTarget_(handler)
    menu.addItem_(mi_quit)
    item.setMenu_(menu)

    # keep strong references so nothing gets garbage-collected
    win._status_bar = bar
    win._status_item = item
    win._status_button = item.button()
    win._menu_handler = handler
    win._status_menu = menu
    _RETAIN.extend([bar, item, item.button(), handler, menu])
    _RETAIN.extend(win._status_images.values())


def main():
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    _force_light_theme(app)

    win = MainWindow()

    def open_window():
        win.show()
        win.raise_()
        win.activateWindow()
        try:
            from AppKit import NSApplication

            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass

    def go_accessory():
        # Hide from the Dock and Command+Tab, AFTER the status item has drawn,
        # then re-assert the item so it redraws under the new policy.
        try:
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

            NSApplication.sharedApplication().setActivationPolicy_(
                NSApplicationActivationPolicyAccessory
            )
            item = getattr(win, "_status_item", None)
            if item is not None:
                item.setVisible_(False)
                item.setVisible_(True)
                img = getattr(win, "_status_images", {}).get("idle")
                if img is not None:
                    item.button().setImage_(img)
        except Exception:
            pass

    def setup_status():
        try:
            _make_status_item(win, open_window, app.quit)
            # Draw the item as a regular app first, then switch to accessory
            # (hides Dock + Command+Tab) and re-assert so the icon survives.
            if os.environ.get("VD_ACCESSORY", "0") == "1":
                QTimer.singleShot(1500, go_accessory)
        except Exception:
            pass

    # Create the menu-bar item once the event loop is running (needed for it to show).
    QTimer.singleShot(0, setup_status)
    win._open_window = open_window

    win.show()
    app.exec()


if __name__ == "__main__":
    main()
