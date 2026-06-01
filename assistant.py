"""
============================================================
  Global Background Dictation Assistant  v2.0
============================================================
  Runs silently in the Windows System Tray.

  USAGE:
    Press the hotkey ONCE  → starts continuous listening.
    Speak naturally — text is auto-injected as you speak.
    Press the hotkey AGAIN → stops listening.

  Default hotkey : Ctrl + Alt + V  (change in Settings)
  Tray icon      : Right-click for menu

  Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Main Thread       → Hidden CTk root + Tkinter loop     │
  │  TrayThread        → pystray system tray icon           │
  │  ContinuousRecorder→ mic capture + auto-chunking        │
  │  TranscribeWorker  → faster-whisper inference queue     │
  │  keyboard lib      → global hotkey hooks                │
  └─────────────────────────────────────────────────────────┘

  Auto-Transcribe Flow (per 3.5-second audio chunk):
    Mic → ContinuousRecorder → TranscribeWorker
        → TextInjector (Ctrl+V paste) → repeat
============================================================
"""

# ── Standard Library ───────────────────────────────────────
import threading
import queue
import time
import sys
import os
import json
import winsound   # Windows built-in beep — no pip install

# ── Audio & ML ────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# ── Automation ────────────────────────────────────────────
import keyboard
import pyperclip

# ── System Tray ───────────────────────────────────────────
import pystray
from PIL import Image, ImageDraw

# ── GUI ───────────────────────────────────────────────────
import tkinter as tk
import customtkinter as ctk


# ============================================================
#  FALLBACK DEFAULTS (overridden by config.json if present)
# ============================================================

DEFAULT_HOTKEY              = "ctrl+alt+v"
DEFAULT_MODEL_SIZE_GPU      = "small.en"
DEFAULT_MODEL_SIZE_CPU      = "base.en"
DEFAULT_AUDIO_SEGMENT_SEC   = 3.5      # seconds per auto-transcription chunk
DEFAULT_CLIPBOARD_DELAY     = 0.8      # seconds before restoring clipboard

SAMPLE_RATE  = 16000
CHANNELS     = 1
BLOCK_SIZE   = 4000
VAD_FILTER   = True

# Audio cue frequencies (Hz) and durations (ms)
BEEP_START  = (880,  130)   # High  = listening started
BEEP_STOP   = (440,  130)   # Low   = listening stopped
BEEP_INJECT = (1200,  70)   # Ping  = text injected

# Overlay position from top-right corner
OVERLAY_RIGHT_MARGIN = 250
OVERLAY_TOP_MARGIN   = 20

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


# ============================================================
#  CONFIG MANAGER — Reads/writes config.json
# ============================================================

class ConfigManager:
    """
    Loads user preferences from config.json at startup.
    Falls back to defaults if the file doesn't exist.
    Saves changes immediately on set().
    """

    DEFAULTS = {
        "hotkey"              : DEFAULT_HOTKEY,
        "model_size_gpu"      : DEFAULT_MODEL_SIZE_GPU,
        "model_size_cpu"      : DEFAULT_MODEL_SIZE_CPU,
        "audio_segment_sec"   : DEFAULT_AUDIO_SEGMENT_SEC,
        "clipboard_delay"     : DEFAULT_CLIPBOARD_DELAY,
    }

    def __init__(self, path: str = CONFIG_FILE):
        self._path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                return {**self.DEFAULTS, **saved}   # saved values override defaults
            except Exception as e:
                print(f"[Config] Could not read config.json: {e}. Using defaults.")
        return dict(self.DEFAULTS)

    def save(self):
        """Persist current config to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"[Config] Could not save config.json: {e}")

    def get(self, key: str):
        return self._data.get(key, self.DEFAULTS.get(key))

    def set(self, key: str, value):
        self._data[key] = value
        self.save()


# ============================================================
#  HARDWARE DETECTION
# ============================================================

def detect_device(cfg: ConfigManager) -> tuple[str, str, str]:
    """
    Detect whether a CUDA GPU is available and pick the right model.

    Returns:
        (device, compute_type, model_size)
    """
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        model = cfg.get("model_size_gpu")
        print(f"[HW] GPU: {name} -> CUDA int8_float16 / {model}")
        return "cuda", "int8_float16", model
    else:
        model = cfg.get("model_size_cpu")
        print(f"[HW] No GPU -> CPU int8 / {model}")
        return "cpu", "int8", model


# ============================================================
#  MODEL LOADER
# ============================================================

def load_model(device: str, compute_type: str, model_size: str) -> WhisperModel:
    print(f"[Model] Loading {model_size} on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=4)
    print("[Model] Model ready.")
    return model


# ============================================================
#  CONTINUOUS RECORDER
# ============================================================

class ContinuousRecorder:
    """
    Records microphone audio indefinitely and automatically slices it into
    fixed-length chunks (AUDIO_SEGMENT_SEC seconds each).

    Each chunk is placed into `chunk_queue` for the TranscriptionWorker
    to process. This creates a seamless, auto-transcribing pipeline —
    no second hotkey press needed per sentence.

    Timeline while listening:
      [0s ─── 3.5s] chunk 1 → queue → Whisper → inject
      [3.5s ─ 7.0s] chunk 2 → queue → Whisper → inject
      [...continues until stop() is called...]
    """

    def __init__(self, chunk_queue: queue.Queue, segment_sec: float = DEFAULT_AUDIO_SEGMENT_SEC):
        self._queue       = chunk_queue
        self._segment_sec = segment_sec
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Begin continuous mic capture."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ContinuousRecorder")
        self._thread.start()

    def stop(self):
        """Stop recording. Flushes any remaining audio as a final chunk."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run(self):
        """
        Internal recording loop.
        sounddevice fires _callback on its own C-level audio thread —
        we ONLY enqueue raw data there, never do heavy work.
        """
        internal_q    = queue.Queue()
        frames_needed = int(SAMPLE_RATE * self._segment_sec)
        buffer: list[np.ndarray] = []
        frame_count   = 0

        def _callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                print(f"[Audio] {status}")
            internal_q.put(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=_callback,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = internal_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                buffer.append(chunk.flatten())
                frame_count += chunk.shape[0]

                # Enough audio for one transcription chunk?
                if frame_count >= frames_needed:
                    audio = np.concatenate(buffer)
                    self._queue.put(("chunk", audio))
                    buffer      = []
                    frame_count = 0

        # Flush remaining audio (could be a partial last sentence)
        if buffer:
            audio = np.concatenate(buffer)
            min_frames = int(SAMPLE_RATE * 0.5)   # ignore < 0.5s of audio
            if len(audio) >= min_frames:
                self._queue.put(("chunk", audio))

        # Signal that recording has fully stopped
        self._queue.put(("done", None))


# ============================================================
#  TEXT INJECTOR
# ============================================================

class TextInjector:
    """
    Pastes text into the currently focused window via clipboard.

    Steps:
      1. Save original clipboard content.
      2. Set transcribed text as clipboard.
      3. Simulate Ctrl+V.
      4. Restore original clipboard after a delay.

    This approach works universally: WhatsApp, Chrome, Notion,
    VS Code, Windows Explorer, Notepad, etc.
    """

    def __init__(self, restore_delay: float = DEFAULT_CLIPBOARD_DELAY):
        self._delay = restore_delay

    def inject(self, text: str):
        if not text.strip():
            return
        try:
            original = pyperclip.paste()
        except Exception:
            original = ""
        try:
            pyperclip.copy(text)
            time.sleep(0.06)
            keyboard.send("ctrl+v")
        except Exception as e:
            print(f"[Injector] Error: {e}")
        finally:
            def _restore():
                time.sleep(self._delay)
                try:
                    pyperclip.copy(original)
                except Exception:
                    pass
            threading.Thread(target=_restore, daemon=True).start()


# ============================================================
#  FLOATING OVERLAY — on-screen status chip
# ============================================================

class FloatingOverlay:
    """
    Frameless always-on-top mini-window in the top-right corner.
    Shows: Listening / Transcribing state.
    All _impl methods MUST run on the Tkinter main thread.
    """

    def __init__(self, root: tk.Tk):
        self._root  = root
        self._win   = None
        self._label = None

    def show(self, message: str, color: str = "#E74C3C"):
        self._root.after(0, self._show_impl, message, color)

    def hide(self):
        self._root.after(0, self._hide_impl)

    def update_text(self, message: str, color: str):
        self._root.after(0, self._update_impl, message, color)

    def _show_impl(self, message: str, color: str):
        if self._win is None or not self._alive():
            self._create()
        self._label.configure(fg=color, text=message)
        self._win.deiconify()
        self._win.lift()
        self._win.attributes("-topmost", True)

    def _hide_impl(self):
        if self._win and self._alive():
            self._win.withdraw()

    def _update_impl(self, message: str, color: str):
        if self._win and self._alive():
            self._label.configure(fg=color, text=message)

    def _alive(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def _create(self):
        self._win = tk.Toplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.attributes("-alpha", 0.93)
        self._win.configure(bg="#0F1117")
        sw = self._win.winfo_screenwidth()
        self._win.geometry(f"230x46+{sw - OVERLAY_RIGHT_MARGIN}+{OVERLAY_TOP_MARGIN}")
        self._label = tk.Label(
            self._win, text="", font=("Segoe UI", 13, "bold"),
            fg="#E74C3C", bg="#0F1117", padx=14, pady=11,
        )
        self._label.pack(fill="both", expand=True)
        self._win.withdraw()


# ============================================================
#  HOTKEY RECORDER WIDGET
# ============================================================

class HotkeyRecorder(ctk.CTkFrame):
    """
    A custom widget that lets the user record a new hotkey combo.

    How it works:
      1. User clicks [Change] button.
      2. Widget grabs keyboard focus on the parent window.
      3. Next key combo pressed (modifier + key) is captured.
      4. Display updates and recording stops automatically.
      5. Click [Save] in the Settings window to apply.
    """

    def __init__(self, parent, initial: str = "ctrl+alt+v", **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._hotkey    = initial
        self._recording = False
        self._bind_ids: list[str] = []

        # -- Layout: [hotkey label]  [Change button]
        self.grid_columnconfigure(0, weight=1)

        self._display = ctk.CTkLabel(
            self,
            text=self._format(initial),
            font=ctk.CTkFont("Consolas", 15, "bold"),
            text_color="#6C63FF",
            fg_color="#1A1D27",
            corner_radius=8,
            padx=12,
            pady=8,
            anchor="center",
        )
        self._display.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self._btn = ctk.CTkButton(
            self,
            text="Change",
            width=90,
            height=36,
            fg_color="#2A2D3E",
            hover_color="#353850",
            command=self._start,
        )
        self._btn.grid(row=0, column=1)

    @staticmethod
    def _format(hotkey: str) -> str:
        """Convert 'ctrl+alt+v' → 'CTRL + ALT + V' for display."""
        return "  " + "  +  ".join(k.upper() for k in hotkey.split("+")) + "  "

    def _start(self):
        self._recording = True
        self._display.configure(text="  Press your hotkey...  ", text_color="#F39C12")
        self._btn.configure(text="Cancel", command=self._cancel)
        # Grab focus on the parent toplevel and listen for key presses
        top = self.winfo_toplevel()
        top.focus_force()
        self._bind_ids.append(top.bind("<KeyPress>", self._on_key, add=True))

    def _on_key(self, event):
        if not self._recording:
            return
        modifier_syms = {
            "control_l", "control_r", "shift_l", "shift_r",
            "alt_l", "alt_r", "meta_l", "meta_r",
            "super_l", "super_r", "caps_lock",
        }
        key = event.keysym.lower()
        if key in modifier_syms:
            return   # Wait for a non-modifier key

        parts = []
        # event.state bitmask: Ctrl=4, Shift=1, Alt=0x20000
        if event.state & 0x4:
            parts.append("ctrl")
        if event.state & 0x20000:
            parts.append("alt")
        if event.state & 0x1:
            parts.append("shift")

        # Map tkinter keysym to keyboard lib format
        key_map = {"space": "space", "return": "enter", "escape": "esc",
                   "tab": "tab", "backspace": "backspace", "delete": "delete"}
        key = key_map.get(key, key)

        parts.append(key)
        self._hotkey = "+".join(parts)
        self._display.configure(text=self._format(self._hotkey), text_color="#6C63FF")
        self._finish()

    def _cancel(self):
        self._display.configure(text=self._format(self._hotkey), text_color="#6C63FF")
        self._finish()

    def _finish(self):
        self._recording = False
        self._btn.configure(text="Change", command=self._start)
        top = self.winfo_toplevel()
        for bid in self._bind_ids:
            try:
                top.unbind("<KeyPress>", bid)
            except Exception:
                pass
        self._bind_ids.clear()

    def get(self) -> str:
        """Return the currently set hotkey string."""
        return self._hotkey

    def set(self, hotkey: str):
        """Programmatically set the displayed hotkey."""
        self._hotkey = hotkey
        self._display.configure(text=self._format(hotkey), text_color="#6C63FF")


# ============================================================
#  SETTINGS WINDOW
# ============================================================

class SettingsWindow:
    """
    Modal settings panel with hotkey recorder and audio controls.
    Changes take effect immediately after clicking [Save Settings].
    """

    def __init__(self, root: tk.Tk, cfg: ConfigManager, on_hotkey_changed):
        self._root             = root
        self._cfg              = cfg
        self._on_hotkey_changed = on_hotkey_changed
        self._win              = None

    def show(self):
        self._root.after(0, self._show_impl)

    def _show_impl(self):
        if self._win and self._alive():
            self._win.deiconify()
            self._win.lift()
            return
        self._create()

    def _alive(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def _create(self):
        self._win = ctk.CTkToplevel(self._root)
        self._win.title("Settings — Dictation Assistant")
        self._win.geometry("480x420")
        self._win.resizable(False, False)
        self._win.configure(fg_color="#0F1117")
        self._win.attributes("-topmost", True)
        self._win.grab_set()   # Modal behavior

        # Center on screen
        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        self._win.geometry(f"480x420+{(sw-480)//2}+{(sh-420)//2}")

        # ── Header ──────────────────────────────────────
        hdr = ctk.CTkFrame(self._win, fg_color="#1A1D27", corner_radius=0, height=54)
        hdr.pack(fill="x")
        ctk.CTkLabel(
            hdr, text="  Settings",
            font=ctk.CTkFont("Segoe UI", 18, "bold"), text_color="#E8E8F0",
        ).pack(side="left", padx=18, pady=14)

        # ── Content ──────────────────────────────────────
        body = ctk.CTkScrollableFrame(self._win, fg_color="#0F1117", corner_radius=0)
        body.pack(fill="both", expand=True, padx=20, pady=10)
        body.grid_columnconfigure(0, weight=1)

        # ── Section: Hotkey ─────────────────────────────
        self._section_label(body, "Global Hotkey", 0)
        ctk.CTkLabel(
            body,
            text="Press [Change] then press your desired key combination.",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color="#8888AA", anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))

        self._hotkey_rec = HotkeyRecorder(body, initial=self._cfg.get("hotkey"))
        self._hotkey_rec.grid(row=2, column=0, sticky="ew", pady=(0, 20))

        # ── Section: Model ───────────────────────────────
        self._section_label(body, "Whisper Model (GPU)", 3)
        gpu_models = ["tiny.en", "base.en", "small.en", "medium.en"]
        self._gpu_model_var = ctk.StringVar(value=self._cfg.get("model_size_gpu"))
        ctk.CTkOptionMenu(
            body, values=gpu_models, variable=self._gpu_model_var,
            fg_color="#1A1D27", button_color="#6C63FF", button_hover_color="#4E46C1",
        ).grid(row=4, column=0, sticky="ew", pady=(4, 16))

        self._section_label(body, "Whisper Model (CPU fallback)", 5)
        cpu_models = ["tiny.en", "base.en", "small.en"]
        self._cpu_model_var = ctk.StringVar(value=self._cfg.get("model_size_cpu"))
        ctk.CTkOptionMenu(
            body, values=cpu_models, variable=self._cpu_model_var,
            fg_color="#1A1D27", button_color="#6C63FF", button_hover_color="#4E46C1",
        ).grid(row=6, column=0, sticky="ew", pady=(4, 16))

        # ── Section: Audio Chunk Length ──────────────────
        self._section_label(body, "Auto-Transcribe Chunk Length", 7)
        self._chunk_var = tk.DoubleVar(value=self._cfg.get("audio_segment_sec"))
        chunk_frame = ctk.CTkFrame(body, fg_color="transparent")
        chunk_frame.grid(row=8, column=0, sticky="ew", pady=(4, 4))
        chunk_frame.grid_columnconfigure(0, weight=1)

        self._chunk_slider = ctk.CTkSlider(
            chunk_frame, from_=1.5, to=6.0, number_of_steps=9,
            variable=self._chunk_var,
            progress_color="#6C63FF", button_color="#6C63FF",
            command=self._on_chunk_change,
        )
        self._chunk_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self._chunk_lbl = ctk.CTkLabel(
            chunk_frame,
            text=f"{self._chunk_var.get():.1f}s",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color="#6C63FF", width=44,
        )
        self._chunk_lbl.grid(row=0, column=1)

        ctk.CTkLabel(
            body,
            text="Shorter = faster response. Longer = better sentence context.",
            font=ctk.CTkFont("Segoe UI", 11),
            text_color="#8888AA", anchor="w",
        ).grid(row=9, column=0, sticky="w", pady=(2, 20))

        # ── Save Button ──────────────────────────────────
        self._status_lbl = ctk.CTkLabel(
            self._win, text="", font=ctk.CTkFont("Segoe UI", 11), text_color="#2ECC71"
        )
        self._status_lbl.pack(pady=(0, 4))

        ctk.CTkButton(
            self._win, text="Save Settings",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color="#6C63FF", hover_color="#4E46C1",
            height=44, corner_radius=10,
            command=self._save,
        ).pack(fill="x", padx=20, pady=(0, 16))

        self._win.protocol("WM_DELETE_WINDOW", self._win.destroy)

    def _section_label(self, parent, text: str, row: int):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color="#E8E8F0", anchor="w",
        ).grid(row=row, column=0, sticky="w", pady=(8, 2))

    def _on_chunk_change(self, val):
        self._chunk_lbl.configure(text=f"{val:.1f}s")

    def _save(self):
        new_hotkey  = self._hotkey_rec.get()
        old_hotkey  = self._cfg.get("hotkey")
        gpu_model   = self._gpu_model_var.get()
        cpu_model   = self._cpu_model_var.get()
        chunk_sec   = round(self._chunk_var.get(), 1)

        self._cfg.set("hotkey",           new_hotkey)
        self._cfg.set("model_size_gpu",   gpu_model)
        self._cfg.set("model_size_cpu",   cpu_model)
        self._cfg.set("audio_segment_sec", chunk_sec)

        # Notify the app to re-register the hotkey if it changed
        if new_hotkey != old_hotkey:
            self._on_hotkey_changed(new_hotkey)

        self._status_lbl.configure(text="Settings saved! Model changes apply on next launch.")
        self._win.after(2500, lambda: self._status_lbl.configure(text=""))


# ============================================================
#  STATUS WINDOW
# ============================================================

class StatusWindow:
    """Small popup showing the live transcript and current state."""

    def __init__(self, root: tk.Tk):
        self._root    = root
        self._win     = None
        self._textbox = None
        self._stat    = None

    def show(self):
        self._root.after(0, self._show_impl)

    def update(self, text: str = "", status: str = ""):
        self._root.after(0, self._update_impl, text, status)

    def _show_impl(self):
        if self._win is None or not self._alive():
            self._build()
        self._win.deiconify()
        self._win.lift()
        self._win.focus_force()

    def _alive(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def _build(self):
        self._win = ctk.CTkToplevel(self._root)
        self._win.title("Dictation Assistant — Live Status")
        self._win.configure(fg_color="#0F1117")
        self._win.geometry("540x360")

        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        self._win.geometry(f"540x360+{(sw-540)//2}+{(sh-360)//2}")

        hdr = ctk.CTkFrame(self._win, fg_color="#1A1D27", corner_radius=0, height=52)
        hdr.pack(fill="x")
        ctk.CTkLabel(
            hdr, text="  Live Transcript",
            font=ctk.CTkFont("Segoe UI", 17, "bold"), text_color="#E8E8F0",
        ).pack(side="left", padx=18, pady=14)

        self._textbox = ctk.CTkTextbox(
            self._win, font=ctk.CTkFont("Segoe UI", 15),
            fg_color="#12141C", text_color="#E8E8F0", wrap="word",
            corner_radius=8, border_width=0,
        )
        self._textbox.pack(fill="both", expand=True, padx=16, pady=(12, 4))
        self._textbox.insert("1.0", "Waiting for dictation...")
        self._textbox.configure(state="disabled")

        self._stat = ctk.CTkLabel(
            self._win, text="Ready",
            font=ctk.CTkFont("Segoe UI", 11), text_color="#8888AA",
        )
        self._stat.pack(anchor="w", padx=18, pady=(2, 12))

        self._win.protocol("WM_DELETE_WINDOW", self._win.withdraw)

    def _update_impl(self, text: str, status: str):
        if not self._alive():
            return
        if text and self._textbox:
            self._textbox.configure(state="normal")
            self._textbox.delete("1.0", "end")
            self._textbox.insert("1.0", text)
            self._textbox.configure(state="disabled")
        if status and self._stat:
            self._stat.configure(text=status)


# ============================================================
#  SYSTEM TRAY
# ============================================================

def _draw_icon(recording: bool = False) -> Image.Image:
    """Draw a 64x64 microphone icon. Red when recording, purple when idle."""
    size = 64
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d    = ImageDraw.Draw(img)
    bg   = (231, 76, 60, 255) if recording else (108, 99, 255, 255)
    d.ellipse([2, 2, size-2, size-2], fill=bg)
    d.rounded_rectangle([22, 8, 42, 36], radius=9, fill="white")
    d.arc([14, 22, 50, 50], start=0, end=180, fill="white", width=3)
    d.line([32, 50, 32, 57], fill="white", width=3)
    d.line([23, 57, 41, 57], fill="white", width=3)
    return img


class SystemTray:
    """pystray icon, tooltip, and right-click menu."""

    def __init__(self, app: "DictationAssistant"):
        self._app  = app
        self.icon: pystray.Icon | None = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="TrayThread").start()

    def _run(self):
        hotkey = self._app.cfg.get("hotkey").upper()
        menu = pystray.Menu(
            pystray.MenuItem("Dictation Assistant v2.0", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Hotkey: {hotkey}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Status Window", lambda i, m: self._app.show_status()),
            pystray.MenuItem("Settings",           lambda i, m: self._app.show_settings()),
            pystray.MenuItem("Copy Last Transcript", lambda i, m: self._copy_last()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit",               lambda i, m: self._app.quit()),
        )
        self.icon = pystray.Icon(
            "DictationAssistant", _draw_icon(False),
            f"Dictation Assistant — {hotkey}",
            menu,
        )
        self.icon.run()

    def set_listening(self, active: bool):
        if self.icon:
            self.icon.icon  = _draw_icon(recording=active)
            hotkey = self._app.cfg.get("hotkey").upper()
            self.icon.title = (
                "Recording... Press hotkey to stop" if active
                else f"Dictation Assistant — {hotkey}"
            )

    def stop(self):
        if self.icon:
            self.icon.stop()

    def _copy_last(self):
        if self._app.last_transcript:
            pyperclip.copy(self._app.last_transcript)


# ============================================================
#  TRANSCRIPTION WORKER
# ============================================================

class TranscriptionWorker:
    """
    Persistent background thread that transcribes audio chunks.
    Keeps the CUDA context warm between segments for zero re-init overhead.
    """

    def __init__(self, model: WhisperModel, on_result, on_error):
        self._model     = model
        self._on_result = on_result
        self._on_error  = on_error
        self._q         = queue.Queue()
        self._thread    = threading.Thread(
            target=self._loop, daemon=True, name="TranscribeWorker"
        )

    def start(self):
        self._thread.start()

    def submit(self, audio: np.ndarray):
        self._q.put(audio)

    def shutdown(self):
        self._q.put(None)

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                text = self._transcribe(item)
                self._on_result(text)
            except Exception as exc:
                print(f"[Transcriber] Error: {exc}")
                self._on_error(exc)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=300, threshold=0.5),
            condition_on_previous_text=False,  # No carry-over hallucinations
            temperature=0.0,                   # Greedy, deterministic
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        return " ".join(s.text.strip() for s in segments if s.text.strip())


# ============================================================
#  MAIN APPLICATION
# ============================================================

class DictationAssistant:
    """
    Orchestrates all components. State machine:

      IDLE ──[hotkey]──► LISTENING ──[auto chunks]──► inject text
           ◄──[hotkey]── LISTENING
    """

    def __init__(self):
        self.cfg             = ConfigManager()
        self.last_transcript = ""
        self._is_listening   = False
        self._lock           = threading.Lock()
        self._session_q      = queue.Queue(maxsize=20)  # audio chunk queue
        self._recorder       = ContinuousRecorder(
            self._session_q,
            segment_sec=self.cfg.get("audio_segment_sec"),
        )
        self._injector = TextInjector(restore_delay=self.cfg.get("clipboard_delay"))
        self._model    = None
        self._worker: TranscriptionWorker | None = None

        # ── Tkinter root (hidden) ──────────────────────
        ctk.set_appearance_mode("dark")
        self._root = ctk.CTk()
        self._root.withdraw()
        self._root.title("DictationAssistant")
        self._root.wm_attributes("-alpha", 0)   # Invisible

        # ── UI components ─────────────────────────────
        self._overlay  = FloatingOverlay(self._root)
        self._status   = StatusWindow(self._root)
        self._settings = SettingsWindow(self._root, self.cfg, self._on_hotkey_changed)
        self._tray     = SystemTray(self)

        # ── Chunk dispatcher thread ────────────────────
        # Reads from _session_q and routes chunks to the transcriber
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="ChunkDispatcher"
        )

    # ----------------------------------------------------------------
    #  Initialization
    # ----------------------------------------------------------------

    def initialize(self):
        """Load model and start services. Call before run()."""
        device, compute_type, model_size = detect_device(self.cfg)
        self._model = load_model(device, compute_type, model_size)

        self._worker = TranscriptionWorker(
            model=self._model,
            on_result=self._on_transcript,
            on_error=self._on_error,
        )
        self._worker.start()
        self._dispatcher.start()
        self._tray.start()

        # Register the global hotkey from config
        hotkey = self.cfg.get("hotkey")
        keyboard.add_hotkey(hotkey, self._on_hotkey, suppress=False)
        print(f"[App] Ready. Hotkey: {hotkey.upper()}")

    # ----------------------------------------------------------------
    #  Chunk Dispatcher Loop
    # ----------------------------------------------------------------

    def _dispatch_loop(self):
        """
        Reads audio chunks from the session queue and routes them
        to the TranscriptionWorker.

        Also handles the "done" sentinel from ContinuousRecorder.
        """
        while True:
            try:
                item = self._session_q.get(timeout=0.5)
            except queue.Empty:
                continue

            kind, data = item

            if kind == "chunk" and self._is_listening:
                # Submit this audio chunk for transcription
                self._worker.submit(data)

            elif kind == "done":
                # Recording has fully stopped — nothing special needed
                pass

    # ----------------------------------------------------------------
    #  Hotkey & State Machine
    # ----------------------------------------------------------------

    def _on_hotkey(self):
        """Toggle between IDLE and LISTENING states."""
        with self._lock:
            if self._is_listening:
                self._stop_listening()
            else:
                self._start_listening()

    def _on_hotkey_changed(self, new_hotkey: str):
        """Called from SettingsWindow when user saves a new hotkey."""
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        keyboard.add_hotkey(new_hotkey, self._on_hotkey, suppress=False)
        print(f"[App] Hotkey changed to: {new_hotkey.upper()}")

    # ----------------------------------------------------------------
    #  Listening Control
    # ----------------------------------------------------------------

    def _start_listening(self):
        """Enter LISTENING state — begin continuous auto-transcription."""
        self._is_listening = True
        _beep(*BEEP_START)

        self._overlay.show("  [REC]  Listening...", "#2ECC71")
        self._tray.set_listening(True)
        self._status.update(self.last_transcript, "[REC] Listening...")

        # Update recorder with current chunk size from config
        self._recorder = ContinuousRecorder(
            self._session_q,
            segment_sec=self.cfg.get("audio_segment_sec"),
        )
        self._recorder.start()
        print("[App] Listening started.")

    def _stop_listening(self):
        """Leave LISTENING state — flush audio and stop recorder."""
        self._is_listening = False
        _beep(*BEEP_STOP)

        self._overlay.update_text("  [...]  Processing...", "#F39C12")
        self._tray.set_listening(False)

        self._recorder.stop()   # Flushes remaining audio into the queue
        self._status.update(self.last_transcript, "Stopped.")
        print("[App] Listening stopped.")

    # ----------------------------------------------------------------
    #  Transcription Callbacks
    # ----------------------------------------------------------------

    def _on_transcript(self, text: str):
        """
        Called by TranscriptionWorker (on worker thread) when a chunk
        has been transcribed.

        Injects text immediately if we're still (or were just) listening.
        """
        if text:
            self.last_transcript += (" " if self.last_transcript else "") + text
            print(f"[Transcript] '{text}'")

            self._injector.inject(text)
            _beep(*BEEP_INJECT)

            self._status.update(self.last_transcript, f"Injected: {text[:50]}")

        # Hide overlay only if we've fully stopped
        if not self._is_listening:
            self._overlay.hide()

    def _on_error(self, exc: Exception):
        print(f"[App] Transcription error: {exc}")
        if not self._is_listening:
            self._overlay.hide()
        self._status.update(self.last_transcript, f"Error: {exc}")

    # ----------------------------------------------------------------
    #  Public Interface (called from tray/settings)
    # ----------------------------------------------------------------

    def show_status(self):
        self._status.show()

    def show_settings(self):
        self._settings.show()

    def quit(self):
        """Gracefully shut everything down."""
        print("[App] Shutting down...")
        if self._is_listening:
            self._recorder.stop()
            self._is_listening = False
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        if self._worker:
            self._worker.shutdown()
        self._tray.stop()
        self._root.after(0, self._root.destroy)

    # ----------------------------------------------------------------
    #  Entry Point
    # ----------------------------------------------------------------

    def run(self):
        """Start the Tkinter event loop (blocks main thread)."""
        try:
            self._root.mainloop()
        except KeyboardInterrupt:
            self.quit()


# ============================================================
#  UTILITY
# ============================================================

def _beep(frequency: int, duration_ms: int):
    """Non-blocking Windows beep."""
    threading.Thread(
        target=lambda: winsound.Beep(frequency, duration_ms),
        daemon=True,
    ).start()


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print()
    print("=" * 56)
    print("  Global Background Dictation Assistant  v2.0")
    print("=" * 56)
    print("  Press hotkey ONCE  -> Start continuous listening")
    print("  Speak naturally    -> Text auto-injected in chunks")
    print("  Press hotkey AGAIN -> Stop")
    print("  Tray icon          -> Right-click for menu")
    print("=" * 56)
    print()

    app = DictationAssistant()
    app.initialize()

    hotkey = app.cfg.get("hotkey").upper()
    print(f"  Hotkey  : {hotkey}")
    print(f"  Config  : {CONFIG_FILE}")
    print(f"  Tray    : Check bottom-right system tray")
    print()

    app.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
