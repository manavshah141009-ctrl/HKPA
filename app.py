"""
============================================================
  Personal Dictation Assistant  v2.0
  ============================================================
  A real-time, word-for-word voice transcription desktop app.

  NEW in v2.0:
    - ⚙️ Settings panel (hotkey, model, chunk size)
    - Hotkey is now configurable and saved to config.json
    - Voice commands: mouse move, click, scroll, shortcuts

  Tech Stack:
    - faster-whisper  : Optimized Whisper model (small.en, int8)
    - sounddevice     : Microphone audio capture
    - customtkinter   : Modern GUI framework
    - keyboard        : Global hotkey support
    - pyautogui       : Voice-controlled mouse/keyboard

  Architecture:
    - Main Thread      → GUI (CustomTkinter event loop)
    - AudioThread      → Continuous microphone recording
    - TranscribeThread → faster-whisper inference
============================================================
"""

import tkinter as tk
import customtkinter as ctk
import threading
import queue
import time
import json
import os
import re
import ctypes
from ctypes import wintypes
import sys
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import sounddevice as sd
import pyperclip
import pyautogui
import keyboard
import torch
import sys
import winsound
from faster_whisper import WhisperModel

# ============================================================
#  AUDIO FEEDBACK (BEEPS)
# ============================================================
BEEP_START   = (1000, 80)
BEEP_STOP    = (700, 80)
BEEP_INJECT  = (1200, 60)

def _beep(freq, duration):
    try:
        threading.Thread(target=lambda: winsound.Beep(freq, duration), daemon=True).start()
    except Exception:
        pass


# ============================================================
#  PYAUTOGUI SETTINGS
# ============================================================
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.01


# ============================================================
#  CONSTANTS & DEFAULTS
# ============================================================
SAMPLE_RATE           = 16000
CHANNELS              = 1
BLOCK_SIZE            = 4000
VAD_FILTER            = True
DEFAULT_HOTKEY        = "ctrl+shift+space"
DEFAULT_SEGMENT_SEC   = 3.5
DEFAULT_MOUSE_STEP    = 120
DEFAULT_MODEL_GPU     = "small.en"
DEFAULT_MODEL_CPU     = "base.en"

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50, "hundred": 100,
}


def _parse_num(token, default):
    if token is None:
        return default
    t = token.lower().strip()
    if t in WORD_TO_NUM:
        return WORD_TO_NUM[t]
    try:
        return int(t)
    except ValueError:
        return default


# ============================================================
#  CONFIG MANAGER
# ============================================================
class ConfigManager:
    DEFAULTS = {
        "hotkey":           DEFAULT_HOTKEY,
        "model_size_gpu":   DEFAULT_MODEL_GPU,
        "model_size_cpu":   DEFAULT_MODEL_CPU,
        "audio_segment_sec": DEFAULT_SEGMENT_SEC,
        "mouse_step":       DEFAULT_MOUSE_STEP,
    }

    def __init__(self):
        self._path = CONFIG_FILE
        self._data = self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return {**self.DEFAULTS, **json.load(f)}
            except Exception:
                pass
        return dict(self.DEFAULTS)

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"[Config] Save error: {e}")

    def get(self, key):
        return self._data.get(key, self.DEFAULTS.get(key))

    def set(self, key, value):
        self._data[key] = value
        self.save()


# ============================================================
#  HARDWARE DETECTION
# ============================================================
def detect_device(cfg: ConfigManager):
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        model = cfg.get("model_size_gpu")
        print(f"[HW] GPU: {name} -> CUDA / {model}")
        return "cuda", "int8_float16", model
    model = cfg.get("model_size_cpu")
    print(f"[HW] CPU -> int8 / {model}")
    return "cpu", "int8", model


def load_model(device, compute_type, model_size):
    print(f"[Model] Loading {model_size} on {device}...")
    m = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=4)
    print("[Model] Ready.")
    return m


# ============================================================
#  COMMAND PARSER (Voice → Mouse / Keyboard)
# ============================================================
class CommandParser:
    def __init__(self, step=DEFAULT_MOUSE_STEP):
        self._step = step
        sw, sh = pyautogui.size()
        self._sw, self._sh = sw, sh
        self._cmds = self._build()

    def update_step(self, step):
        self._step = step
        self._cmds = self._build()

    def _build(self):
        step = self._step
        sw, sh = self._sw, self._sh
        return [
            (r"^(?:left )?click$",         lambda m: pyautogui.click()),
            (r"^right[\s\-]?click$",        lambda m: pyautogui.rightClick()),
            (r"^double[\s\-]?click$",       lambda m: pyautogui.doubleClick()),
            (r"^scroll up(?: (\w+))?$",     lambda m: pyautogui.scroll(_parse_num(m.group(1), 3))),
            (r"^scroll down(?: (\w+))?$",   lambda m: pyautogui.scroll(-_parse_num(m.group(1), 3))),
            (r"^move left(?: (\w+))?$",     lambda m: pyautogui.moveRel(-_parse_num(m.group(1), step), 0, duration=0.12)),
            (r"^move right(?: (\w+))?$",    lambda m: pyautogui.moveRel(_parse_num(m.group(1), step), 0, duration=0.12)),
            (r"^move up(?: (\w+))?$",       lambda m: pyautogui.moveRel(0, -_parse_num(m.group(1), step), duration=0.12)),
            (r"^move down(?: (\w+))?$",     lambda m: pyautogui.moveRel(0, _parse_num(m.group(1), step), duration=0.12)),
            (r"^(?:go to |mouse )?center$", lambda m: pyautogui.moveTo(sw // 2, sh // 2, duration=0.2)),
        ]

    def try_command(self, text: str) -> bool:
        norm = re.sub(r"[.!?,;:]+$", "", text.lower().strip())
        for pattern, handler in self._cmds:
            m = re.match(pattern, norm, re.IGNORECASE)
            if m:
                try:
                    handler(m)
                    print(f"[CMD] {norm}")
                    threading.Thread(
                        target=lambda: winsound.Beep(1000, 60), daemon=True
                    ).start()
                except Exception as e:
                    print(f"[CMD] Error: {e}")
                return True
        return False


# ============================================================
#  WIN32 SENDINPUT HELPER (bypasses keyboard hooks)
# ============================================================
_user32_app    = ctypes.WinDLL("user32", use_last_error=True)
_KEYEVENTF_KEYUP_APP = 0x0002
_VK_CONTROL_APP = 0x11
_VK_V_APP       = 0x56

class _KEYBDINPUT_APP(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT_UNION_APP(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT_APP)]

class _INPUT_APP(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("_", _INPUT_UNION_APP)]

_INPUT_KEYBOARD_APP = 1

def _send_ctrl_v_app():
    """Send Ctrl+V via SendInput — completely bypasses keyboard hooks."""
    inputs = (_INPUT_APP * 4)(
        _INPUT_APP(type=_INPUT_KEYBOARD_APP, _=_INPUT_UNION_APP(ki=_KEYBDINPUT_APP(wVk=_VK_CONTROL_APP, dwFlags=0))),
        _INPUT_APP(type=_INPUT_KEYBOARD_APP, _=_INPUT_UNION_APP(ki=_KEYBDINPUT_APP(wVk=_VK_V_APP, dwFlags=0))),
        _INPUT_APP(type=_INPUT_KEYBOARD_APP, _=_INPUT_UNION_APP(ki=_KEYBDINPUT_APP(wVk=_VK_V_APP, dwFlags=_KEYEVENTF_KEYUP_APP))),
        _INPUT_APP(type=_INPUT_KEYBOARD_APP, _=_INPUT_UNION_APP(ki=_KEYBDINPUT_APP(wVk=_VK_CONTROL_APP, dwFlags=_KEYEVENTF_KEYUP_APP))),
    )
    _user32_app.SendInput(4, inputs, ctypes.sizeof(_INPUT_APP))





# ============================================================
#  TEXT INJECTOR
# ============================================================
class TextInjector:
    """
    Pastes text into the active focused window via clipboard swap + Ctrl+V.
    Works strictly at the OS level to prevent focus stealing.
    """
    def __init__(self, restore_delay: float = 0.5):
        self._delay = restore_delay

    def inject(self, text: str, target_hwnd=None):
        if not text.strip():
            return
            
        try:
            original = pyperclip.paste()
        except Exception:
            original = ""

        try:
            import pyautogui
            import win32gui
            
            # Restore target window focus if we have a valid handle
            if target_hwnd:
                print(f"[Injector] Activating target hwnd: {target_hwnd}")
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shell.SendKeys('%') # Dummy ALT to allow focus stealing
                    win32gui.SetForegroundWindow(target_hwnd)
                    time.sleep(0.1) # Brief delay for OS focus shift
                except Exception as e:
                    print(f"[Injector] Focus restore error: {e}")

            pyperclip.copy(text)
            print("[Injector] Copied to clipboard. Sending Ctrl+V...")
            time.sleep(0.05)
            pyautogui.hotkey('ctrl', 'v')
            print("[Injector] Ctrl+V sent.")
            _beep(*BEEP_INJECT)
            time.sleep(0.05)
        except Exception as e:
            print(f"[Injector] Error: {e}")
        finally:
            def restore():
                try:
                    pyperclip.copy(original)
                except:
                    pass
            threading.Timer(self._delay, restore).start()


# ============================================================
#  AUDIO RECORDER THREAD
# ============================================================
class AudioRecorderThread(threading.Thread):
    def __init__(self, audio_queue: queue.Queue, status_callback, segment_sec=DEFAULT_SEGMENT_SEC):
        super().__init__(daemon=True)
        self.audio_queue     = audio_queue
        self.status_callback = status_callback
        self.segment_sec     = segment_sec
        self._stop_event     = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        _q = queue.Queue()

        def _cb(indata, frames, time_info, status):
            if status:
                print(f"[Audio] {status}")
            _q.put(indata.copy())

        self.status_callback("🎙️ Listening...", "green")
        frames_needed = int(SAMPLE_RATE * self.segment_sec)
        silence_frames_needed = int(SAMPLE_RATE * 0.6)  # 0.6s silence threshold
        buffer, frame_count, current_silence = [], 0, 0

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="float32", blocksize=BLOCK_SIZE, callback=_cb,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = _q.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                flat_chunk = chunk.flatten()
                buffer.append(flat_chunk)
                frame_count += chunk.shape[0]
                
                # Check volume (RMS) to detect silence
                rms = np.sqrt(np.mean(np.square(flat_chunk)))
                if rms < 0.01:  # Simple volume threshold
                    current_silence += chunk.shape[0]
                else:
                    current_silence = 0
                
                # Stop chunk if max segment reached OR silence duration exceeded (after initial 0.5s)
                if frame_count >= frames_needed or (frame_count > SAMPLE_RATE * 0.5 and current_silence > silence_frames_needed):
                    if frame_count > 0:
                        self.audio_queue.put(np.concatenate(buffer))
                    buffer, frame_count, current_silence = [], 0, 0

        self.status_callback("⏹️ Stopped", "gray")


class SemanticRouter:
    """
    Passes transcribed text to Groq LLM for grammar/spelling correction.
    """
    def __init__(self, on_dictation, on_command_executed):
        self._q = queue.Queue()
        self._on_dictation = on_dictation
        self._on_command_executed = on_command_executed
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[SemanticRouter] WARNING: GROQ_API_KEY not found in .env")
            
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or "missing_key"
        )
        self.model = "llama-3.1-8b-instant"
        
        self.system_prompt = (
            "You receive transcribed text of the user speaking. "
            "Carefully correct any spelling, grammar, or punctuation errors in the text, and return ONLY the corrected text. Make it sound natural and polished. Do not add any conversational filler, explanations, or quotes. "
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, text: str, trigger_source: str, target_hwnd=None):
        self._q.put((text, trigger_source, target_hwnd))
        
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            text, trigger_source, target_hwnd = item
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.0,
                    max_tokens=150
                )
                content = response.choices[0].message.content or text
                self._on_dictation(content, trigger_source, target_hwnd)
            except Exception as e:
                print(f"[SemanticRouter] LLM routing error: {e}")
                self._on_dictation(text, trigger_source, target_hwnd)


# ============================================================
#  TRANSCRIBER THREAD
# ============================================================
class TranscriberThread(threading.Thread):
    def __init__(self, model, audio_queue, text_callback, status_callback, cmd_parser, trigger_source: str, target_hwnd=None):
        super().__init__(daemon=True)
        self.model           = model
        self.audio_queue     = audio_queue
        self.text_callback   = text_callback
        self.status_callback = status_callback
        self.cmd_parser      = cmd_parser
        self.trigger_source  = trigger_source
        self.target_hwnd     = target_hwnd
        self.injector        = TextInjector()
        self._stop_event     = threading.Event()
        
        def _on_dict(t, source, hwnd):
            if source == "gui_button":
                self.text_callback(t)
            elif source == "global_hotkey":
                self.injector.inject(t + " ", hwnd)
            
        self.router = SemanticRouter(_on_dict, lambda t, s: None)

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self.status_callback("⚡ Transcribing...", "orange")
            try:
                segs, _ = self.model.transcribe(
                    audio, language="en", beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                )
                text = " ".join(s.text.strip() for s in segs if s.text.strip())
                if text:
                    if not self.cmd_parser.try_command(text):
                        self.router.submit(text, self.trigger_source, self.target_hwnd)
            except Exception as e:
                print(f"[Transcriber] Error: {e}")
            finally:
                self.status_callback("🎙️ Listening...", "green")


# ============================================================
#  SETTINGS PANEL (slides in from the right side of the window)
# ============================================================
class SettingsPanel:
    """
    A slide-in settings panel drawn on top of the main window.
    Opens by clicking the ⚙️ button in the header.
    Fully embedded inside the CTk window — no Toplevel needed.
    """

    C_BG      = "#0D0F18"
    C_PANEL   = "#161929"
    C_BORDER  = "#252840"
    C_ACCENT  = "#6C63FF"
    C_HOVER   = "#4E46C1"
    C_TEXT    = "#E8E8F0"
    C_MUTED   = "#8888AA"
    C_GREEN   = "#2ECC71"

    def __init__(self, parent_app: "DictationApp"):
        self._app      = parent_app
        self._cfg      = parent_app.cfg
        self._visible  = False
        self._recording_hotkey = False
        self._current_hotkey   = self._cfg.get("hotkey")

        # Use plain tk.Frame as outer container so place(width=, height=) works.
        # CustomTkinter's CTkFrame raises ValueError if width/height are passed to place().
        self._frame = tk.Frame(
            parent_app,
            bg=self.C_BG,
            highlightthickness=1,
            highlightbackground=self.C_BORDER,
        )
        self._build()

    # ----------------------------------------------------------
    #  Build all widgets inside the panel
    # ----------------------------------------------------------
    def _build(self):
        self._frame.grid_columnconfigure(0, weight=1)
        row = 0

        # ── Header row ──────────────────────────────────────────
        hdr = ctk.CTkFrame(self._frame, fg_color=self.C_PANEL, corner_radius=0, height=52)
        hdr.grid(row=row, column=0, sticky="ew"); row += 1
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr, text="  ⚙  Settings",
            font=ctk.CTkFont("Segoe UI", 16, "bold"),
            text_color=self.C_TEXT,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=14)

        ctk.CTkButton(
            hdr, text="✕", width=36, height=36,
            fg_color="transparent", hover_color="#2A2D3E",
            font=ctk.CTkFont("Segoe UI", 14),
            text_color=self.C_MUTED,
            command=self.hide,
        ).grid(row=0, column=1, padx=8, pady=8)

        # ── Scrollable body ─────────────────────────────────────
        body = ctk.CTkScrollableFrame(
            self._frame, fg_color=self.C_BG, corner_radius=0,
        )
        body.grid(row=row, column=0, sticky="nsew", padx=0, pady=0)
        self._frame.grid_rowconfigure(row, weight=1); row += 1
        body.grid_columnconfigure(0, weight=1)
        br = 0   # body row counter

        def section(text):
            nonlocal br
            ctk.CTkLabel(
                body, text=text,
                font=ctk.CTkFont("Segoe UI", 12, "bold"),
                text_color=self.C_MUTED, anchor="w",
            ).grid(row=br, column=0, sticky="w", padx=18, pady=(18, 4)); br += 1
            # thin divider
            ctk.CTkFrame(body, fg_color=self.C_BORDER, height=1, corner_radius=0
                         ).grid(row=br, column=0, sticky="ew", padx=18, pady=(0, 10)); br += 1

        def label(text):
            nonlocal br
            ctk.CTkLabel(
                body, text=text,
                font=ctk.CTkFont("Segoe UI", 11),
                text_color=self.C_MUTED, anchor="w",
            ).grid(row=br, column=0, sticky="w", padx=18, pady=(0, 8)); br += 1

        # ── HOTKEY ──────────────────────────────────────────────
        section("GLOBAL HOTKEY")

        # Hotkey display + record button in one row
        hk_row = ctk.CTkFrame(body, fg_color="transparent")
        hk_row.grid(row=br, column=0, sticky="ew", padx=18, pady=(0, 4)); br += 1
        hk_row.grid_columnconfigure(0, weight=1)

        self._hk_display = ctk.CTkLabel(
            hk_row,
            text=self._fmt_hotkey(self._current_hotkey),
            font=ctk.CTkFont("Consolas", 14, "bold"),
            text_color=self.C_ACCENT,
            fg_color=self.C_PANEL,
            corner_radius=8,
            padx=14, pady=10,
            anchor="center",
        )
        self._hk_display.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self._hk_btn = ctk.CTkButton(
            hk_row, text="Change", width=90, height=38,
            fg_color="#252840", hover_color="#353860",
            font=ctk.CTkFont("Segoe UI", 12),
            command=self._start_hotkey_capture,
        )
        self._hk_btn.grid(row=0, column=1)

        label("Click [Change] then press any key combination.")

        # ── MODEL ────────────────────────────────────────────────
        section("WHISPER MODEL")

        ctk.CTkLabel(body, text="GPU Model:", font=ctk.CTkFont("Segoe UI", 12),
                     text_color=self.C_TEXT, anchor="w"
                     ).grid(row=br, column=0, sticky="w", padx=18, pady=(0, 4)); br += 1
        self._gpu_var = ctk.StringVar(value=self._cfg.get("model_size_gpu"))
        ctk.CTkOptionMenu(
            body, values=["tiny.en", "base.en", "small.en", "medium.en"],
            variable=self._gpu_var,
            fg_color=self.C_PANEL, button_color=self.C_ACCENT,
            button_hover_color=self.C_HOVER,
        ).grid(row=br, column=0, sticky="ew", padx=18, pady=(0, 12)); br += 1

        ctk.CTkLabel(body, text="CPU Model:", font=ctk.CTkFont("Segoe UI", 12),
                     text_color=self.C_TEXT, anchor="w"
                     ).grid(row=br, column=0, sticky="w", padx=18, pady=(0, 4)); br += 1
        self._cpu_var = ctk.StringVar(value=self._cfg.get("model_size_cpu"))
        ctk.CTkOptionMenu(
            body, values=["tiny.en", "base.en", "small.en"],
            variable=self._cpu_var,
            fg_color=self.C_PANEL, button_color=self.C_ACCENT,
            button_hover_color=self.C_HOVER,
        ).grid(row=br, column=0, sticky="ew", padx=18, pady=(0, 4)); br += 1
        label("Model changes apply on next launch.")

        # ── CHUNK LENGTH ─────────────────────────────────────────
        section("CHUNK LENGTH (seconds)")

        self._chunk_var = tk.DoubleVar(value=self._cfg.get("audio_segment_sec"))
        chunk_row = ctk.CTkFrame(body, fg_color="transparent")
        chunk_row.grid(row=br, column=0, sticky="ew", padx=18); br += 1
        chunk_row.grid_columnconfigure(0, weight=1)

        ctk.CTkSlider(
            chunk_row, from_=1.5, to=6.0, number_of_steps=9,
            variable=self._chunk_var,
            progress_color=self.C_ACCENT, button_color=self.C_ACCENT,
            command=lambda v: self._chunk_lbl.configure(text=f"{v:.1f}s"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self._chunk_lbl = ctk.CTkLabel(
            chunk_row, text=f"{self._chunk_var.get():.1f}s",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=self.C_ACCENT, width=46,
        )
        self._chunk_lbl.grid(row=0, column=1)
        label("Shorter = faster. Longer = better sentence context.")

        # ── MOUSE STEP ───────────────────────────────────────────
        section("MOUSE STEP (pixels per voice command)")

        self._step_var = tk.IntVar(value=self._cfg.get("mouse_step"))
        step_row = ctk.CTkFrame(body, fg_color="transparent")
        step_row.grid(row=br, column=0, sticky="ew", padx=18); br += 1
        step_row.grid_columnconfigure(0, weight=1)

        ctk.CTkSlider(
            step_row, from_=40, to=400, number_of_steps=18,
            variable=self._step_var,
            progress_color=self.C_ACCENT, button_color=self.C_ACCENT,
            command=lambda v: self._step_lbl.configure(text=f"{int(v)}px"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self._step_lbl = ctk.CTkLabel(
            step_row, text=f"{self._step_var.get()}px",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=self.C_ACCENT, width=52,
        )
        self._step_lbl.grid(row=0, column=1)
        label("Say 'move left/right/up/down [N]' to move the mouse.")

        # ── COMMAND REFERENCE ────────────────────────────────────
        section("VOICE COMMANDS REFERENCE")

        cmds = [
            ("click",              "Left mouse click"),
            ("right click",        "Right mouse click"),
            ("double click",       "Double click"),
            ("scroll up/down [N]", "Scroll the page"),
            ("move left/right [N]","Move mouse (default 120px)"),
            ("press enter/escape", "Press keyboard keys"),
            ("select all",         "Ctrl+A"),
            ("copy / paste",       "Ctrl+C / Ctrl+V"),
            ("undo / redo",        "Ctrl+Z / Ctrl+Y"),
            ("save",               "Ctrl+S"),
            ("new tab",            "Ctrl+T"),
            ("go back",            "Alt+Left"),
            ("screenshot",         "Win+Shift+S"),
            ("zoom in / zoom out", "Ctrl+= / Ctrl+-"),
        ]
        for cmd, desc in cmds:
            row_f = ctk.CTkFrame(body, fg_color="transparent")
            row_f.grid(row=br, column=0, sticky="ew", padx=18, pady=1); br += 1
            row_f.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                row_f, text=cmd,
                font=ctk.CTkFont("Consolas", 11),
                text_color=self.C_ACCENT, anchor="w", width=170,
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                row_f, text=desc,
                font=ctk.CTkFont("Segoe UI", 11),
                text_color=self.C_MUTED, anchor="w",
            ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        # ── SAVE BUTTON ──────────────────────────────────────────
        self._save_msg = ctk.CTkLabel(
            self._frame, text="",
            font=ctk.CTkFont("Segoe UI", 11), text_color=self.C_GREEN,
        )
        self._save_msg.grid(row=row, column=0, pady=(4, 0)); row += 1

        ctk.CTkButton(
            self._frame, text="💾  Save Settings",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color=self.C_ACCENT, hover_color=self.C_HOVER,
            height=46, corner_radius=0,
            command=self._save,
        ).grid(row=row, column=0, sticky="ew", padx=0, pady=0); row += 1

    # ----------------------------------------------------------
    #  Hotkey Capture
    # ----------------------------------------------------------
    @staticmethod
    def _fmt_hotkey(hk: str) -> str:
        return "  " + "  +  ".join(k.upper() for k in hk.split("+")) + "  "

    def _start_hotkey_capture(self):
        self._recording_hotkey = True
        self._hk_display.configure(
            text="  Press your hotkey...  ", text_color="#F39C12"
        )
        self._hk_btn.configure(text="Cancel", command=self._cancel_hotkey_capture)
        # Bind to the top-level window
        self._app.bind("<KeyPress>", self._on_keypress)

    def _on_keypress(self, event):
        if not self._recording_hotkey:
            return
        mods = {"control_l", "control_r", "shift_l", "shift_r",
                "alt_l", "alt_r", "meta_l", "meta_r", "super_l", "super_r", "caps_lock"}
        key = event.keysym.lower()
        if key in mods:
            return
        parts = []
        if event.state & 0x4:     parts.append("ctrl")
        if event.state & 0x20000: parts.append("alt")
        if event.state & 0x1:     parts.append("shift")
        key_map = {"space": "space", "return": "enter", "escape": "esc",
                   "tab": "tab", "backspace": "backspace"}
        parts.append(key_map.get(key, key))
        self._current_hotkey = "+".join(parts)
        self._hk_display.configure(
            text=self._fmt_hotkey(self._current_hotkey),
            text_color=self.C_ACCENT,
        )
        self._finish_hotkey_capture()

    def _cancel_hotkey_capture(self):
        self._hk_display.configure(
            text=self._fmt_hotkey(self._current_hotkey),
            text_color=self.C_ACCENT,
        )
        self._finish_hotkey_capture()

    def _finish_hotkey_capture(self):
        self._recording_hotkey = False
        self._hk_btn.configure(text="Change", command=self._start_hotkey_capture)
        try:
            self._app.unbind("<KeyPress>")
        except Exception:
            pass

    # ----------------------------------------------------------
    #  Save
    # ----------------------------------------------------------
    def _save(self):
        old_hk   = self._cfg.get("hotkey")
        new_hk   = self._current_hotkey
        new_step = int(self._step_var.get())

        self._cfg.set("hotkey",            new_hk)
        self._cfg.set("model_size_gpu",    self._gpu_var.get())
        self._cfg.set("model_size_cpu",    self._cpu_var.get())
        self._cfg.set("audio_segment_sec", round(self._chunk_var.get(), 1))
        self._cfg.set("mouse_step",        new_step)

        # Re-register the hotkey if it changed
        if new_hk != old_hk:
            self._app.update_hotkey(new_hk)

        # Update the mouse step in the command parser
        self._app.cmd_parser.update_step(new_step)

        self._save_msg.configure(text="✅ Settings saved!")
        self._frame.after(2500, lambda: self._save_msg.configure(text=""))

    # ----------------------------------------------------------
    #  Show / Hide (slide in/out using place())
    # ----------------------------------------------------------
    def toggle(self):
        if self._visible:
            self.hide()
        else:
            self.show()

    def show(self):
        """Place the panel on the right side of the main window."""
        if self._visible:
            return
        self._visible = True
        pw = self._app.winfo_width()
        ph = self._app.winfo_height()
        panel_w = min(360, pw - 20)
        self._frame.place(x=pw - panel_w, y=0, width=panel_w, height=ph)
        self._frame.lift()

    def hide(self):
        """Remove the panel."""
        if not self._visible:
            return
        self._visible = False
        self._frame.place_forget()


# ============================================================
#  MAIN APPLICATION GUI
# ============================================================
class DictationApp(ctk.CTk):

    C_BG          = "#0F1117"
    C_PANEL       = "#1A1D27"
    C_ACCENT      = "#6C63FF"
    C_ACCENT_DARK = "#4E46C1"
    C_TEXT        = "#E8E8F0"
    C_SUBTEXT     = "#8888AA"
    C_SUCCESS     = "#2ECC71"
    C_WARN        = "#F39C12"

    def __init__(self, model, device: str, cfg: ConfigManager):
        super().__init__()
        self.model        = model
        self.device       = device
        self.cfg          = cfg
        self.cmd_parser   = CommandParser(step=cfg.get("mouse_step"))
        self._hotkey      = cfg.get("hotkey")

        self.audio_thread       = None
        self.transcriber_thread = None
        self.audio_queue        = queue.Queue(maxsize=10)
        self.is_listening       = False

        self._configure_window()
        self._build_ui()

        # Settings panel (overlays the window)
        self.settings_panel = SettingsPanel(self)

        # Thread-safe queue for hotkey triggers
        self._hotkey_queue = queue.Queue()
        self._poll_hotkey_queue()

        # Start the polling hotkey listener (immune to hook interference)
        self._hk_thread = threading.Thread(target=self._hotkey_listener_loop, daemon=True)
        self._hk_thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------
    def _configure_window(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("🎙️ Personal Dictation Assistant")
        self.geometry("900x680")
        self.attributes("-topmost", True)
        self.lift()
        self.minsize(700, 500)
        self.configure(fg_color=self.C_BG)
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"900x680+{(sw-900)//2}+{(sh-680)//2}")

    # ----------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_transcript_area()
        self._build_controls()
        self._build_status_bar()

    # ----------------------------------------------------------
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=self.C_PANEL, corner_radius=0, height=60)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="  🎙️  Personal Dictation Assistant",
            font=ctk.CTkFont("Segoe UI", 20, "bold"),
            text_color=self.C_TEXT,
        ).grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # Hardware / model info — updates after hotkey changes
        device_text  = "⚡ GPU (CUDA)" if self.device == "cuda" else "🖥️ CPU"
        device_color = self.C_SUCCESS if self.device == "cuda" else self.C_SUBTEXT
        self._header_info = ctk.CTkLabel(
            header,
            text=self._header_text(),
            font=ctk.CTkFont("Segoe UI", 11),
            text_color=device_color,
        )
        self._header_info.grid(row=0, column=1, padx=(0, 12), pady=15, sticky="e")

        # ⚙️ Settings button
        self._settings_btn = ctk.CTkButton(
            header,
            text="⚙  Settings",
            font=ctk.CTkFont("Segoe UI", 12),
            fg_color="#252840",
            hover_color="#353860",
            width=110,
            height=36,
            corner_radius=8,
            command=self._toggle_settings,
        )
        self._settings_btn.grid(row=0, column=2, padx=(0, 16), pady=12)

    def _header_text(self):
        model = self.cfg.get("model_size_gpu" if self.device == "cuda" else "model_size_cpu")
        hk    = self.cfg.get("hotkey").upper()
        device_text = "GPU (CUDA)" if self.device == "cuda" else "CPU"
        return f"⚡ {device_text}  •  Model: {model}  •  Hotkey: {hk}"

    # ----------------------------------------------------------
    def _build_transcript_area(self):
        text_frame = ctk.CTkFrame(self, fg_color=self.C_PANEL, corner_radius=12)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(15, 10))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            text_frame, text="Transcript",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=self.C_SUBTEXT,
        ).grid(row=0, column=0, padx=15, pady=(12, 4), sticky="w")

        self.transcript_box = ctk.CTkTextbox(
            text_frame,
            font=ctk.CTkFont("Segoe UI", 16),
            fg_color="#12141C",
            text_color=self.C_TEXT,
            border_width=0,
            corner_radius=8,
            wrap="word",
            scrollbar_button_color=self.C_ACCENT,
        )
        self.transcript_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self._set_placeholder()

    def _set_placeholder(self):
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        hk = self.cfg.get("hotkey").upper().replace("+", " + ")
        self.transcript_box.insert(
            "1.0",
            f"Press  ▶ Start Listening  or  {hk}  to begin dictating...\n\n"
            "Your words will appear here as a live transcript.\n\n"
            "💡 TIP: This window shows your transcript only.\n"
            "   To type by voice into ANY other app (WhatsApp, Chrome, etc.):\n"
            "   1. Run assistant.py separately (it lives in the system tray).\n"
            "   2. Click inside the chat box in the other app.\n"
            "   3. Press the hotkey — speak — your words are auto-typed.\n\n"
            "🖥️ Voice commands: 'scroll down', 'click', 'move right'",
        )
        self.transcript_box.configure(state="disabled", text_color=self.C_SUBTEXT)
        self._placeholder_active = True

    def _clear_placeholder(self):
        if getattr(self, "_placeholder_active", False):
            self.transcript_box.configure(state="normal", text_color=self.C_TEXT)
            self.transcript_box.delete("1.0", "end")
            self._placeholder_active = False

    # ----------------------------------------------------------
    def _build_controls(self):
        controls = ctk.CTkFrame(self, fg_color=self.C_PANEL, corner_radius=12, height=70)
        controls.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))
        controls.grid_columnconfigure((0, 1, 2), weight=1)

        self.toggle_btn = ctk.CTkButton(
            controls,
            text="▶  Start Listening",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color=self.C_ACCENT, hover_color=self.C_ACCENT_DARK,
            corner_radius=10, height=44,
            command=self._toggle_listening,
        )
        self.toggle_btn.grid(row=0, column=0, padx=15, pady=13, sticky="ew")

        ctk.CTkButton(
            controls, text="🗑  Clear Text",
            font=ctk.CTkFont("Segoe UI", 14),
            fg_color="#2A2D3E", hover_color="#353850",
            corner_radius=10, height=44,
            command=self._clear_text,
        ).grid(row=0, column=1, padx=15, pady=13, sticky="ew")

        ctk.CTkButton(
            controls, text="📋  Copy to Clipboard",
            font=ctk.CTkFont("Segoe UI", 14),
            fg_color="#2A2D3E", hover_color="#353850",
            corner_radius=10, height=44,
            command=self._copy_to_clipboard,
        ).grid(row=0, column=2, padx=15, pady=13, sticky="ew")

    # ----------------------------------------------------------
    def _build_status_bar(self):
        status_frame = ctk.CTkFrame(
            self, fg_color=self.C_PANEL, corner_radius=0, height=36
        )
        status_frame.grid(row=3, column=0, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)

        self.pulse_canvas = tk.Canvas(
            status_frame, width=16, height=16,
            bg=self.C_PANEL, highlightthickness=0,
        )
        self.pulse_canvas.grid(row=0, column=0, padx=(16, 4), pady=10)
        self._pulse_dot = self.pulse_canvas.create_oval(
            2, 2, 14, 14, fill="gray", outline=""
        )

        self.status_label = ctk.CTkLabel(
            status_frame,
            text=f"⏹️ Ready  —  Press Start or {self.cfg.get('hotkey').upper()}",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=self.C_SUBTEXT,
        )
        self.status_label.grid(row=0, column=1, padx=4, pady=10, sticky="w")
        self._pulse_running = False

    # ----------------------------------------------------------
    #  Pulse animation
    # ----------------------------------------------------------
    def _start_pulse(self):
        self._pulse_running = True
        self._pulse_state   = False
        self._animate_pulse()

    def _stop_pulse(self):
        self._pulse_running = False
        self.pulse_canvas.itemconfig(self._pulse_dot, fill="gray")

    def _animate_pulse(self):
        if not self._pulse_running:
            return
        color = "#2ECC71" if self._pulse_state else "#1A6E3C"
        self.pulse_canvas.itemconfig(self._pulse_dot, fill=color)
        self._pulse_state = not self._pulse_state
        self.after(600, self._animate_pulse)

    # ----------------------------------------------------------
    #  Thread-safe callbacks
    # ----------------------------------------------------------
    def _append_text_safe(self, text: str):
        self._clear_placeholder()
        self.transcript_box.configure(state="normal")
        current = self.transcript_box.get("1.0", "end-1c")
        if current.strip():
            self.transcript_box.insert("end", " " + text)
        else:
            self.transcript_box.insert("end", text)
        self.transcript_box.see("end")

    def _update_status_safe(self, message: str, color: str):
        self.status_label.configure(text=message)
        color_map = {"green": self.C_SUCCESS, "orange": self.C_WARN, "gray": self.C_SUBTEXT}
        self.pulse_canvas.itemconfig(self._pulse_dot, fill=color_map.get(color, self.C_SUBTEXT))

    def _on_text_received(self, text: str):
        self.after(0, self._append_text_safe, text)

    def _on_status_changed(self, msg: str, color: str):
        self.after(0, self._update_status_safe, msg, color)

    # ----------------------------------------------------------
    #  Listening toggle
    # ----------------------------------------------------------
    def _toggle_listening(self, source="gui_button", target_hwnd=None):
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening(source, target_hwnd)

    def _hotkey_listener_loop(self):
        # Polls GetAsyncKeyState via keyboard.is_pressed
        held = False
        import win32gui
        while True:
            try:
                hk_parts = self._hotkey.split('+')
                if all(keyboard.is_pressed(p) for p in hk_parts):
                    if not held:
                        held = True
                        print("[App] Hotkey physical press detected (Poller)!")
                        
                        # Capture the exact active window right now
                        target_hwnd = None
                        try:
                            hwnd = win32gui.GetForegroundWindow()
                            title = win32gui.GetWindowText(hwnd)
                            if "Personal Dictation Assistant" not in title:
                                target_hwnd = hwnd
                        except:
                            pass
                            
                        self._hotkey_queue.put(("global_hotkey", target_hwnd))
                else:
                    held = False
            except Exception:
                pass
            time.sleep(0.05)

    def _poll_hotkey_queue(self):
        # Runs in main GUI thread
        try:
            while True:
                item = self._hotkey_queue.get_nowait()
                if isinstance(item, tuple):
                    source, target_hwnd = item
                else:
                    source, target_hwnd = item, None
                self._toggle_listening(source, target_hwnd)
        except queue.Empty:
            pass
        self.after(100, self._poll_hotkey_queue)

    def _start_listening(self, source="gui_button", target_hwnd=None):
        self.is_listening = True
        _beep(*BEEP_START)
        self.toggle_btn.configure(
            text="⏹  Stop Listening",
            fg_color="#C0392B", hover_color="#922B21",
        )
        self._clear_placeholder()
        self._start_pulse()
        self.audio_queue = queue.Queue(maxsize=10)

        self.audio_thread = AudioRecorderThread(
            audio_queue=self.audio_queue,
            status_callback=self._on_status_changed,
            segment_sec=self.cfg.get("audio_segment_sec"),
        )
        self.audio_thread.start()

        self.transcriber_thread = TranscriberThread(
            model=self.model,
            audio_queue=self.audio_queue,
            text_callback=self._on_text_received,
            status_callback=self._on_status_changed,
            cmd_parser=self.cmd_parser,
            trigger_source=source,
            target_hwnd=target_hwnd,
        )
        self.transcriber_thread.start()

    def _stop_listening(self):
        self.is_listening = False
        _beep(*BEEP_STOP)
        self.toggle_btn.configure(
            text="▶  Start Listening",
            fg_color=self.C_ACCENT, hover_color=self.C_ACCENT_DARK,
        )
        self._stop_pulse()
        hk = self.cfg.get("hotkey").upper()
        self._update_status_safe(f"⏹️ Stopped  —  Press Start or {hk}", "gray")
        if self.audio_thread:
            self.audio_thread.stop()
        if self.transcriber_thread:
            self.transcriber_thread.stop()

    # ----------------------------------------------------------
    #  Settings panel toggle
    # ----------------------------------------------------------
    def _toggle_settings(self):
        self.settings_panel.toggle()
        # Update button text to reflect state
        if self.settings_panel._visible:
            self._settings_btn.configure(text="✕  Close")
        else:
            self._settings_btn.configure(text="⚙  Settings")

    # ----------------------------------------------------------
    #  Hotkey update (called from SettingsPanel._save)
    # ----------------------------------------------------------
    def update_hotkey(self, new_hotkey: str):
        self._hotkey = new_hotkey
        # Update header display
        self._header_info.configure(text=self._header_text())
        # Update status bar
        hk = new_hotkey.upper()
        if not self.is_listening:
            self.status_label.configure(
                text=f"⏹️ Ready  —  Press Start or {hk}"
            )
        print(f"[App] Hotkey updated -> {hk}")

    # ----------------------------------------------------------
    #  Other actions
    # ----------------------------------------------------------
    def _clear_text(self):
        self._set_placeholder()

    def _copy_to_clipboard(self):
        text = self.transcript_box.get("1.0", "end-1c").strip()
        if text and not getattr(self, "_placeholder_active", False):
            pyperclip.copy(text)
            self._update_status_safe("✅ Copied to clipboard!", "green")
            self.after(2000, lambda: self._update_status_safe(
                f"⏹️ Ready  —  Press Start or {self.cfg.get('hotkey').upper()}", "gray"
            ))

    def _on_close(self):
        print("[App] Closing...")
        if self.audio_thread:
            self.audio_thread.stop()
        if self.transcriber_thread:
            self.transcriber_thread.stop()
        self.destroy()


# ============================================================
#  SPLASH SCREEN
# ============================================================
class SplashScreen(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("")
        self.geometry("420x210")
        self.resizable(False, False)
        self.configure(fg_color="#1A1D27")
        self.overrideredirect(True)
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 420) // 2
        y = (self.winfo_screenheight() - 210) // 2
        self.geometry(f"420x210+{x}+{y}")

        ctk.CTkLabel(
            self,
            text="🎙️  Personal Dictation Assistant",
            font=ctk.CTkFont("Segoe UI", 18, "bold"),
            text_color="#E8E8F0",
        ).pack(pady=(36, 8))

        self.status_lbl = ctk.CTkLabel(
            self, text="Initializing...",
            font=ctk.CTkFont("Segoe UI", 13), text_color="#8888AA",
        )
        self.status_lbl.pack(pady=4)

        self.progress = ctk.CTkProgressBar(
            self, mode="indeterminate",
            progress_color="#6C63FF", fg_color="#2A2D3E",
        )
        self.progress.pack(pady=18, padx=40, fill="x")
        self.progress.start()

    def update_status(self, text: str):
        self.status_lbl.configure(text=text)
        self.update()

    def safe_destroy(self):
        try:
            self.progress.stop()
        except Exception:
            pass
        self.destroy()


# ============================================================
#  ENTRY POINT
# ============================================================
def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    ctk.set_appearance_mode("dark")
    cfg  = ConfigManager()
    root = ctk.CTk()
    root.withdraw()

    splash = SplashScreen(root)
    splash.update_status("Detecting hardware...")
    splash.update()

    device, compute_type, model_size = detect_device(cfg)
    splash.update_status(f"Loading Whisper ({model_size}) on {'GPU' if device=='cuda' else 'CPU'}...")
    splash.update()

    try:
        model = load_model(device, compute_type, model_size)
    except Exception as e:
        splash.destroy()
        root.destroy()
        print(f"FATAL: Could not load Whisper model: {e}")
        sys.exit(1)

    splash.update_status("Launching...")
    splash.update()
    time.sleep(0.3)
    splash.safe_destroy()
    root.destroy()

    app = DictationApp(model=model, device=device, cfg=cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
