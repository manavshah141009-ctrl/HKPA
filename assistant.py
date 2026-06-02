"""
============================================================
  Global Background Dictation Assistant  v3.0
============================================================
  Press hotkey ONCE  → continuous listening + auto-inject
  Press hotkey AGAIN → stop
  Voice commands     → control mouse & keyboard hands-free

  NEW in v3.0:
    - Voice mouse control (move, click, scroll, shortcuts)
    - Bug fix: no more re-transcription on stop
    - Configurable mouse step size in Settings

  Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Main Thread       → Hidden CTk root + Tkinter loop     │
  │  TrayThread        → pystray icon                       │
  │  ContinuousRecorder→ mic capture + auto-chunking        │
  │  ChunkDispatcher   → routes audio chunks to worker      │
  │  TranscribeWorker  → faster-whisper inference queue     │
  │  CommandParser     → intercepts voice commands          │
  │  TextInjector      → Ctrl+V paste into active window    │
  └─────────────────────────────────────────────────────────┘

  Voice Command Flow:
    Speech → Whisper → CommandParser.try_command()
      → YES: execute mouse/keyboard action  (not injected as text)
      → NO:  TextInjector.inject()          (pasted normally)
============================================================
"""

# ── Standard Library ───────────────────────────────────────
import threading
import queue
import time
import sys
import os
import json
import re
import winsound
import subprocess
import base64
import io
import ctypes
from ctypes import wintypes

# ── Audio & ML ────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# ── Automation ────────────────────────────────────────────
import keyboard
import pyperclip
import pyautogui
import pywinauto
from pywinauto import Desktop
import webbrowser
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── System Tray ───────────────────────────────────────────
import pystray
from PIL import Image, ImageDraw

# ── GUI ───────────────────────────────────────────────────
import tkinter as tk
import customtkinter as ctk


# ============================================================
#  PYAUTOGUI GLOBAL SETTINGS
# ============================================================
pyautogui.FAILSAFE  = True    # Move mouse to top-left (0,0) to abort any macro
pyautogui.PAUSE     = 0.04    # Small delay between pyautogui calls (prevents race)


# ============================================================
#  FALLBACK DEFAULTS  (overridden by config.json)
# ============================================================
DEFAULT_HOTKEY              = "ctrl+alt+v"
DEFAULT_MODEL_SIZE_GPU      = "small.en"
DEFAULT_MODEL_SIZE_CPU      = "base.en"
DEFAULT_AUDIO_SEGMENT_SEC   = 3.5
DEFAULT_CLIPBOARD_DELAY     = 0.8
DEFAULT_MOUSE_STEP          = 120   # pixels per "move left/right/up/down" command

SAMPLE_RATE  = 16000
CHANNELS     = 1
BLOCK_SIZE   = 4000
VAD_FILTER   = True

BEEP_START  = (880,  130)
BEEP_STOP   = (440,  130)
BEEP_CMD    = (1000,  60)   # Short tick = command executed
BEEP_INJECT = (1200,  70)   # Higher ping = text injected

OVERLAY_RIGHT_MARGIN = 250
OVERLAY_TOP_MARGIN   = 20

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


# ============================================================
#  WORD → NUMBER LOOKUP  (for commands like "scroll down three")
# ============================================================
WORD_TO_NUM: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "two hundred": 200, "three hundred": 300, "five hundred": 500,
}


def _parse_num(token: str | None, default: int) -> int:
    """Convert a word or digit string to an integer."""
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
        "hotkey"             : DEFAULT_HOTKEY,
        "model_size_gpu"     : DEFAULT_MODEL_SIZE_GPU,
        "model_size_cpu"     : DEFAULT_MODEL_SIZE_CPU,
        "audio_segment_sec"  : DEFAULT_AUDIO_SEGMENT_SEC,
        "clipboard_delay"    : DEFAULT_CLIPBOARD_DELAY,
        "mouse_step"         : DEFAULT_MOUSE_STEP,
    }

    def __init__(self, path: str = CONFIG_FILE):
        self._path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return {**self.DEFAULTS, **json.load(f)}
            except Exception as e:
                print(f"[Config] Could not read config.json: {e}. Using defaults.")
        return dict(self.DEFAULTS)

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"[Config] Could not save: {e}")

    def get(self, key: str):
        return self._data.get(key, self.DEFAULTS.get(key))

    def set(self, key: str, value):
        self._data[key] = value
        self.save()


# ============================================================
#  HARDWARE DETECTION & MODEL LOADER
# ============================================================

def detect_device(cfg: ConfigManager) -> tuple[str, str, str]:
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        model = cfg.get("model_size_gpu")
        print(f"[HW] GPU: {name} -> CUDA int8_float16 / {model}")
        return "cuda", "int8_float16", model
    model = cfg.get("model_size_cpu")
    print(f"[HW] No GPU -> CPU int8 / {model}")
    return "cpu", "int8", model


def load_model(device: str, compute_type: str, model_size: str) -> WhisperModel:
    print(f"[Model] Loading {model_size} on {device} ({compute_type})...")
    m = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=4)
    print("[Model] Model ready.")
    return m


# ============================================================
#  OS AUTOMATION ENGINE (pywinauto + pyautogui)
# ============================================================



class ExecutorWorker:
    """Dedicated background thread to process GUI actions safely."""
    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ExecutorWorker")
        self._thread.start()

    def submit(self, func_name: str, args: dict):
        self._q.put((func_name, args))

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            func_name, args = item
            ToolExecutor.execute(func_name, args)



class ToolExecutor:
    """Comprehensive OS automation suites requested by the LLM."""

    # ── Mouse & Trackpad Emulation Suite ────────────────────────
    @staticmethod
    def execute_click(click_type: str, double_click: bool):
        print(f"[ToolExecutor] Mouse click: {click_type} (double={double_click})")
        click_type = click_type.lower()
        clicks = 2 if double_click else 1
        if click_type == "left":
            pyautogui.click(clicks=clicks)
        elif click_type == "right":
            pyautogui.rightClick()
        elif click_type == "middle":
            pyautogui.middleClick()

    @staticmethod
    def execute_scroll(direction: str, clicks: int):
        print(f"[ToolExecutor] Scroll: {direction} ({clicks} clicks)")
        direction = direction.lower()
        amount = clicks * 120
        if direction == "up":
            pyautogui.scroll(amount)
        elif direction == "down":
            pyautogui.scroll(-amount)
        elif direction == "left":
            pyautogui.hscroll(-amount)
        elif direction == "right":
            pyautogui.hscroll(amount)

    @staticmethod
    def execute_mouse_move(direction: str, distance: int):
        print(f"[ToolExecutor] Mouse move: {direction} ({distance} px)")
        direction = direction.lower()
        if direction == "up":
            pyautogui.moveRel(0, -distance)
        elif direction == "down":
            pyautogui.moveRel(0, distance)
        elif direction == "left":
            pyautogui.moveRel(-distance, 0)
        elif direction == "right":
            pyautogui.moveRel(distance, 0)
        elif direction == "center":
            width, height = pyautogui.size()
            pyautogui.moveTo(width / 2, height / 2)

    # ── Master Executor ─────────────────────────────────────────
    @classmethod
    def execute(cls, function_name: str, arguments: dict):
        func = getattr(cls, function_name, None)
        if func:
            try:
                func(**arguments)
            except Exception as e:
                print(f"[ToolExecutor] Error executing {function_name}: {e}")
        else:
            print(f"[ToolExecutor] Unknown tool: {function_name}")


# ============================================================
#  SEMANTIC ROUTER (LLM Routing)
# ============================================================

class SemanticRouter:
    """
    Background thread that receives transcribed text, passes it to the LLM,
    and pushes tools to the ExecutorWorker.
    """
    def __init__(self, on_dictation, on_command_executed):
        self._q = queue.Queue()
        self._on_dictation = on_dictation
        self._on_command_executed = on_command_executed
        self._worker = ExecutorWorker()
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[SemanticRouter] WARNING: GROQ_API_KEY not found in .env")
            
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key or "missing_key"
        )
        self.model = "llama-3.1-8b-instant"
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_click",
                    "description": "Perform mouse clicks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "click_type": {"type": "string", "enum": ["left", "right", "middle"]},
                            "double_click": {"type": "boolean"}
                        },
                        "required": ["click_type", "double_click"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_scroll",
                    "description": "Scroll the screen in any direction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                            "clicks": {"type": "integer", "description": "Number of scroll clicks. Default to 5 if not specified."}
                        },
                        "required": ["direction", "clicks"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_mouse_move",
                    "description": "Moves the mouse cursor in a specific direction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string", "enum": ["up", "down", "left", "right", "center"]},
                            "distance": {"type": "integer", "description": "Pixels to move (e.g. 100, 200). Use 0 if direction is center."}
                        },
                        "required": ["direction", "distance"]
                    }
                }
            }
        ]

        self.system_prompt = (
            "You are a silent semantic routing engine for a dictation and computer control application. "
            "You receive transcribed text of the user speaking. "
            "If the text is standard speech meant to be typed out (dictation), DO NOT call any tools. Instead, carefully correct any spelling, grammar, or punctuation errors in the text, and return ONLY the corrected text. Make it sound natural and polished. Do not add any conversational filler, explanations, or quotes. "
            "If the text is a command intended to control the computer, you MUST call the appropriate tool. "
            "NEVER output conversational text or reasoning. Output ONLY the raw tool call JSON.\n\n"
            "Structural Few-Shot Examples:\n"
            "USER: 'Scroll down a little bit'\n"
            "TOOL: execute_scroll(direction=\"down\", clicks=5)\n\n"
            "USER: 'Move mouse up 100 pixels'\n"
            "TOOL: execute_mouse_move(direction=\"up\", distance=100)\n\n"
            "USER: 'Double click that'\n"
            "TOOL: execute_click(click_type=\"left\", double_click=True)"
        )

        self._thread = threading.Thread(target=self._run, daemon=True, name="SemanticRouter")
        self._thread.start()

    def submit(self, text: str):
        self._q.put(text)
        
    def _run(self):
        while True:
            text = self._q.get()
            if text is None:
                break
                
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.0
                )
                
                message = response.choices[0].message
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except:
                            args = {}
                        print(f"[SemanticRouter] Routed to tool: {func_name} with {args}")
                        self._worker.submit(func_name, args)
                    self._on_command_executed(text, f"[CMD] {func_name}")
                else:
                    content = message.content or text
                    self._on_dictation(content)
                    
            except Exception as e:
                print(f"[SemanticRouter] LLM routing error: {e}")
                self._on_dictation(text)


# ============================================================
#  CONTINUOUS RECORDER
# ============================================================

class ContinuousRecorder:
    """
    Records mic audio indefinitely, auto-slicing into fixed-length chunks.
    Each chunk is placed into `chunk_queue` as ("chunk", np.ndarray).
    A ("done", None) sentinel is sent when recording fully stops.

    BUG FIX v3.0:
      Remaining audio on stop() is DISCARDED (not flushed).
      Flushing partial audio caused the same words to be re-transcribed
      because the partial buffer overlapped with the tail of the previous
      full chunk that was already sent to the transcription worker.
    """

    def __init__(self, chunk_queue: queue.Queue, segment_sec: float):
        self._queue       = chunk_queue
        self._segment_sec = segment_sec
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ContinuousRecorder"
        )
        self._thread.start()

    def stop(self):
        """Stop recording. Partial remaining audio is discarded (not flushed)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run(self):
        internal_q    = queue.Queue()
        frames_needed = int(SAMPLE_RATE * self._segment_sec)
        buffer: list[np.ndarray] = []
        frame_count   = 0

        def _cb(indata: np.ndarray, frames: int, _time, status):
            if status:
                print(f"[Audio] {status}")
            internal_q.put(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="float32", blocksize=BLOCK_SIZE, callback=_cb,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = internal_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                buffer.append(chunk.flatten())
                frame_count += chunk.shape[0]

                if frame_count >= frames_needed:
                    self._queue.put(("chunk", np.concatenate(buffer)))
                    buffer      = []   # Reset — prevents re-transcription
                    frame_count = 0

        # ── INTENTIONALLY no flush here ──────────────────────────────
        # Flushing `buffer` here caused the re-transcription bug.
        # The partial remaining audio is simply discarded.
        self._queue.put(("done", None))


# ============================================================
#  TEXT INJECTOR
# ============================================================

# ── Win32 constants for SendInput ────────────────────────────
_user32    = ctypes.WinDLL("user32", use_last_error=True)
_KEYEVENTF_KEYUP   = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_VK_CONTROL = 0x11
_VK_V       = 0x56
_WM_PASTE   = 0x0302

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT)]

class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("_", _INPUT_UNION)]

_INPUT_KEYBOARD = 1

def _send_ctrl_v():
    """Send Ctrl+V via SendInput — completely bypasses keyboard hooks."""
    inputs = (_INPUT * 4)(
        # Ctrl down
        _INPUT(type=_INPUT_KEYBOARD, _=_INPUT_UNION(ki=_KEYBDINPUT(
            wVk=_VK_CONTROL, dwFlags=0))),
        # V down
        _INPUT(type=_INPUT_KEYBOARD, _=_INPUT_UNION(ki=_KEYBDINPUT(
            wVk=_VK_V, dwFlags=0))),
        # V up
        _INPUT(type=_INPUT_KEYBOARD, _=_INPUT_UNION(ki=_KEYBDINPUT(
            wVk=_VK_V, dwFlags=_KEYEVENTF_KEYUP))),
        # Ctrl up
        _INPUT(type=_INPUT_KEYBOARD, _=_INPUT_UNION(ki=_KEYBDINPUT(
            wVk=_VK_CONTROL, dwFlags=_KEYEVENTF_KEYUP))),
    )
    _user32.SendInput(4, inputs, ctypes.sizeof(_INPUT))


class TextInjector:
    """
    Pastes text into the active focused window via clipboard + SendInput.
    Uses Win32 SendInput directly — bypasses the `keyboard` library hook
    so the paste never triggers the hotkey listener or causes beeps.
    Works in any app: Chrome, WhatsApp, Notion, VS Code, Notepad, Explorer.
    """

    def __init__(self, restore_delay: float = 0.8):
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
            time.sleep(0.12)   # give clipboard time to settle
            _send_ctrl_v()     # SendInput — invisible to keyboard hook
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
#  FLOATING OVERLAY
# ============================================================

class FloatingOverlay:
    """Frameless always-on-top status chip in the top-right corner."""

    def __init__(self, root: tk.Tk):
        self._root  = root
        self._win   = None
        self._label = None

    def show(self, msg: str, color: str = "#E74C3C"):
        self._root.after(0, self._show_impl, msg, color)

    def hide(self):
        self._root.after(0, self._hide_impl)

    def update_text(self, msg: str, color: str):
        self._root.after(0, self._upd_impl, msg, color)

    def _show_impl(self, msg, color):
        if not self._alive():
            self._create()
        self._label.configure(fg=color, text=msg)
        self._win.deiconify()
        self._win.lift()
        self._win.attributes("-topmost", True)

    def _hide_impl(self):
        if self._alive():
            self._win.withdraw()

    def _upd_impl(self, msg, color):
        if self._alive():
            self._label.configure(fg=color, text=msg)

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
        self._win.geometry(f"240x46+{sw - OVERLAY_RIGHT_MARGIN}+{OVERLAY_TOP_MARGIN}")
        self._label = tk.Label(
            self._win, text="", font=("Segoe UI", 13, "bold"),
            fg="#E74C3C", bg="#0F1117", padx=14, pady=11,
        )
        self._label.pack(fill="both", expand=True)
        self._win.withdraw()


# ============================================================
#  HOTKEY RECORDER WIDGET
# ============================================================

class HotkeyRecorder(tk.Frame):
    """Records a key combination when the user clicks [Change]."""

    def __init__(self, parent, initial: str = "ctrl+alt+v", **kw):
        super().__init__(parent, bg="#1A1D27", **kw)
        self._hotkey    = initial
        self._recording = False
        self._bids: list[str] = []

        self.grid_columnconfigure(0, weight=1)
        self._disp = tk.Label(
            self, text=self._fmt(initial),
            font=("Consolas", 15, "bold"),
            fg="#6C63FF", bg="#1A1D27",
            padx=12, pady=8,
        )
        self._disp.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._btn = tk.Button(
            self, text="Change", width=12, height=2,
            bg="#2A2D3E", fg="white",
            command=self._start,
        )
        self._btn.grid(row=0, column=1)

    @staticmethod
    def _fmt(hk: str) -> str:
        return "  " + "  +  ".join(k.upper() for k in hk.split("+")) + "  "

    def _start(self):
        self._recording = True
        self._disp.configure(text="  Press your hotkey...  ", fg="#F39C12")
        self._btn.configure(text="Cancel", command=self._cancel)
        root = self.winfo_toplevel()
        root.focus_force()
        self._bids.append(root.bind("<KeyPress>", self._on_key, add=True))

    def _on_key(self, event):
        if not self._recording:
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
                   "tab": "tab", "backspace": "backspace", "delete": "delete"}
        parts.append(key_map.get(key, key))
        self._hotkey = "+".join(parts)
        self._disp.configure(text=self._fmt(self._hotkey), fg="#6C63FF")
        self._finish()

    def _cancel(self):
        self._disp.configure(text=self._fmt(self._hotkey), fg="#6C63FF")
        self._finish()

    def _finish(self):
        self._recording = False
        self._btn.configure(text="Change", command=self._start)
        root = self.winfo_toplevel()
        for bid in self._bids:
            try:
                root.unbind("<KeyPress>", bid)
            except Exception:
                pass
        self._bids.clear()

    def get(self) -> str:
        return self._hotkey

    def set(self, hk: str):
        self._hotkey = hk
        self._disp.configure(text=self._fmt(hk))


# ============================================================
#  SETTINGS WINDOW
# ============================================================

class SettingsWindow:
    """Settings panel: hotkey, models, chunk length, mouse step."""

    def __init__(self, root, cfg: ConfigManager, on_hotkey_changed, on_step_changed):
        self._root             = root
        self._cfg              = cfg
        self._on_hk_changed    = on_hotkey_changed
        self._on_step_changed  = on_step_changed
        self._win              = None

    def show(self):
        self._root.after(0, self._show_impl)

    def _show_impl(self):
        if self._win and self._alive():
            self._win.deiconify()
            self._win.after(50, self._win.lift)
            self._win.after(60, self._win.focus_force)
            return
        self._build()

    def _alive(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def _build(self):
        self._win = tk.Toplevel(self._root)
        self._win.title("Dictation Assistant — Settings")
        self._win.configure(bg="#0F1117")
        sw = self._win.winfo_screenwidth(); sh = self._win.winfo_screenheight()
        self._win.geometry(f"540x600+{(sw-540)//2}+{(sh-600)//2}")
        self._win.resizable(False, False)

        # ── Scrollable Body ──────────────────────────────────────────
        canvas = tk.Canvas(self._win, bg="#0F1117", highlightthickness=0)
        vsb = tk.Scrollbar(self._win, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg="#0F1117")
        
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw", width=520)
        canvas.configure(yscrollcommand=vsb.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Helpers
        def sec(title, r):
            tk.Label(body, text=title, font=("Segoe UI", 12, "bold"),
                         fg="#8888AA", bg="#0F1117", anchor="w"
                         ).grid(row=r, column=0, sticky="ew", pady=(24, 8), padx=10)
        def hint(text, r):
            tk.Label(body, text=text, font=("Segoe UI", 11),
                         fg="#666688", bg="#0F1117", anchor="w"
                         ).grid(row=r, column=0, sticky="ew", pady=(4, 0), padx=10)

        row = 0
        # ── Global Hotkey ────────────────────────────────────
        sec("Global Hotkey (Click 'Change' then press keys)", row); row += 1
        self._hk_rec = HotkeyRecorder(body, initial=self._cfg.get("hotkey"))
        self._hk_rec.grid(row=row, column=0, sticky="ew", padx=10); row += 1

        # ── Whisper Model ────────────────────────────────────
        sec("Whisper Model Size", row); row += 1
        
        mf = tk.Frame(body, bg="#0F1117"); mf.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        mf.grid_columnconfigure(1, weight=1)
        
        tk.Label(mf, text="GPU:", font=("Segoe UI", 12), fg="white", bg="#0F1117").grid(row=0, column=0, sticky="w", pady=4)
        self._gpu_var = tk.StringVar(value=self._cfg.get("model_size_gpu"))
        tk.OptionMenu(mf, self._gpu_var, "tiny.en", "base.en", "small.en", "medium.en").grid(row=0, column=1, sticky="ew", padx=(10,0))
        
        tk.Label(mf, text="CPU:", font=("Segoe UI", 12), fg="white", bg="#0F1117").grid(row=1, column=0, sticky="w", pady=4)
        self._cpu_var = tk.StringVar(value=self._cfg.get("model_size_cpu"))
        tk.OptionMenu(mf, self._cpu_var, "tiny.en", "base.en", "small.en").grid(row=1, column=1, sticky="ew", padx=(10,0))

        # ── Chunk Length ─────────────────────────────────────
        sec("Auto-Transcribe Chunk Length", row); row += 1
        self._chunk_var = tk.DoubleVar(value=self._cfg.get("audio_segment_sec"))
        cf = tk.Frame(body, bg="#0F1117"); cf.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        cf.grid_columnconfigure(0, weight=1)
        tk.Scale(cf, from_=1.5, to=6.0, resolution=0.5, orient="horizontal", variable=self._chunk_var, bg="#0F1117", fg="white", highlightthickness=0).grid(row=0, column=0, sticky="ew")
        hint("Shorter = faster response. Longer = better sentence context.", row); row += 1

        # ── Mouse Step ───────────────────────────────────────
        sec("Mouse Step Size (pixels per 'move' command)", row); row += 1
        self._step_var = tk.IntVar(value=self._cfg.get("mouse_step"))
        sf = tk.Frame(body, bg="#0F1117"); sf.grid(row=row, column=0, sticky="ew", padx=10); row += 1
        sf.grid_columnconfigure(0, weight=1)
        tk.Scale(sf, from_=40, to=400, resolution=20, orient="horizontal", variable=self._step_var, bg="#0F1117", fg="white", highlightthickness=0).grid(row=0, column=0, sticky="ew")
        hint("How far the cursor moves per 'move left/right/up/down' command.", row); row += 1

        # ── Save ─────────────────────────────────────────────
        self._stat = tk.Label(self._win, text="", font=("Segoe UI", 11), fg="#2ECC71", bg="#0F1117")
        self._stat.pack(pady=(4, 4))
        tk.Button(self._win, text="Save Settings", font=("Segoe UI", 14, "bold"),
                  bg="#6C63FF", fg="white", height=2, command=self._save
                 ).pack(fill="x", padx=20, pady=(0, 16))

        self._win.protocol("WM_DELETE_WINDOW", self._win.destroy)

        # ── Force the window to appear ────────────────────────────────
        self._win.deiconify()
        self._win.update_idletasks()
        self._win.after(80, self._win.lift)
        self._win.after(100, self._win.focus_force)
        self._win.after(120, lambda: self._win.attributes("-topmost", True))

    def _save(self):
        new_hk   = self._hk_rec.get()
        old_hk   = self._cfg.get("hotkey")
        new_step = int(self._step_var.get())

        self._cfg.set("hotkey",            new_hk)
        self._cfg.set("model_size_gpu",    self._gpu_var.get())
        self._cfg.set("model_size_cpu",    self._cpu_var.get())
        self._cfg.set("audio_segment_sec", round(self._chunk_var.get(), 1))
        self._cfg.set("mouse_step",        new_step)

        if new_hk != old_hk:
            self._on_hk_changed(new_hk)
        self._on_step_changed(new_step)

        self._stat.configure(text="Saved! Model changes apply on next launch.")
        self._win.after(2500, lambda: self._stat.configure(text=""))


# ============================================================
#  STATUS WINDOW
# ============================================================

class StatusWindow:
    def __init__(self, root: tk.Tk):
        self._root = root
        self._win  = None
        self._tb   = None
        self._sl   = None

    def show(self):
        self._root.after(0, self._show_impl)

    def update(self, text: str = "", status: str = ""):
        self._root.after(0, self._upd_impl, text, status)

    def _show_impl(self):
        if not self._alive():
            self._build()
        else:
            self._win.deiconify()
            self._win.after(50, self._win.lift)
            self._win.after(60, self._win.focus_force)

    def _alive(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def _build(self):
        self._win = tk.Toplevel(self._root)
        self._win.title("Dictation Assistant — Status")
        self._win.configure(bg="#0F1117")
        sw = self._win.winfo_screenwidth(); sh = self._win.winfo_screenheight()
        self._win.geometry(f"560x380+{(sw-560)//2}+{(sh-380)//2}")

        hdr = tk.Frame(self._win, bg="#1A1D27", height=52)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Live Transcript",
                     font=("Segoe UI", 17, "bold"),
                     fg="#E8E8F0", bg="#1A1D27").pack(side="left", padx=18, pady=14)

        self._tb = tk.Text(self._win,
                                  font=("Segoe UI", 15),
                                  bg="#12141C", fg="#E8E8F0",
                                  wrap="word")
        self._tb.pack(fill="both", expand=True, padx=16, pady=(12, 4))
        self._tb.insert("1.0", "Waiting for dictation...")
        self._tb.configure(state="disabled")

        self._sl = tk.Label(self._win, text="Ready",
                                font=("Segoe UI", 11), fg="#8888AA", bg="#0F1117")
        self._sl.pack(anchor="w", padx=18, pady=(2, 12))
        self._win.protocol("WM_DELETE_WINDOW", self._win.withdraw)

        self._win.deiconify()
        self._win.update_idletasks()
        self._win.after(80, self._win.lift)
        self._win.after(100, self._win.focus_force)

    def _upd_impl(self, text, status):
        if not self._alive():
            return
        if text and self._tb:
            self._tb.configure(state="normal")
            self._tb.delete("1.0", "end")
            self._tb.insert("1.0", text)
            self._tb.configure(state="disabled")
        if status and self._sl:
            self._sl.configure(text=status)


# ============================================================
#  SYSTEM TRAY
# ============================================================

def _draw_icon(recording: bool = False) -> Image.Image:
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
    def __init__(self, app: "DictationAssistant"):
        self._app  = app
        self.icon: pystray.Icon | None = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="TrayThread").start()

    def _run(self):
        hk = self._app.cfg.get("hotkey").upper()
        menu = pystray.Menu(
            pystray.MenuItem("Dictation Assistant v3.0", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Hotkey: {hk}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Show Status Window",    lambda i, m: self._app.show_status()),
            pystray.MenuItem("Settings",              lambda i, m: self._app.show_settings()),
            pystray.MenuItem("Copy Last Transcript",  lambda i, m: self._copy_last()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit",                  lambda i, m: self._app.quit()),
        )
        self.icon = pystray.Icon(
            "DictationAssistant", _draw_icon(False),
            f"Dictation Assistant — {hk}", menu,
        )
        self.icon.run()

    def set_listening(self, active: bool):
        if not self.icon:
            return
        self.icon.icon  = _draw_icon(active)
        hk = self._app.cfg.get("hotkey").upper()
        self.icon.title = ("Recording... Press hotkey to stop" if active
                           else f"Dictation Assistant — {hk}")

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
    """Dedicated thread that runs faster-whisper inference."""

    def __init__(self, model: WhisperModel, on_result, on_error):
        self._model  = model
        self._on_res = on_result
        self._on_err = on_error
        self._q      = queue.Queue()
        self._thread = threading.Thread(
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
                self._on_res(self._transcribe(item))
            except Exception as exc:
                print(f"[Transcriber] Error: {exc}")
                self._on_err(exc)

    def _transcribe(self, audio: np.ndarray) -> str:
        segs, _ = self._model.transcribe(
            audio, language="en", beam_size=5,
            vad_filter=VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=300, threshold=0.5),
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        return " ".join(s.text.strip() for s in segs if s.text.strip())


# ============================================================
#  MAIN APPLICATION
# ============================================================

class DictationAssistant:
    """
    Orchestrates all components.

    State machine:
      IDLE  ──[hotkey]──► LISTENING ──[auto chunks]──► parse command OR inject text
            ◄──[hotkey]── LISTENING
    """

    def __init__(self):
        self.cfg             = ConfigManager()
        self.last_transcript = ""
        self._is_listening   = False
        self._lock           = threading.Lock()

        # Session queue shared between ContinuousRecorder and ChunkDispatcher
        self._session_q = queue.Queue(maxsize=30)

        self._injector  = TextInjector(restore_delay=self.cfg.get("clipboard_delay"))
        self._router    = SemanticRouter(
            on_dictation=self._handle_dictation,
            on_command_executed=self._handle_command
        )
        self._model     = None
        self._worker: TranscriptionWorker | None = None
        self._recorder: ContinuousRecorder | None = None

        # ── Hidden CTk root ───────────────────────────────
        ctk.set_appearance_mode("dark")
        self._root = ctk.CTk()
        self._root.withdraw()
        self._root.title("DictationAssistant")
        self._root.wm_attributes("-alpha", 0)

        # ── UI components ─────────────────────────────────
        self._overlay  = FloatingOverlay(self._root)
        self._status   = StatusWindow(self._root)
        self._settings = SettingsWindow(
            self._root, self.cfg,
            on_hotkey_changed=self._on_hotkey_changed,
            on_step_changed=self._on_step_changed,
        )
        self._tray = SystemTray(self)

        # ── Chunk dispatcher thread ───────────────────────
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="ChunkDispatcher"
        )

    # ----------------------------------------------------------------
    #  Initialization
    # ----------------------------------------------------------------

    def initialize(self):
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

        hotkey = self.cfg.get("hotkey")
        keyboard.add_hotkey(hotkey, self._on_hotkey, suppress=True)
        print(f"[App] Ready. Hotkey: {hotkey.upper()}")

    # ----------------------------------------------------------------
    #  Chunk Dispatcher
    # ----------------------------------------------------------------

    def _dispatch_loop(self):
        """
        Reads (kind, data) from _session_q.
        Only submits to the transcription worker if we're still listening.
        """
        while True:
            try:
                item = self._session_q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind, data = item
            if kind == "chunk" and self._is_listening:
                self._worker.submit(data)
            # "done" sentinel — recording stopped, nothing to do

    def _drain_session_queue(self):
        """
        Remove all pending items from the session queue.
        Called at the START of each new listening session to prevent
        stale audio from a previous session from being transcribed.
        """
        drained = 0
        while True:
            try:
                self._session_q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f"[App] Drained {drained} stale chunk(s) from queue.")

    # ----------------------------------------------------------------
    #  Hotkey Toggle
    # ----------------------------------------------------------------

    def _on_hotkey(self):
        with self._lock:
            if self._is_listening:
                self._stop_listening()
            else:
                self._start_listening()

    def _on_hotkey_changed(self, new_hk: str):
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        keyboard.add_hotkey(new_hk, self._on_hotkey, suppress=True)
        print(f"[App] Hotkey -> {new_hk.upper()}")

    def _on_step_changed(self, step: int):
        print(f"[App] Mouse step -> {step}px")

    # ----------------------------------------------------------------
    #  Listening Control
    # ----------------------------------------------------------------

    def _start_listening(self):
        """
        Enter LISTENING state.
        Drains any stale queue items before starting the new recorder.
        """
        _beep(*BEEP_START)
        self._overlay.show("  [REC]  Listening...", "#2ECC71")
        self._tray.set_listening(True)
        self._status.update(self.last_transcript, "[REC] Listening...")

        # ── BUG FIX: drain stale audio from previous session ──────
        self._drain_session_queue()

        self._recorder = ContinuousRecorder(
            self._session_q,
            segment_sec=self.cfg.get("audio_segment_sec"),
        )
        self._is_listening = True   # Set AFTER drain to prevent race with dispatcher
        self._recorder.start()
        print("[App] Listening started.")

    def _stop_listening(self):
        """
        Leave LISTENING state.
        Sets flag first so dispatcher ignores any in-flight chunks.
        """
        self._is_listening = False  # Stop dispatcher from submitting new chunks
        _beep(*BEEP_STOP)
        self._overlay.update_text("  [...]  Finishing...", "#F39C12")
        self._tray.set_listening(False)

        if self._recorder:
            self._recorder.stop()  # Discards partial audio (no re-transcription)

        self._status.update(self.last_transcript, "Stopped.")
        print("[App] Listening stopped.")

        # Auto-hide overlay after remaining worker results are processed (2s buffer)
        self._root.after(2500, self._hide_overlay_if_idle)

    def _hide_overlay_if_idle(self):
        if not self._is_listening:
            self._overlay.hide()

    # ----------------------------------------------------------------
    #  Transcription Callbacks
    # ----------------------------------------------------------------

    def _on_transcript(self, text: str):
        """
        Called by TranscriptionWorker on the worker thread.
        Passes text to the SemanticRouter.
        """
        if not text:
            if not self._is_listening:
                self._overlay.hide()
            return

        print(f"[Transcript] '{text}'")
        self._router.submit(text)

    def _handle_command(self, raw_text: str, status: str):
        self._status.update(self.last_transcript, status)
        _beep(*BEEP_CMD)
        
    def _handle_dictation(self, text: str):
        self.last_transcript += (" " if self.last_transcript else "") + text
        self._injector.inject(text)
        _beep(*BEEP_INJECT)
        self._status.update(self.last_transcript, f"Injected: {text[:55]}")

    def _on_error(self, exc: Exception):
        print(f"[App] Transcription error: {exc}")
        if not self._is_listening:
            self._overlay.hide()
        self._status.update(self.last_transcript, f"Error: {exc}")

    # ----------------------------------------------------------------
    #  Public
    # ----------------------------------------------------------------

    def show_status(self):
        self._status.show()

    def show_settings(self):
        self._settings.show()

    def quit(self):
        print("[App] Shutting down...")
        self._is_listening = False
        if self._recorder:
            self._recorder.stop()
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        if self._worker:
            self._worker.shutdown()
        self._tray.stop()
        self._root.after(0, self._root.destroy)

    def run(self):
        try:
            self._root.mainloop()
        except KeyboardInterrupt:
            self.quit()


# ============================================================
#  UTILITY
# ============================================================

def _beep(freq: int, dur: int):
    threading.Thread(
        target=lambda: winsound.Beep(freq, dur), daemon=True
    ).start()


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

    print()
    print("=" * 58)
    print("  Global Background Dictation Assistant  v3.0")
    print("=" * 58)
    print("  Press hotkey ONCE  -> Start continuous listening")
    print("  Speak naturally    -> Text auto-injected in chunks")
    print("  Say a command      -> Mouse/keyboard action fired")
    print("  Press hotkey AGAIN -> Stop")
    print("=" * 58)
    print()

    app = DictationAssistant()
    app.initialize()

    hk = app.cfg.get("hotkey").upper()
    print(f"  Hotkey   : {hk}")
    print(f"  Config   : {CONFIG_FILE}")
    print(f"  Tray     : Check bottom-right system tray")
    print()
    print("  Voice Commands (say exactly):")
    print("    Mouse   : click, right click, double click")
    print("    Scroll  : scroll up [N], scroll down [N]")
    print("    Move    : move left/right/up/down [N]")
    print("    Keys    : press enter, press escape, page up/down")
    print("    Hotkeys : select all, copy, paste, undo, save ...")
    print()

    app.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
