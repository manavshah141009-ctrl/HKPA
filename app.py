"""
============================================================
  Personal Dictation Assistant
  ============================================================
  A real-time, word-for-word voice transcription desktop app.
  
  Tech Stack:
    - faster-whisper  : Optimized Whisper model (small.en, int8)
    - sounddevice     : Microphone audio capture
    - customtkinter   : Modern GUI framework
    - keyboard        : Global hotkey support

  Architecture:
    - Main Thread      → GUI (CustomTkinter event loop)
    - AudioThread      → Continuous microphone recording
    - TranscribeThread → faster-whisper inference

  Key Design Decisions:
    - VAD filter ON  : Prevents hallucinations during silence
    - int8 quantization : Keeps RAM under 1.5GB
    - Auto CUDA detection : Uses GPU if available, falls back to CPU
    - Thread-safe Queue : Audio and transcription pass through queues
                          so threads never directly touch GUI widgets
============================================================
"""

import tkinter as tk
import customtkinter as ctk
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import pyperclip
import keyboard
import torch
import sys
import os
from faster_whisper import WhisperModel


# ============================================================
#  CONFIGURATION — Tweak these values to adjust behavior
# ============================================================

# Audio settings
SAMPLE_RATE       = 16000   # Whisper expects 16kHz mono audio
CHANNELS          = 1       # Mono
BLOCK_SIZE        = 4000    # Frames per sounddevice callback (~250ms chunks)
SILENCE_THRESHOLD = 0.5     # Seconds of silence before sending buffer to Whisper

# Whisper model settings
MODEL_SIZE        = "small.en"   # Options: tiny.en, base.en, small.en
COMPUTE_TYPE      = "int8"       # int8 = ~600MB RAM, great for 4GB systems
VAD_FILTER        = True         # CRITICAL: Prevents hallucinations during silence

# How long (in seconds) of accumulated audio to send to Whisper at once
# Shorter = more "live" but more API calls. 3-4s is a good balance.
AUDIO_SEGMENT_SECONDS = 3.5


# ============================================================
#  HARDWARE DETECTION
# ============================================================

def detect_device() -> tuple[str, str]:
    """
    Automatically detect whether CUDA (NVIDIA GPU) is available.
    Returns a tuple of (device_string, compute_type_string) for WhisperModel.
    
    We check torch.cuda.is_available() which returns True only if:
      1. An NVIDIA GPU is present
      2. CUDA drivers are installed
      3. PyTorch was installed with CUDA support
    """
    if torch.cuda.is_available():
        device      = "cuda"
        compute_type = "int8_float16"   # Faster on GPU: INT8 weights, FP16 ops
        gpu_name    = torch.cuda.get_device_name(0)
        print(f"[HW] GPU detected: {gpu_name}. Running on CUDA with int8_float16.")
    else:
        device      = "cpu"
        compute_type = "int8"           # Pure INT8 for CPU keeps RAM very low
        print("[HW] No CUDA GPU found. Running on CPU with int8 quantization.")
    
    return device, compute_type


# ============================================================
#  WHISPER MODEL LOADER
# ============================================================

def load_whisper_model(device: str, compute_type: str) -> WhisperModel:
    """
    Load the faster-whisper model into memory.
    
    On first run, the model (~240MB for small.en) will be automatically
    downloaded to ~/.cache/huggingface/hub/
    
    Args:
        device       : "cuda" or "cpu"
        compute_type : "int8_float16" (GPU) or "int8" (CPU)
    
    Returns:
        A loaded WhisperModel instance ready for transcription.
    """
    print(f"[Model] Loading {MODEL_SIZE} on {device} ({compute_type})...")
    model = WhisperModel(
        MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        # cpu_threads controls parallelism on CPU — 4 is a balanced default
        cpu_threads=4,
    )
    print("[Model] Model loaded successfully.")
    return model


# ============================================================
#  AUDIO RECORDER THREAD
# ============================================================

class AudioRecorderThread(threading.Thread):
    """
    Runs in a background thread to continuously capture audio from
    the default system microphone using sounddevice.
    
    It accumulates raw audio samples into a rolling buffer and, once
    AUDIO_SEGMENT_SECONDS of audio is collected, pushes the buffer
    into the `audio_queue` for the TranscriberThread to process.
    """

    def __init__(self, audio_queue: queue.Queue, status_callback):
        """
        Args:
            audio_queue     : Thread-safe queue shared with TranscriberThread.
                              This thread puts audio arrays; the transcriber gets them.
            status_callback : Function to call to update the GUI status label.
        """
        super().__init__(daemon=True)   # daemon=True: thread dies when main app exits
        self.audio_queue     = audio_queue
        self.status_callback = status_callback
        self._stop_event     = threading.Event()   # Signal to stop the loop
        self.audio_buffer    = []                  # Accumulated raw audio chunks

    def stop(self):
        """Signal the thread to stop recording."""
        self._stop_event.set()

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def run(self):
        """
        Main thread loop. Opens a sounddevice InputStream that fires
        a callback each time BLOCK_SIZE audio frames arrive from the mic.
        The callback appends each chunk to our internal buffer.
        
        When enough audio has accumulated (AUDIO_SEGMENT_SECONDS), we
        concatenate the buffer, normalize it to float32 [-1.0, 1.0],
        and enqueue it for Whisper to transcribe.
        """
        # This internal queue is used only within this thread to
        # pass data from the sounddevice callback (which runs in a
        # C-level audio thread) safely into our Python thread.
        _internal_q = queue.Queue()

        def _sd_callback(indata, frames, time_info, status):
            """
            sounddevice calls this function every BLOCK_SIZE frames.
            We must return quickly, so we just put the data in a queue.
            DO NOT do heavy work (like Whisper inference) here.
            """
            if status:
                # Report xruns (buffer under/overflows) to console
                print(f"[Audio] sounddevice status: {status}")
            # indata is a (frames, channels) numpy array — copy it to avoid corruption
            _internal_q.put(indata.copy())

        self.status_callback("🎙️ Listening...", "green")

        # Open the microphone stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",         # sounddevice delivers float32 by default
            blocksize=BLOCK_SIZE,
            callback=_sd_callback,
        ):
            # Frame counter to know when we have AUDIO_SEGMENT_SECONDS of audio
            frames_needed = int(SAMPLE_RATE * AUDIO_SEGMENT_SECONDS)
            frame_count   = 0

            while not self._stop_event.is_set():
                try:
                    # Grab the next chunk from the internal sounddevice queue
                    chunk = _internal_q.get(timeout=0.1)
                except queue.Empty:
                    continue   # Nothing yet, loop and check stop_event again

                # Flatten from (frames, 1) to (frames,) for Whisper
                self.audio_buffer.append(chunk.flatten())
                frame_count += chunk.shape[0]

                # Once we have enough audio, send the segment to Whisper
                if frame_count >= frames_needed:
                    # Concatenate all accumulated chunks into one flat array
                    audio_segment = np.concatenate(self.audio_buffer, axis=0)
                    self.audio_queue.put(audio_segment)
                    # Reset buffer for the next segment
                    self.audio_buffer = []
                    frame_count       = 0

        self.status_callback("⏹️ Stopped", "gray")


# ============================================================
#  TRANSCRIBER THREAD
# ============================================================

class TranscriberThread(threading.Thread):
    """
    Runs in a background thread to transcribe audio segments using
    faster-whisper. It picks up audio arrays from the `audio_queue`,
    runs Whisper inference, and sends results via `text_callback`.
    
    This ensures the Whisper model (which can be slow on CPU) never
    blocks the GUI or the audio capture threads.
    """

    def __init__(
        self,
        model: WhisperModel,
        audio_queue: queue.Queue,
        text_callback,
        status_callback,
    ):
        """
        Args:
            model           : Pre-loaded WhisperModel instance.
            audio_queue     : Thread-safe queue. This thread gets audio; 
                              AudioRecorderThread puts audio.
            text_callback   : Function to call with transcribed text string.
            status_callback : Function to call to update the GUI status label.
        """
        super().__init__(daemon=True)
        self.model           = model
        self.audio_queue     = audio_queue
        self.text_callback   = text_callback
        self.status_callback = status_callback
        self._stop_event     = threading.Event()

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        """
        Main transcriber loop. Waits for audio segments from the queue,
        then runs faster-whisper inference on each one.
        
        faster-whisper returns a generator of Segment objects. Each segment
        has a `.text` attribute with the transcribed words.
        
        VAD (Voice Activity Detection) is enabled to skip silent segments
        and prevent Whisper from hallucinating words into silence.
        """
        while not self._stop_event.is_set():
            try:
                # Block until audio is available (with timeout to check stop_event)
                audio_data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # --- Run Whisper Inference ---
            self.status_callback("⚡ Transcribing...", "orange")

            try:
                # transcribe() returns (segments_generator, TranscriptionInfo)
                segments, info = self.model.transcribe(
                    audio_data,
                    language="en",           # Lock to English for speed
                    beam_size=5,             # Beam search width (5 = balanced accuracy)
                    vad_filter=VAD_FILTER,   # CRITICAL: Skip silent chunks
                    vad_parameters=dict(
                        min_silence_duration_ms=300,   # Silence gap to split segments
                        threshold=0.5,                 # Voice probability threshold (0-1)
                    ),
                    # No initial_prompt — we want literal transcription, not guided output
                    condition_on_previous_text=False,  # Prevents hallucination carry-over
                    temperature=0.0,                   # Greedy decoding = most accurate
                    no_speech_threshold=0.6,           # Skip if speech probability < 60%
                    log_prob_threshold=-1.0,           # Skip low-confidence segments
                    compression_ratio_threshold=2.4,
                )

                # Collect all segment texts into one string
                full_text = ""
                for segment in segments:
                    # Whisper sometimes adds leading/trailing whitespace — strip it
                    text = segment.text.strip()
                    if text:
                        full_text += text + " "

                # Only update UI if we actually got text
                if full_text.strip():
                    self.text_callback(full_text.strip())

            except Exception as e:
                print(f"[Transcriber] Error during transcription: {e}")

            finally:
                self.status_callback("🎙️ Listening...", "green")


# ============================================================
#  MAIN APPLICATION GUI
# ============================================================

class DictationApp(ctk.CTk):
    """
    The main GUI window of the Dictation Assistant.
    
    Uses CustomTkinter for a modern, dark-mode interface.
    All UI updates triggered by background threads are routed
    through `self.after()` to ensure thread safety — Tkinter
    widgets must ONLY be updated from the main (GUI) thread.
    """

    # ---- Color Palette ----
    COLOR_BG          = "#0F1117"   # Deep dark background
    COLOR_PANEL       = "#1A1D27"   # Slightly lighter panel background
    COLOR_ACCENT      = "#6C63FF"   # Purple accent (buttons, highlights)
    COLOR_ACCENT_DARK = "#4E46C1"   # Darker accent for hover states
    COLOR_TEXT        = "#E8E8F0"   # Primary text color
    COLOR_SUBTEXT     = "#8888AA"   # Secondary/muted text
    COLOR_SUCCESS     = "#2ECC71"   # Green for "Listening" state
    COLOR_WARN        = "#F39C12"   # Orange for "Transcribing" state

    def __init__(self, model: WhisperModel, device: str):
        """
        Initialize the app window and all child widgets.
        
        Args:
            model  : Pre-loaded WhisperModel instance.
            device : "cuda" or "cpu" — displayed in the status bar.
        """
        super().__init__()

        self.model  = model
        self.device = device

        # Thread management
        self.audio_thread      = None
        self.transcriber_thread = None
        self.audio_queue       = queue.Queue(maxsize=10)  # Max 10 segments buffered
        self.is_listening      = False

        # Build the UI
        self._configure_window()
        self._build_ui()

        # Register the global hotkey Ctrl+Shift+Space to toggle listening
        keyboard.add_hotkey("ctrl+shift+space", self._toggle_listening_hotkey)

        # Handle window close gracefully
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------------
    #  Window Configuration
    # ----------------------------------------------------------------

    def _configure_window(self):
        """Set window title, size, colors, and dark mode appearance."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("🎙️ Personal Dictation Assistant")
        self.geometry("900x680")
        self.minsize(700, 500)
        self.configure(fg_color=self.COLOR_BG)

        # Center the window on screen
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w, win_h = 900, 680
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # ----------------------------------------------------------------
    #  UI Construction
    # ----------------------------------------------------------------

    def _build_ui(self):
        """Construct all UI widgets and lay them out."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)   # Row 1 (text area) expands

        # ---- Header ----
        self._build_header()

        # ---- Transcript Area ----
        self._build_transcript_area()

        # ---- Controls ----
        self._build_controls()

        # ---- Status Bar ----
        self._build_status_bar()

    def _build_header(self):
        """Top header bar with app title and device info."""
        header = ctk.CTkFrame(
            self,
            fg_color=self.COLOR_PANEL,
            corner_radius=0,
            height=60,
        )
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_columnconfigure(1, weight=1)

        # App icon + title
        title_label = ctk.CTkLabel(
            header,
            text="  🎙️  Personal Dictation Assistant",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=self.COLOR_TEXT,
        )
        title_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # Device badge (shows GPU or CPU)
        device_text  = f"⚡ GPU (CUDA)" if self.device == "cuda" else "🖥️ CPU"
        device_color = self.COLOR_SUCCESS if self.device == "cuda" else self.COLOR_SUBTEXT
        device_badge = ctk.CTkLabel(
            header,
            text=f"{device_text}  •  Model: {MODEL_SIZE}  •  Hotkey: Ctrl+Shift+Space",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=device_color,
        )
        device_badge.grid(row=0, column=1, padx=20, pady=15, sticky="e")

    def _build_transcript_area(self):
        """Main text area where transcribed text appears."""
        # Wrapper frame with a subtle border effect
        text_frame = ctk.CTkFrame(
            self,
            fg_color=self.COLOR_PANEL,
            corner_radius=12,
        )
        text_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(15, 10))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(1, weight=1)

        # Label above the text area
        label = ctk.CTkLabel(
            text_frame,
            text="Transcript",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=self.COLOR_SUBTEXT,
        )
        label.grid(row=0, column=0, padx=15, pady=(12, 4), sticky="w")

        # The main transcript textbox
        self.transcript_box = ctk.CTkTextbox(
            text_frame,
            font=ctk.CTkFont(family="Segoe UI", size=16),
            fg_color="#12141C",          # Darker than the panel
            text_color=self.COLOR_TEXT,
            border_width=0,
            corner_radius=8,
            wrap="word",                 # Wrap at word boundaries
            scrollbar_button_color=self.COLOR_ACCENT,
        )
        self.transcript_box.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=(0, 10)
        )

        # Placeholder text
        self._set_placeholder()

    def _set_placeholder(self):
        """Insert placeholder hint text into the empty transcript box."""
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert(
            "1.0",
            "Press  ▶ Start Listening  or  Ctrl+Shift+Space  to begin dictating...\n\n"
            "Your words will appear here in real time.",
        )
        self.transcript_box.configure(
            state="disabled",   # Read-only for the placeholder
            text_color=self.COLOR_SUBTEXT,
        )
        self._placeholder_active = True

    def _clear_placeholder(self):
        """Remove placeholder text and enable editing."""
        if getattr(self, "_placeholder_active", False):
            self.transcript_box.configure(state="normal", text_color=self.COLOR_TEXT)
            self.transcript_box.delete("1.0", "end")
            self._placeholder_active = False

    def _build_controls(self):
        """Bottom control strip with action buttons."""
        controls = ctk.CTkFrame(
            self,
            fg_color=self.COLOR_PANEL,
            corner_radius=12,
            height=70,
        )
        controls.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))
        # Allow buttons to expand evenly
        controls.grid_columnconfigure((0, 1, 2), weight=1)

        # ---- Start/Stop Toggle Button ----
        self.toggle_btn = ctk.CTkButton(
            controls,
            text="▶  Start Listening",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            fg_color=self.COLOR_ACCENT,
            hover_color=self.COLOR_ACCENT_DARK,
            corner_radius=10,
            height=44,
            command=self._toggle_listening,
        )
        self.toggle_btn.grid(row=0, column=0, padx=15, pady=13, sticky="ew")

        # ---- Clear Button ----
        clear_btn = ctk.CTkButton(
            controls,
            text="🗑  Clear Text",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color="#2A2D3E",
            hover_color="#353850",
            corner_radius=10,
            height=44,
            command=self._clear_text,
        )
        clear_btn.grid(row=0, column=1, padx=15, pady=13, sticky="ew")

        # ---- Copy to Clipboard Button ----
        copy_btn = ctk.CTkButton(
            controls,
            text="📋  Copy to Clipboard",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color="#2A2D3E",
            hover_color="#353850",
            corner_radius=10,
            height=44,
            command=self._copy_to_clipboard,
        )
        copy_btn.grid(row=0, column=2, padx=15, pady=13, sticky="ew")

    def _build_status_bar(self):
        """Bottom status bar showing current microphone/processing state."""
        status_frame = ctk.CTkFrame(
            self,
            fg_color=self.COLOR_PANEL,
            corner_radius=0,
            height=36,
        )
        status_frame.grid(row=3, column=0, sticky="ew", padx=0, pady=0)
        status_frame.grid_columnconfigure(1, weight=1)

        # Animated pulse dot (canvas-based)
        self.pulse_canvas = tk.Canvas(
            status_frame,
            width=16,
            height=16,
            bg=self.COLOR_PANEL,
            highlightthickness=0,
        )
        self.pulse_canvas.grid(row=0, column=0, padx=(16, 4), pady=10)
        self._pulse_dot = self.pulse_canvas.create_oval(2, 2, 14, 14, fill="gray", outline="")

        # Status text label
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="⏹️ Ready  —  Press Start or Ctrl+Shift+Space",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=self.COLOR_SUBTEXT,
        )
        self.status_label.grid(row=0, column=1, padx=4, pady=10, sticky="w")

        self._pulse_running = False

    # ----------------------------------------------------------------
    #  Pulse Animation (dot in status bar)
    # ----------------------------------------------------------------

    def _start_pulse(self):
        """Begin animating the pulse dot in the status bar."""
        self._pulse_running = True
        self._pulse_state   = False
        self._animate_pulse()

    def _stop_pulse(self):
        """Stop the pulse animation and reset dot to gray."""
        self._pulse_running = False
        self.pulse_canvas.itemconfig(self._pulse_dot, fill="gray")

    def _animate_pulse(self):
        """Alternate the dot color between green and dark green every 600ms."""
        if not self._pulse_running:
            return
        color = "#2ECC71" if self._pulse_state else "#1A6E3C"
        self.pulse_canvas.itemconfig(self._pulse_dot, fill=color)
        self._pulse_state = not self._pulse_state
        self.after(600, self._animate_pulse)

    # ----------------------------------------------------------------
    #  Thread-Safe GUI Update Callbacks
    # ----------------------------------------------------------------

    def _append_text_safe(self, text: str):
        """
        Thread-safe method to append transcribed text to the textbox.
        MUST be called via self.after() from non-GUI threads.
        """
        self._clear_placeholder()
        self.transcript_box.configure(state="normal")
        
        # Add a space before appending if text already exists
        current = self.transcript_box.get("1.0", "end-1c")
        if current.strip():
            self.transcript_box.insert("end", " " + text)
        else:
            self.transcript_box.insert("end", text)
        
        # Auto-scroll to the latest text
        self.transcript_box.see("end")
        # Keep it editable so user can manually correct text
        # self.transcript_box.configure(state="disabled")

    def _update_status_safe(self, message: str, color: str):
        """
        Thread-safe method to update the status bar label and dot color.
        MUST be called via self.after() from non-GUI threads.
        """
        self.status_label.configure(text=message)
        color_map = {
            "green" : self.COLOR_SUCCESS,
            "orange": self.COLOR_WARN,
            "gray"  : self.COLOR_SUBTEXT,
        }
        dot_color = color_map.get(color, self.COLOR_SUBTEXT)
        self.pulse_canvas.itemconfig(self._pulse_dot, fill=dot_color)

    def _on_text_received(self, text: str):
        """
        Called by TranscriberThread when new text is ready.
        Routes the GUI update to the main thread via self.after().
        """
        self.after(0, self._append_text_safe, text)

    def _on_status_changed(self, message: str, color: str):
        """
        Called by AudioRecorderThread or TranscriberThread to update status.
        Routes the GUI update to the main thread via self.after().
        """
        self.after(0, self._update_status_safe, message, color)

    # ----------------------------------------------------------------
    #  Core Actions
    # ----------------------------------------------------------------

    def _toggle_listening(self):
        """Toggle between listening and stopped states."""
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _toggle_listening_hotkey(self):
        """Called by the global keyboard hotkey. Routes to main thread."""
        self.after(0, self._toggle_listening)

    def _start_listening(self):
        """Start audio capture and transcription threads."""
        self.is_listening = True

        # Update button appearance
        self.toggle_btn.configure(
            text="⏹  Stop Listening",
            fg_color="#C0392B",
            hover_color="#922B21",
        )

        self._clear_placeholder()
        self._start_pulse()

        # Create a fresh queue for this session
        self.audio_queue = queue.Queue(maxsize=10)

        # Start the audio recorder thread
        self.audio_thread = AudioRecorderThread(
            audio_queue=self.audio_queue,
            status_callback=self._on_status_changed,
        )
        self.audio_thread.start()

        # Start the transcriber thread
        self.transcriber_thread = TranscriberThread(
            model=self.model,
            audio_queue=self.audio_queue,
            text_callback=self._on_text_received,
            status_callback=self._on_status_changed,
        )
        self.transcriber_thread.start()

    def _stop_listening(self):
        """Stop audio capture and transcription threads."""
        self.is_listening = False

        # Update button appearance
        self.toggle_btn.configure(
            text="▶  Start Listening",
            fg_color=self.COLOR_ACCENT,
            hover_color=self.COLOR_ACCENT_DARK,
        )

        self._stop_pulse()
        self._update_status_safe("⏹️ Stopped  —  Press Start or Ctrl+Shift+Space", "gray")

        # Signal threads to stop
        if self.audio_thread:
            self.audio_thread.stop()
        if self.transcriber_thread:
            self.transcriber_thread.stop()

    def _clear_text(self):
        """Clear the transcript text area and reset placeholder."""
        self._set_placeholder()

    def _copy_to_clipboard(self):
        """Copy all transcript text to the system clipboard."""
        text = self.transcript_box.get("1.0", "end-1c").strip()
        if text and not getattr(self, "_placeholder_active", False):
            pyperclip.copy(text)
            # Brief visual feedback: flash the status bar
            self._update_status_safe("✅ Copied to clipboard!", "green")
            self.after(2000, lambda: self._update_status_safe(
                "⏹️ Ready  —  Press Start or Ctrl+Shift+Space", "gray"
            ))

    def _on_close(self):
        """Gracefully stop all threads before destroying the window."""
        self._stop_listening()
        # Give threads a moment to stop
        time.sleep(0.3)
        keyboard.unhook_all()
        self.destroy()


# ============================================================
#  SPLASH / LOADING SCREEN
# ============================================================

class SplashScreen(ctk.CTkToplevel):
    """
    A loading screen shown while the Whisper model is being loaded.
    Model loading can take 5-15 seconds on first run (model download)
    or 2-5 seconds on subsequent runs (loading from disk cache).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("")
        self.geometry("400x200")
        self.resizable(False, False)
        self.configure(fg_color="#1A1D27")
        self.overrideredirect(True)    # No title bar — clean look

        # Center it
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 400) // 2
        y = (self.winfo_screenheight() - 200) // 2
        self.geometry(f"400x200+{x}+{y}")

        # Content
        ctk.CTkLabel(
            self,
            text="🎙️ Personal Dictation Assistant",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color="#E8E8F0",
        ).pack(pady=(30, 8))

        self.status_lbl = ctk.CTkLabel(
            self,
            text="Initializing...",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#8888AA",
        )
        self.status_lbl.pack(pady=4)

        self.progress = ctk.CTkProgressBar(
            self,
            mode="indeterminate",
            progress_color="#6C63FF",
            fg_color="#2A2D3E",
        )
        self.progress.pack(pady=16, padx=40, fill="x")
        self.progress.start()

    def update_status(self, text: str):
        self.status_lbl.configure(text=text)
        self.update()

    def safe_destroy(self):
        """Stop the progress bar animation before destroying to prevent
        'invalid command name' Tkinter warnings from pending after() callbacks."""
        try:
            self.progress.stop()
        except Exception:
            pass
        self.destroy()


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    """
    Application entry point.
    
    1. Creates a hidden root window + splash screen.
    2. Detects CUDA/CPU hardware.
    3. Loads the Whisper model in the main thread
       (faster-whisper is NOT thread-safe during model init).
    4. Shows the main DictationApp window.
    5. Starts the Tkinter event loop.
    """
    # Create a temporary hidden root to host the splash
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    root.withdraw()   # Hide the temporary root

    splash = SplashScreen(root)
    splash.update_status("Detecting hardware...")
    splash.update()

    # Step 1: Detect device
    device, compute_type = detect_device()

    device_msg = "CUDA GPU" if device == "cuda" else "CPU"
    splash.update_status(f"Loading Whisper ({MODEL_SIZE}) on {device_msg}...")
    splash.update()

    # Step 2: Load the model (this is the slow step)
    try:
        model = load_whisper_model(device, compute_type)
    except Exception as e:
        splash.destroy()
        root.destroy()
        print(f"FATAL: Could not load Whisper model. Error: {e}")
        sys.exit(1)

    splash.update_status("Launching UI...")
    splash.update()
    time.sleep(0.3)   # Brief pause for visual polish

    # Step 3: Destroy splash and launch main app
    splash.safe_destroy()
    root.destroy()

    # Create and run the main application
    app = DictationApp(model=model, device=device)
    app.mainloop()


if __name__ == "__main__":
    main()
