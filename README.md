# 🎙️ Global Dictation Assistant

> **Real-time, word-for-word voice dictation that works inside any Windows application.**  
> Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with GPU acceleration and VAD-based anti-hallucination.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Auto-Transcribe** | Press hotkey once → speak naturally → text auto-injects in 3.5s chunks |
| **Universal Paste** | Works in WhatsApp, Chrome, Notion, VS Code, Notepad, Explorer — anywhere |
| **GPU Accelerated** | Auto-detects NVIDIA CUDA. Falls back to CPU if unavailable |
| **Anti-Hallucination** | VAD filter + greedy decoding = no random words during silence |
| **System Tray** | Runs silently in background. Purple = idle, Red = recording |
| **Configurable Hotkey** | Change your hotkey anytime via the Settings window |
| **Low Memory** | `int8` quantization keeps RAM under 600MB even with `small.en` |

---

## 🖥️ System Requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 | Windows 11 |
| **RAM** | 4 GB | 8 GB |
| **GPU** | None (CPU mode) | NVIDIA GPU with CUDA |
| **Python** | 3.9+ | 3.11+ |
| **Mic** | Any built-in | USB condenser |

---

## 🚀 Installation

### Step 1 — Clone the repo
```powershell
git clone https://github.com/YOUR_USERNAME/dictation-assistant.git
cd dictation-assistant
```

### Step 2 — Create a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3 — Install PyTorch (choose ONE)

**With NVIDIA GPU (CUDA 12.1):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```powershell
pip install torch torchvision torchaudio
```

### Step 4 — Install all other dependencies
```powershell
pip install -r requirements.txt
```

> **First launch** downloads the Whisper model (~240MB). This only happens once.

---

## ▶️ Usage

### Background Assistant (recommended)
```powershell
.\venv\Scripts\activate
python assistant.py
```

The app disappears to your **system tray** (bottom-right, near the clock).

| Action | Result |
|---|---|
| **Ctrl+Alt+V** (1st press) | 🟢 Green overlay — starts continuous listening |
| **Speak naturally** | Text is auto-pasted into your cursor every ~3.5 seconds |
| **Ctrl+Alt+V** (2nd press) | Stops listening, flushes remaining audio |
| Right-click tray → **Settings** | Change hotkey, model size, chunk length |
| Right-click tray → **Show Status** | See live transcript popup |
| Right-click tray → **Quit** | Exit the app |

### GUI App (standalone window)
```powershell
python app.py
```

---

## ⚙️ Configuration

Settings are saved automatically to `config.json` when you click **Save Settings** in the UI.

You can also edit `config.json` manually:

```json
{
  "hotkey": "ctrl+alt+v",
  "model_size_gpu": "small.en",
  "model_size_cpu": "base.en",
  "audio_segment_sec": 3.5,
  "clipboard_delay": 0.8
}
```

See [`config.example.json`](config.example.json) for all available options.

### Model Size Reference

| Model | RAM Usage | Speed | Best For |
|---|---|---|---|
| `tiny.en` | ~200 MB | Fastest | Quick notes, fast CPU |
| `base.en` | ~300 MB | Fast | CPU fallback (default) |
| `small.en` | ~600 MB | Medium | **GPU default** ✅ |
| `medium.en` | ~1.5 GB | Slow | Maximum accuracy |

---

## 🛡️ Anti-Hallucination Stack

The app uses 4 layers to ensure **literal, word-for-word** transcription:

1. **`vad_filter=True`** — Silero VAD skips all silent audio segments
2. **`temperature=0.0`** — Greedy decoding (no randomness, fully deterministic)
3. **`condition_on_previous_text=False`** — Each chunk is independent (no carry-over)
4. **`no_speech_threshold=0.6`** — Discards segments where Whisper is < 60% confident

---

## 📁 Project Structure

```
dictation-assistant/
├── assistant.py          ← Background tray app (v2 — recommended)
├── app.py                ← Standalone GUI window (v1)
├── requirements.txt      ← Python dependencies
├── config.example.json   ← Example configuration file
├── setup.bat             ← One-click Windows setup script
├── LICENSE               ← MIT License
└── README.md             ← This file
```

---

## 🔧 Troubleshooting

**Hotkey doesn't work in some apps**  
→ Run the terminal as **Administrator** (right-click → Run as administrator)

**Very slow on CPU**  
→ Switch to `tiny.en` in Settings → Chunk length to `2.0s`

**"No module named X"**  
→ Make sure your venv is activated: `.\venv\Scripts\activate`

**App crashes on launch**  
→ Delete `config.json` and restart — it will regenerate with defaults

**Microphone not detected**  
→ Windows Settings → Privacy → Microphone → Allow desktop apps

---

## 📄 License

MIT — see [LICENSE](LICENSE)
