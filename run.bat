@echo off
call venv\Scripts\activate.bat
echo Starting Personal Dictation Assistant (GUI + Background Injector)...
start "Background Injector" python assistant.py
python app.py
