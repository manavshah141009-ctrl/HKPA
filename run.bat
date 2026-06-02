@echo off
call venv\Scripts\activate.bat
echo Starting Personal Dictation Assistant (GUI + Background Injector)...
start "" "venv\Scripts\pythonw.exe" assistant.py
venv\Scripts\python.exe app.py
