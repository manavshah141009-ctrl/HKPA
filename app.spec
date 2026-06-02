# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'faster_whisper', 
        'ctranslate2',
        'pystray',
        'customtkinter',
        'torch',
        'sounddevice',
        'numpy',
        'pyautogui',
        'keyboard',
        'pyperclip'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Grab the CTranslate2 and PyTorch audio DLLs which PyInstaller often misses
import os
import ctranslate2
import torch

ct2_dir = os.path.dirname(ctranslate2.__file__)
torch_dir = os.path.dirname(torch.__file__)

a.binaries += Tree(ct2_dir, prefix='ctranslate2', excludes=['*.pyc'])
a.binaries += Tree(torch_dir, prefix='torch', excludes=['*.pyc'])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PersonalDictationAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # <--- Windowless Execution
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico'      # Make sure to place app_icon.ico in the folder before compiling
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PersonalDictationAssistant',
)
