import tkinter as tk
import keyboard

def on_hotkey():
    print("HOTKEY FIRED!")
    label.config(text="HOTKEY FIRED!")

root = tk.Tk()
label = tk.Label(root, text="Press ctrl+shift+space")
label.pack(padx=50, pady=50)

keyboard.add_hotkey("ctrl+shift+space", on_hotkey)

root.mainloop()
