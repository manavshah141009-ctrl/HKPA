import customtkinter as ctk
import keyboard

def on_hk():
    print("HK FIRED")

app = ctk.CTk()
keyboard.add_hotkey("ctrl+alt+v", on_hk)
app.mainloop()
