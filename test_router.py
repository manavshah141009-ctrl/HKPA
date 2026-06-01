import time
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = ""

# Import from assistant
from assistant import SemanticRouter

def on_dictation(text):
    print(f"DICTATION: {text}")

def on_command(text, status):
    print(f"COMMAND: {text} -> {status}")

print("Initializing router...")
router = SemanticRouter(on_dictation, on_command)
time.sleep(1)

print("Submitting command...")
router.submit("move the mouse to the center")
time.sleep(3)

print("Submitting command...")
router.submit("open anti gravity")
time.sleep(3)

print("Submitting dictation...")
router.submit("hello world this is a test")
time.sleep(3)

print("Done")
