from pathlib import Path

def checkpath(p):
    Path(p).mkdir(parents=True, exist_ok=True)