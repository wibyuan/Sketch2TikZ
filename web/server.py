"""FastAPI entry point for the Sketch2TikZ web UI."""
import os, sys, webbrowser
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root on path so train.* imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.api import router

app = FastAPI(title="Sketch2TikZ", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve static frontend files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SKETCH2TIKZ_HOST", "127.0.0.1")
    port = int(os.getenv("SKETCH2TIKZ_PORT", "8000"))
    url = f"http://{host}:{port}"
    print(f"\n[START] Sketch2TikZ web UI starting at {url}")
    print("        Press Ctrl+C to stop\n")
    # Open browser after a short delay
    def _open():
        import time
        time.sleep(1.5)
        webbrowser.open(url)
    import threading
    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run("web.server:app", host=host, port=port, reload=False, log_level="warning")
