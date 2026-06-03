"""In-memory async task manager with thread-pool execution."""
import os, json, time, uuid, threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
HISTORY_FILE = os.path.join(UPLOADS_DIR, "history.json")


class Task:
    def __init__(self, task_id: str, image_path: str, output_dir: str,
                 custom_prompt: Optional[str] = None):
        self.task_id = task_id
        self.image_path = image_path
        self.output_dir = output_dir
        self.custom_prompt = custom_prompt
        self.status = "pending"      # pending | running | done | error
        self.stage = ""
        self.stage_status = ""
        self.message = ""
        self.progress = 0            # 0–100 rough estimate
        self.result: Optional[dict] = None
        self.events: list = []       # SSE event buffer
        self.created_at = time.time()
        self.finished_at: Optional[float] = None
        self._lock = threading.Lock()

    def emit(self, stage: str, status: str, message: str, data: Optional[dict] = None):
        event = {
            "stage": stage,
            "status": status,
            "message": message,
            "data": data or {},
            "ts": time.time(),
        }
        with self._lock:
            self.events.append(event)
            self.stage = stage
            self.stage_status = status
            self.message = message
            # Rough progress mapping
            prog_map = {
                "vision": 10, "codegen": 35, "compile": 60,
                "critic": 85, "done": 100, "error": 100,
            }
            self.progress = prog_map.get(stage, self.progress)
            if stage == "done":
                self.status = "done" if status == "success" else "error"
                self.finished_at = time.time()
                self.result = data
            elif stage == "error":
                self.status = "error"
                self.finished_at = time.time()
                self.result = data


class TaskManager:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        self._load_history()

    def create_task(self, image_path: str, custom_prompt: Optional[str] = None) -> Task:
        task_id = uuid.uuid4().hex[:12]
        output_dir = os.path.join(UPLOADS_DIR, task_id)
        os.makedirs(output_dir, exist_ok=True)
        task = Task(task_id, image_path, output_dir, custom_prompt)
        with self._lock:
            self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self.tasks.get(task_id)

    def submit(self, task: Task, fn: Callable, *args, **kwargs):
        task.status = "running"
        task.emit("vision", "running", "Task queued, starting soon...")
        future = self.executor.submit(fn, *args, **kwargs)

        def _on_done(f):
            try:
                f.result()
            except Exception as e:
                task.emit("error", "fail", f"Unhandled exception: {str(e)[:200]}",
                          {"error": str(e)})
                self._persist_task(task)
            else:
                self._persist_task(task)

        future.add_done_callback(_on_done)
        return task.task_id

    def _persist_task(self, task: Task):
        """Append to history JSON file."""
        entry = {
            "task_id": task.task_id,
            "status": task.status,
            "stage": task.stage,
            "progress": task.progress,
            "created_at": task.created_at,
            "finished_at": task.finished_at,
            "compile_ok": task.result.get("compile_ok", False) if task.result else False,
            "score": task.result.get("critic_final_score", 0.0) if task.result else 0.0,
            "custom_prompt": bool(task.custom_prompt),
        }
        history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass
        # Update if exists, else append
        existing = [i for i, h in enumerate(history) if h["task_id"] == task.task_id]
        if existing:
            history[existing[0]] = entry
        else:
            history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _load_history(self):
        if not os.path.exists(HISTORY_FILE):
            return
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            # Don't restore full Task objects, just the list for the API
        except Exception:
            pass

    def list_history(self) -> list:
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []


# Global singleton
manager = TaskManager()
