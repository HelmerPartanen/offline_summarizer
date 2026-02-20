from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import json
import uuid
import subprocess
import threading
import shutil
from datetime import datetime
try:
    from backend.summarizer import summarize, summarizer_instance, reload_model
except ImportError:
    try:
        from summarizer import summarize, summarizer_instance, reload_model
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from backend.summarizer import summarize, summarizer_instance, reload_model

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_EXE = sys.executable
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports")

JOBS = {}
RUNNING_PROCESSES = {}
JOB_LOCK = threading.Lock()


def _append_job_output(job, line):
    job["output"].append(line)
    if len(job["output"]) > 2000:
        job["output"] = job["output"][-2000:]


def _build_job_command(job_type, args):
    script_map = {
        "verify_backend": [PYTHON_EXE, "verify_backend.py"],
        "submit_test_feedback": [PYTHON_EXE, "submit_test_feedback.py"],
        "run_training_test": [PYTHON_EXE, "run_training_test.py"],
        "train_incremental": [PYTHON_EXE, os.path.join("training", "train_incremental.py")],
        "train_from_jsonl": [PYTHON_EXE, os.path.join("training", "train_from_jsonl.py")],
        "train_cnn": [PYTHON_EXE, os.path.join("training", "train_model.py")],
        "evaluate_model": [PYTHON_EXE, os.path.join("training", "evaluate_model.py")],
    }
    if job_type not in script_map:
        raise ValueError(f"Unsupported job type: {job_type}")

    extra_args = args if args else []
    return script_map[job_type] + extra_args


def _resolve_active_model_dir():
    model_dir = os.path.join(PROJECT_ROOT, "models")
    pointers = [
        os.path.join(model_dir, "active_model.txt"),
        os.path.join(model_dir, "active_dpo_model.txt"),
    ]
    for pointer_file in pointers:
        if os.path.exists(pointer_file):
            with open(pointer_file, "r", encoding="utf-8") as f:
                pointed = f.read().strip()
            if pointed and os.path.exists(os.path.join(pointed, "config.json")):
                return os.path.abspath(pointed)

    runtime_model = getattr(summarizer_instance, "model_path", None)
    if runtime_model and os.path.exists(os.path.join(runtime_model, "config.json")):
        return os.path.abspath(runtime_model)

    return None


def _run_job(job_id):
    with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"

    process = None
    try:
        process = subprocess.Popen(
            job["command"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with JOB_LOCK:
            RUNNING_PROCESSES[job_id] = process
            job["pid"] = process.pid

        if process.stdout:
            for line in process.stdout:
                with JOB_LOCK:
                    _append_job_output(job, line.rstrip())

        exit_code = process.wait()

        with JOB_LOCK:
            job["exit_code"] = exit_code
            job["ended_at"] = datetime.now().isoformat()
            if job["status"] == "stopping":
                job["status"] = "stopped"
            else:
                job["status"] = "completed" if exit_code == 0 else "failed"
    except Exception as e:
        with JOB_LOCK:
            job["status"] = "failed"
            job["ended_at"] = datetime.now().isoformat()
            job["exit_code"] = -1
            _append_job_output(job, f"[job-error] {str(e)}")
    finally:
        with JOB_LOCK:
            if job_id in RUNNING_PROCESSES:
                del RUNNING_PROCESSES[job_id]

# Allow CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_endpoint(request: SummarizeRequest):
    try:
        summary = summarize(request.text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

class FeedbackRequest(BaseModel):
    text: str
    rejected_summary: str
    chosen_summary: str


class ReloadRequest(BaseModel):
    model_path: str | None = None


class RunJobRequest(BaseModel):
    job_type: str
    args: list[str] = []

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    try:
        print(f"--- API RECEIVED FEEDBACK ---")
        print(f"Request data: {request.dict()}")
        summarizer_instance.log_feedback(
            request.text, 
            request.rejected_summary, 
            request.chosen_summary
        )
        return {"status": "success", "message": "Feedback recorded."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def status_endpoint():
    return {
        "model_path": summarizer_instance.model_path,
        "device": summarizer_instance.device
    }


@app.post("/jobs/run")
def run_job_endpoint(request: RunJobRequest):
    try:
        command = _build_job_command(request.job_type, request.args)
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": request.job_type,
            "args": request.args,
            "command": command,
            "status": "queued",
            "pid": None,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "exit_code": None,
            "output": [],
        }
        with JOB_LOCK:
            JOBS[job_id] = job

        worker = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        worker.start()

        return {
            "status": "started",
            "job_id": job_id,
            "job": job,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/jobs")
def list_jobs_endpoint():
    with JOB_LOCK:
        jobs = []
        for job in JOBS.values():
            jobs.append(
                {
                    "id": job["id"],
                    "type": job["type"],
                    "status": job["status"],
                    "pid": job["pid"],
                    "started_at": job["started_at"],
                    "ended_at": job["ended_at"],
                    "exit_code": job["exit_code"],
                    "output_tail": job["output"][-20:],
                }
            )
        jobs.sort(key=lambda item: item["started_at"], reverse=True)
        return {"jobs": jobs}


@app.get("/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {"error": "Job not found"}
        return {"job": job}


@app.post("/jobs/{job_id}/stop")
def stop_job_endpoint(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {"error": "Job not found"}
        process = RUNNING_PROCESSES.get(job_id)
        if not process:
            return {"status": "not_running", "job_id": job_id}
        job["status"] = "stopping"

    try:
        process.terminate()
        return {"status": "stopping", "job_id": job_id}
    except Exception as e:
        return {"error": str(e)}


@app.post("/reload_model")
def reload_model_endpoint(request: ReloadRequest):
    try:
        return reload_model(request.model_path)
    except Exception as e:
        return {"error": str(e)}


@app.get("/model/export")
def export_model_endpoint():
    try:
        active_model_dir = _resolve_active_model_dir()
        if not active_model_dir:
            return {"error": "No active trained model found to export."}

        os.makedirs(EXPORT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"offline_summarizer_model_{stamp}"
        archive_root = os.path.join(EXPORT_DIR, base_name)
        zip_path = shutil.make_archive(archive_root, "zip", active_model_dir)

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=os.path.basename(zip_path),
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/evaluation/latest")
def latest_evaluation_endpoint():
    try:
        history_file = os.path.join(PROJECT_ROOT, "data", "evaluation", "eval_history.jsonl")
        if not os.path.exists(history_file):
            return {"evaluation": None}
        with open(history_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return {"evaluation": None}
        return {"evaluation": json.loads(lines[-1])}
    except Exception as e:
        return {"error": str(e)}


@app.get("/evaluation/history")
def evaluation_history_endpoint(limit: int = 10):
    try:
        history_file = os.path.join(PROJECT_ROOT, "data", "evaluation", "eval_history.jsonl")
        if not os.path.exists(history_file):
            return {"history": []}
        with open(history_file, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = rows[-max(1, min(limit, 100)) :]
        rows.reverse()
        return {"history": rows}
    except Exception as e:
        return {"error": str(e)}

@app.get("/read_feedback")
def read_feedback_endpoint():
    import os
    feedback_file = os.path.join(os.path.dirname(__file__), "..", "data", "feedback", "feedback.jsonl")
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            return {"feedback": f.readlines()}
    return {"feedback": [], "path": os.path.abspath(feedback_file)}

@app.get("/")
def read_root():
    return {"message": "Offline Summarizer API is running."}

if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
