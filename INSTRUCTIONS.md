# Offline Summarizer - How to Use and Test

This file explains how to run the app, run tests, and train models.

## 1) One-time setup

### Backend (Python)

From project root:

```powershell
python -m pip install -r requirements.txt
```

### Frontend (React/Vite)

```powershell
cd frontend
npm install
cd ..
```

---

## 2) Run the app

### Start backend API

```powershell
python -m backend.app
```

Backend URL: `http://localhost:8000`

### Start frontend UI (in a second terminal)

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in terminal (usually `http://localhost:5173`).

### New GUI: Control Center

The frontend now includes a **Control Center** panel where you can:
- run backend tests,
- run feedback and JSONL training jobs,
- run CNN/DailyMail training with chosen article count,
- stop running jobs,
- view live job logs,
- reload model,
- see current backend/model status.

No terminal is needed for daily test/training after backend + frontend are running.

---

## 3) Quick tests

### A) Verify backend health + summarize endpoint

(Backend must be running)

```powershell
python verify_backend.py
```

### B) Submit sample feedback through API

(Backend must be running)

```powershell
python submit_test_feedback.py
```

### C) Local feedback + training pipeline smoke test

(Does not require frontend; writes feedback then triggers training scripts)

```powershell
python run_training_test.py
```

---

## 4) Daily training workflows

## A) Fast incremental feedback training (DPO)

Use this after collecting corrected feedback (👎 + improved summary).

```powershell
python training/train_incremental.py
```

Useful options:

```powershell
python training/train_incremental.py --max-new-pairs 1
python training/train_incremental.py --max-new-pairs 2
python training/train_incremental.py --quality
python training/train_incremental.py --no-reload
```

What it does:
- Builds incremental feedback dataset
- Trains on only new pairs
- Saves model run under `models/dpo_model_runs/`
- Updates active model pointer
- Reloads backend model automatically (unless `--no-reload`)

## B) Automatic supervised training from JSONL website dataset

Uses `data/training_dataset/website_summarization_training_data.jsonl` by default.

```powershell
python training/train_from_jsonl.py
```

Useful options:

```powershell
python training/train_from_jsonl.py --max-steps 1 --no-reload
python training/train_from_jsonl.py --epochs 5
python training/train_from_jsonl.py --batch-size 1
```

What it does:
- Reads `source_text` and `summary` pairs (category-aware prompts)
- Trains with checkpoint support
- Saves run under `models/sft_model_runs/`
- Updates `models/active_model.txt`
- Reloads backend model automatically (unless `--no-reload`)

## C) CNN/DailyMail training with custom article count

You can train using `data/processed/cnn_dailymail_train.json` and choose article count:

```powershell
python training/train_model.py --max-articles 500 --epochs 1
```

Examples:

```powershell
python training/train_model.py --max-articles 100 --epochs 1
python training/train_model.py --max-articles 1000 --epochs 2
python training/train_model.py --max-articles 200 --epochs 1 --no-reload
```

What it does:
- Trains on first N CNN/DailyMail articles you choose
- Saves run under `models/cnn_model_runs/`
- Updates `models/active_model.txt`
- Reloads backend model automatically (unless `--no-reload`)

### Stop training safely anytime

Press `Ctrl + C` during `train_from_jsonl.py`:
- Current progress is saved
- Model snapshot is still written
- Active model pointer is updated to latest saved run

---

## 5) Model selection behavior

### Single unified model (important)

You do **not** need to manage separate final models.
All training modes (DPO feedback, JSONL SFT, CNN/DailyMail) keep updating one active model chain.
The currently active model is always the one in:

- `models/active_model.txt`

Run folders are retained only as history/checkpoints.

Backend picks model in this priority:
1. `models/active_model.txt`
2. `models/active_dpo_model.txt`
3. fallback model folders (`dpo_model`, `trained_model_new`, `trained_model`)
4. base model (`t5-small`)

You can force backend to reload currently active model:

```powershell
curl -X POST http://localhost:8000/reload_model -H "Content-Type: application/json" -d "{}"
```

### Export full active model

In frontend Control Center -> Unified Model Status:
- click **Export Full Model**
- app downloads a zip of the full currently active model folder
- you can import that zip into another project

---

## 6) Useful status/debug endpoints

When backend is running:
- `GET /` -> API health
- `GET /status` -> current loaded model path and device
- `GET /read_feedback` -> current feedback log
- `POST /reload_model` -> reload model in-memory
- `GET /evaluation/latest` -> latest quality scores
- `GET /evaluation/history` -> previous score runs

## 6.1) Evaluate summary quality (recommended)

Run a quick quality benchmark:

```powershell
python training/evaluate_model.py --max-samples 20
```

This produces comparable scores over time:
- `token_f1`
- `rouge_l_f1`

Results are saved to:
- `data/evaluation/eval_history.jsonl`

In the frontend Control Center:
- click **Evaluate Active Model**
- view scores in the **Quality Scores** panel
- compare latest and historical runs to see improvements.

---

## 7) Common issues

### `git add .` fails
This folder may not be initialized as a Git repository. Initialize first if needed:

```powershell
git init
```

### JSONL parse errors
If your training file is not strict JSONL, the auto trainer has a fallback parser. Training can still proceed, but strict one-object-per-line JSONL is recommended for portability.

### Training is slow on CPU
Use fast incremental mode:

```powershell
python training/train_incremental.py --max-new-pairs 1
```

and increase pairs gradually only when needed.
