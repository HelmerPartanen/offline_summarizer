# Offline Summarizer

Fast local summarization with incremental feedback training.

## Daily workflow (1 command training)

1. Start backend:

```bash
python -m backend.app
```

2. Use the frontend to summarize text and submit corrected feedback (👎 + improved summary).

3. Run incremental training:

```bash
python training/train_incremental.py
```

That command will:
- prepare feedback datasets,
- train only on **new** corrected feedback pairs,
- continue from your latest tuned model,
- update a training manifest,
- trigger backend model hot-reload automatically (if backend is running).

## Automatic training from JSONL dataset

For your `data/training_dataset/website_summarization_training_data.jsonl` file:

```bash
python training/train_from_jsonl.py
```

What it does:
- reads `source_text` + `summary` pairs (and uses `category` in the prompt),
- trains automatically with checkpointing,
- resumes from the latest checkpoint on next run,
- saves the trained model to `models/sft_model_runs/run_*`,
- updates `models/active_model.txt`,
- reloads backend model automatically (if backend is running).

Stop anytime safely:
- press `Ctrl+C` during training,
- current progress is saved,
- the saved model is still activated.

Quick test run:

```bash
python training/train_from_jsonl.py --max-steps 1 --no-reload
```

## Why this is faster on CPU

- Trains only on new feedback instead of full retraining every session.
- Reuses latest tuned model (`models/dpo_model`) as the next base.
- Uses adaptive epochs for small incremental batches.

## Key files

- `training/train_incremental.py` — one-command pipeline.
- `training/prepare_feedback.py` — cumulative + incremental dataset builder.
- `training/train_dpo.py` — incremental DPO trainer.
- `backend/app.py` — API including `/reload_model`.
- `backend/summarizer.py` — runtime model loading and hot reload.

## Notes

- `👍 Good` feedback is logged but not used for training.
- Training data state is tracked in `data/feedback/training_manifest.json`.
- If backend is not running, training still completes and you can manually call reload later.
