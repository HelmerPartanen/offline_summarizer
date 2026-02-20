import { useEffect, useMemo, useRef, useState } from "react";
import {
  getSystemStatus,
  listJobs,
  getJob,
  runJob,
  stopJob,
  reloadActiveModel,
  exportActiveModel,
  getLatestEvaluation,
  getEvaluationHistory,
  getFeedbackStats,
} from "./api";

const ACTIONS = [
  { label: "Verify Backend", type: "verify_backend", args: [] },
  { label: "Test Feedback", type: "submit_test_feedback", args: [] },
  { label: "Training Smoke Test", type: "run_training_test", args: [] },
  { label: "Fast Incremental", type: "train_incremental", args: ["--max-new-pairs", "1"] },
  { label: "JSONL Training", type: "train_from_jsonl", args: [] },
  { label: "Evaluate Model", type: "evaluate_model", args: ["--max-samples", "20"] },
];

function StatusBadge({ status }) {
  const styles = {
    running:   "bg-amber-500/10 text-amber-400 border-amber-500/20",
    stopping:  "bg-amber-500/10 text-amber-400 border-amber-500/20",
    completed: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    failed:    "bg-red-500/10 text-red-400 border-red-500/20",
    stopped:   "bg-neutral-500/10 text-neutral-400 border-neutral-500/20",
  };
  return (
    <span
      className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium border ${styles[status] || styles.stopped}`}
    >
      {status}
    </span>
  );
}

const btn =
  "rounded-md border border-neutral-700 bg-transparent px-2.5 py-1 text-xs text-neutral-400 transition-colors hover:border-neutral-500 hover:text-neutral-200 disabled:opacity-40 disabled:cursor-not-allowed";
const btnPrimary =
  "rounded-md bg-neutral-100 px-2.5 py-1.5 text-xs font-medium text-neutral-900 transition-colors hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed";
const input =
  "w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-500 focus:outline-none";

export default function OpsDashboard() {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState("");
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobsError, setJobsError] = useState("");
  const [busy, setBusy] = useState(false);
  const [incrementalPairs, setIncrementalPairs] = useState("1");
  const [jsonlMaxSteps, setJsonlMaxSteps] = useState("");
  const [jsonlNoReload, setJsonlNoReload] = useState(false);
  const [cnnArticles, setCnnArticles] = useState("100");
  const [cnnEpochs, setCnnEpochs] = useState("1");
  const [modelMessage, setModelMessage] = useState("");
  const [latestEval, setLatestEval] = useState(null);
  const [evalHistory, setEvalHistory] = useState([]);
  const [feedbackStats, setFeedbackStats] = useState(null);
  const logBoxRef = useRef(null);

  const runningJobIds = useMemo(
    () =>
      new Set(
        jobs
          .filter((j) => j.status === "running" || j.status === "stopping")
          .map((j) => j.id)
      ),
    [jobs]
  );

  /* ── data fetchers ── */
  const refreshStatus = async () => {
    try {
      setStatus(await getSystemStatus());
      setStatusError("");
    } catch {
      setStatus(null);
      setStatusError("Backend offline");
    }
  };

  const refreshEvaluation = async () => {
    try {
      const latest = await getLatestEvaluation();
      setLatestEval(latest?.evaluation || null);
      const h = await getEvaluationHistory(6);
      setEvalHistory(h?.history || []);
    } catch {
      setLatestEval(null);
      setEvalHistory([]);
    }
  };

  const refreshJobs = async () => {
    try {
      const d = await listJobs();
      setJobs(d.jobs || []);
      setJobsError("");
    } catch {
      setJobsError("Failed to load jobs");
    }
  };

  const refreshFeedbackStats = async () => {
    try {
      const d = await getFeedbackStats();
      setFeedbackStats(d?.stats || null);
    } catch {
      setFeedbackStats(null);
    }
  };

  const refreshSelectedJob = async (id) => {
    if (!id) return;
    try {
      const d = await getJob(id);
      if (d.job) setSelectedJob(d.job);
    } catch {
      setSelectedJob(null);
    }
  };

  useEffect(() => {
    refreshStatus();
    refreshJobs();
    refreshEvaluation();
    refreshFeedbackStats();
  }, []);

  useEffect(() => {
    const iv = setInterval(() => {
      refreshStatus();
      refreshJobs();
      refreshEvaluation();
      refreshFeedbackStats();
      if (selectedJobId) refreshSelectedJob(selectedJobId);
    }, 2500);
    return () => clearInterval(iv);
  }, [selectedJobId]);

  useEffect(() => {
    if (logBoxRef.current && selectedJob)
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [selectedJob?.output, selectedJob?.status]);

  /* ── handlers ── */
  const handleRunJob = async (type, args = []) => {
    setBusy(true);
    try {
      const r = await runJob(type, args);
      if (r.error) setJobsError(r.error);
      else if (r.job_id) setSelectedJobId(r.job_id);
      await refreshJobs();
      await refreshEvaluation();
    } finally {
      setBusy(false);
    }
  };

  const handleStopSelected = async () => {
    if (!selectedJobId) return;
    setBusy(true);
    try {
      await stopJob(selectedJobId);
      await refreshJobs();
      await refreshSelectedJob(selectedJobId);
    } finally {
      setBusy(false);
    }
  };

  const handleReload = async () => {
    setBusy(true);
    try {
      const r = await reloadActiveModel();
      setModelMessage(r?.error ? `Reload failed: ${r.error}` : "Model reloaded.");
      await refreshStatus();
      await refreshEvaluation();
    } finally {
      setBusy(false);
    }
  };

  const handleExportModel = async () => {
    setBusy(true);
    try {
      await exportActiveModel();
      setModelMessage("Downloading model zip…");
    } catch (e) {
      setModelMessage(`Export failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const runIncrementalCustom = () => {
    const n = Number.parseInt(incrementalPairs, 10);
    const safe = Number.isNaN(n) || n < 1 ? 1 : n;
    handleRunJob("train_incremental", ["--max-new-pairs", String(safe)]);
  };

  const runJsonlCustom = () => {
    const args = [];
    const n = Number.parseInt(jsonlMaxSteps, 10);
    if (!Number.isNaN(n) && n > 0) args.push("--max-steps", String(n));
    if (jsonlNoReload) args.push("--no-reload");
    handleRunJob("train_from_jsonl", args);
  };

  const runCnnCustom = () => {
    const args = [];
    const a = Number.parseInt(cnnArticles, 10);
    if (!Number.isNaN(a) && a > 0) args.push("--max-articles", String(a));
    const e = Number.parseInt(cnnEpochs, 10);
    if (!Number.isNaN(e) && e > 0) args.push("--epochs", String(e));
    handleRunJob("train_cnn", args);
  };

  const pct = (v) => {
    if (v == null || Number.isNaN(Number(v))) return "–";
    return `${(Number(v) * 100).toFixed(1)}%`;
  };

  const score = latestEval
    ? (Number(latestEval.metrics?.token_f1 || 0) +
        Number(latestEval.metrics?.rouge_l_f1 || 0)) /
      2
    : null;

  const prevScores = useMemo(() => {
    if (!latestEval?.timestamp) return evalHistory;
    return evalHistory.filter((r) => r.timestamp !== latestEval.timestamp);
  }, [evalHistory, latestEval?.timestamp]);

  /* ── render ── */
  return (
    <div className="flex flex-col gap-5">
      {/* ▸ STATUS BAR */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-2.5">
        <div className="flex items-center gap-3 text-xs">
          {statusError ? (
            <>
              <span className="h-2 w-2 rounded-full bg-red-500" />
              <span className="text-red-400">Offline</span>
            </>
          ) : (
            <>
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              <span className="text-neutral-300">Online</span>
              <span className="h-3.5 w-px bg-neutral-800" />
              <span className="text-neutral-500">{status?.device || "–"}</span>
              <span className="h-3.5 w-px bg-neutral-800" />
              <span
                className="max-w-[220px] truncate font-mono text-neutral-500"
                title={status?.model_path}
              >
                {status?.model_path || "–"}
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleReload} disabled={busy} className={btn}>
            Reload
          </button>
          <button onClick={handleExportModel} disabled={busy} className={btn}>
            Export
          </button>
        </div>
      </div>
      {modelMessage && (
        <p className="-mt-3 text-[11px] text-amber-400">{modelMessage}</p>
      )}

      {/* ▸ ACTIONS + SCORES */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr_220px]">
        <div className="flex flex-col gap-2">
          <h3 className="text-[11px] font-medium uppercase tracking-wider text-neutral-500">
            Actions
          </h3>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {ACTIONS.map((a) => (
              <button
                key={a.label}
                onClick={() => handleRunJob(a.type, a.args)}
                disabled={busy}
                className={`text-left ${btn}`}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="text-[11px] font-medium uppercase tracking-wider text-neutral-500">
              Scores
            </h3>
            <button
              onClick={refreshEvaluation}
              disabled={busy}
              className="text-[10px] text-neutral-600 hover:text-neutral-400 transition-colors disabled:opacity-40"
            >
              refresh
            </button>
          </div>
          {!latestEval ? (
            <p className="text-xs text-neutral-600">No evaluation data</p>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-3">
                <div className="text-[10px] uppercase tracking-wider text-neutral-600">
                  Model Score
                </div>
                <div className="mt-0.5 text-xl font-semibold tabular-nums text-neutral-100">
                  {pct(score)}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-md border border-neutral-800 bg-neutral-900 px-2.5 py-1.5">
                  <div className="text-[10px] text-neutral-600">F1</div>
                  <div className="text-sm tabular-nums text-neutral-300">
                    {pct(latestEval.metrics?.token_f1)}
                  </div>
                </div>
                <div className="rounded-md border border-neutral-800 bg-neutral-900 px-2.5 py-1.5">
                  <div className="text-[10px] text-neutral-600">ROUGE-L</div>
                  <div className="text-sm tabular-nums text-neutral-300">
                    {pct(latestEval.metrics?.rouge_l_f1)}
                  </div>
                </div>
              </div>
              <p className="text-[10px] text-neutral-700">
                {latestEval.metrics?.samples} samples · {latestEval.timestamp}
              </p>
            </div>
          )}
          {prevScores.length > 0 && (
            <div className="flex flex-col gap-0.5 mt-1">
              {prevScores.map((r, i) => (
                <div
                  key={`${r.timestamp}-${i}`}
                  className="flex items-center justify-between py-1 text-[10px] border-b border-neutral-800/50 last:border-0"
                >
                  <span className="text-neutral-600">
                    {r.timestamp?.split(" ")[0]}
                  </span>
                  <span className="tabular-nums text-neutral-400">
                    {pct(
                      (Number(r.metrics?.token_f1 || 0) +
                        Number(r.metrics?.rouge_l_f1 || 0)) /
                        2
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ▸ TRAINING */}
      <div className="flex flex-col gap-3 border-t border-neutral-800 pt-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-[11px] font-medium uppercase tracking-wider text-neutral-500">
            Training
          </h3>
          <span className="text-[11px] text-neutral-600">
            {feedbackStats?.total_feedback_rows ?? "–"} rows ·{" "}
            {feedbackStats?.unique_pairs ?? "–"} pairs ·{" "}
            {feedbackStats?.new_pairs ?? "–"} new
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div className="flex flex-col gap-2 rounded-lg border border-neutral-800 p-3">
            <label className="text-[11px] text-neutral-500">
              Max new pairs
            </label>
            <input
              value={incrementalPairs}
              onChange={(e) => setIncrementalPairs(e.target.value)}
              type="number"
              min="1"
              className={input}
            />
            <button
              onClick={runIncrementalCustom}
              disabled={busy}
              className={btnPrimary}
            >
              Run Incremental
            </button>
          </div>

          <div className="flex flex-col gap-2 rounded-lg border border-neutral-800 p-3">
            <label className="text-[11px] text-neutral-500">Max steps</label>
            <input
              value={jsonlMaxSteps}
              onChange={(e) => setJsonlMaxSteps(e.target.value)}
              type="number"
              min="1"
              placeholder="all"
              className={input}
            />
            <label className="flex items-center gap-2 text-[11px] text-neutral-500">
              <input
                type="checkbox"
                checked={jsonlNoReload}
                onChange={(e) => setJsonlNoReload(e.target.checked)}
                className="accent-neutral-400"
              />
              Skip reload
            </label>
            <button
              onClick={runJsonlCustom}
              disabled={busy}
              className={`mt-auto ${btnPrimary}`}
            >
              Run JSONL
            </button>
          </div>

          <div className="flex flex-col gap-2 rounded-lg border border-neutral-800 p-3">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] text-neutral-500">Articles</label>
                <input
                  value={cnnArticles}
                  onChange={(e) => setCnnArticles(e.target.value)}
                  type="number"
                  min="1"
                  className={input}
                />
              </div>
              <div>
                <label className="text-[11px] text-neutral-500">Epochs</label>
                <input
                  value={cnnEpochs}
                  onChange={(e) => setCnnEpochs(e.target.value)}
                  type="number"
                  min="1"
                  className={input}
                />
              </div>
            </div>
            <button
              onClick={runCnnCustom}
              disabled={busy}
              className={`mt-auto ${btnPrimary}`}
            >
              Run CNN
            </button>
          </div>
        </div>
      </div>

      {/* ▸ JOBS */}
      <div className="flex flex-col gap-3 border-t border-neutral-800 pt-4">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] font-medium uppercase tracking-wider text-neutral-500">
            Jobs
          </h3>
          <button onClick={refreshJobs} disabled={busy} className={btn}>
            Refresh
          </button>
        </div>
        {jobsError && <p className="text-xs text-red-400">{jobsError}</p>}

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[200px_1fr]">
          {/* job list */}
          <div className="flex flex-col gap-0.5 max-h-[340px] overflow-y-auto">
            {jobs.length === 0 && (
              <p className="text-xs text-neutral-600 py-2">No jobs</p>
            )}
            {jobs.map((job) => (
              <button
                key={job.id}
                className={`flex items-center justify-between rounded-md px-2.5 py-2 text-left transition-colors ${
                  selectedJobId === job.id
                    ? "bg-neutral-800"
                    : "hover:bg-neutral-800/50"
                }`}
                onClick={() => {
                  setSelectedJobId(job.id);
                  refreshSelectedJob(job.id);
                }}
              >
                <div className="flex flex-col min-w-0">
                  <span className="text-xs text-neutral-300 truncate">
                    {job.type}
                  </span>
                  <span className="text-[10px] font-mono text-neutral-600">
                    {job.id.slice(0, 8)}
                  </span>
                </div>
                <StatusBadge status={job.status} />
              </button>
            ))}
          </div>

          {/* job output */}
          {selectedJob ? (
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {runningJobIds.has(selectedJob.id) && (
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-500" />
                    </span>
                  )}
                  <span className="text-xs font-mono text-neutral-500">
                    {selectedJob.id.slice(0, 12)}
                  </span>
                  <StatusBadge status={selectedJob.status} />
                </div>
                <button
                  onClick={handleStopSelected}
                  disabled={!runningJobIds.has(selectedJob.id) || busy}
                  className={btn}
                >
                  Stop
                </button>
              </div>
              <div
                ref={logBoxRef}
                className="h-[280px] overflow-y-auto rounded-lg border border-neutral-800 bg-neutral-900 p-3 font-mono text-[11px] leading-5 text-neutral-400"
              >
                {(selectedJob.output || []).length === 0 ? (
                  <span className="text-neutral-700">
                    Waiting for output…
                  </span>
                ) : (
                  (selectedJob.output || []).map((line, i) => (
                    <div
                      key={`${i}-${line.slice(0, 16)}`}
                      className="flex gap-2 rounded px-1 hover:bg-neutral-900"
                    >
                      <span className="w-5 shrink-0 select-none text-right text-neutral-700">
                        {i + 1}
                      </span>
                      <span className="whitespace-pre-wrap break-all">
                        {line}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : (
            <div className="flex h-[280px] items-center justify-center rounded-lg border border-dashed border-neutral-800 text-xs text-neutral-600">
              Select a job to view output
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
