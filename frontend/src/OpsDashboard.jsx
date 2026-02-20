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
} from "./api";
import "./OpsDashboard.css";

const quickActions = [
  { label: "Verify Backend", type: "verify_backend", args: [] },
  { label: "Submit Test Feedback", type: "submit_test_feedback", args: [] },
  { label: "Run Training Smoke Test", type: "run_training_test", args: [] },
  { label: "Fast Incremental Training", type: "train_incremental", args: ["--max-new-pairs", "1"] },
  { label: "JSONL Supervised Training", type: "train_from_jsonl", args: [] },
  { label: "Evaluate Active Model", type: "evaluate_model", args: ["--max-samples", "20"] },
];

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
  const logBoxRef = useRef(null);

  const runningJobIds = useMemo(
    () => new Set(jobs.filter((job) => job.status === "running" || job.status === "stopping").map((job) => job.id)),
    [jobs]
  );

  const refreshStatus = async () => {
    try {
      const current = await getSystemStatus();
      setStatus(current);
      setStatusError("");
    } catch {
      setStatus(null);
      setStatusError("Backend is offline or unreachable.");
    }
  };

  const refreshEvaluation = async () => {
    try {
      const latest = await getLatestEvaluation();
      setLatestEval(latest?.evaluation || null);
      const history = await getEvaluationHistory(6);
      setEvalHistory(history?.history || []);
    } catch {
      setLatestEval(null);
      setEvalHistory([]);
    }
  };

  const refreshJobs = async () => {
    try {
      const data = await listJobs();
      setJobs(data.jobs || []);
      setJobsError("");
    } catch {
      setJobsError("Failed to load jobs.");
    }
  };

  const refreshSelectedJob = async (jobId) => {
    if (!jobId) return;
    try {
      const data = await getJob(jobId);
      if (data.job) {
        setSelectedJob(data.job);
      }
    } catch {
      setSelectedJob(null);
    }
  };

  useEffect(() => {
    refreshStatus();
    refreshJobs();
    refreshEvaluation();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      refreshStatus();
      refreshJobs();
      refreshEvaluation();
      if (selectedJobId) {
        refreshSelectedJob(selectedJobId);
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [selectedJobId]);

  useEffect(() => {
    if (!logBoxRef.current || !selectedJob) return;
    logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [selectedJob?.output, selectedJob?.status]);

  const handleRunJob = async (jobType, args = []) => {
    setBusy(true);
    try {
      const result = await runJob(jobType, args);
      if (result.error) {
        setJobsError(result.error);
      } else if (result.job_id) {
        setSelectedJobId(result.job_id);
      }
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
      const result = await reloadActiveModel();
      if (result?.error) {
        setModelMessage(`Reload failed: ${result.error}`);
      } else {
        setModelMessage("Model reloaded successfully.");
      }
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
      setModelMessage("Export started. Your browser is downloading the full active model zip.");
    } catch (error) {
      setModelMessage(`Export failed: ${error.message}`);
    } finally {
      setBusy(false);
    }
  };

  const runIncrementalCustom = () => {
    const parsed = Number.parseInt(incrementalPairs, 10);
    const safePairs = Number.isNaN(parsed) || parsed < 1 ? 1 : parsed;
    handleRunJob("train_incremental", ["--max-new-pairs", String(safePairs)]);
  };

  const runJsonlCustom = () => {
    const args = [];
    const parsed = Number.parseInt(jsonlMaxSteps, 10);
    if (!Number.isNaN(parsed) && parsed > 0) {
      args.push("--max-steps", String(parsed));
    }
    if (jsonlNoReload) {
      args.push("--no-reload");
    }
    handleRunJob("train_from_jsonl", args);
  };

  const runCnnCustom = () => {
    const args = [];
    const parsedArticles = Number.parseInt(cnnArticles, 10);
    if (!Number.isNaN(parsedArticles) && parsedArticles > 0) {
      args.push("--max-articles", String(parsedArticles));
    }

    const parsedEpochs = Number.parseInt(cnnEpochs, 10);
    if (!Number.isNaN(parsedEpochs) && parsedEpochs > 0) {
      args.push("--epochs", String(parsedEpochs));
    }

    handleRunJob("train_cnn", args);
  };

  const toPercent = (value) => {
    if (value === undefined || value === null || Number.isNaN(Number(value))) {
      return "-";
    }
    return `${(Number(value) * 100).toFixed(2)}%`;
  };

  const currentModelScore = latestEval
    ? ((Number(latestEval.metrics?.token_f1 || 0) + Number(latestEval.metrics?.rouge_l_f1 || 0)) / 2)
    : null;

  const previousScores = useMemo(() => {
    if (!latestEval?.timestamp) {
      return evalHistory;
    }
    return evalHistory.filter((row) => row.timestamp !== latestEval.timestamp);
  }, [evalHistory, latestEval?.timestamp]);

  return (
    <div className="ops-container">
      <div className="ops-header">
        <h2>Control Center</h2>
        <p>Run tests, training jobs, and monitor backend status in one place.</p>
      </div>

      <div className="ops-grid">
        <section className="ops-card">
          <div className="card-title-row">
            <h3>Unified Model Status</h3>
            <div className="inline-actions">
              <button onClick={refreshStatus} disabled={busy}>Refresh</button>
              <button onClick={handleReload} disabled={busy}>Reload Model</button>
              <button onClick={handleExportModel} disabled={busy}>Export Full Model</button>
            </div>
          </div>
          {statusError ? (
            <div className="status-badge status-offline">{statusError}</div>
          ) : (
            <>
              <div className="status-badge status-online">Backend Online</div>
              <div className="status-note">
                All training modes (feedback DPO, JSONL SFT, CNN) continuously update one active unified model.
              </div>
              <div className="status-list">
                <div><span>Device</span><strong>{status?.device || "-"}</strong></div>
                <div><span>Loaded Model</span><strong className="mono">{status?.model_path || "-"}</strong></div>
              </div>
              {modelMessage && <div className="status-info">{modelMessage}</div>}
            </>
          )}
        </section>

        <section className="ops-card">
          <h3>Quick Actions</h3>
          <div className="action-grid">
            {quickActions.map((action) => (
              <button
                key={action.label}
                onClick={() => handleRunJob(action.type, action.args)}
                disabled={busy}
                className="quick-btn"
              >
                {action.label}
              </button>
            ))}
          </div>
        </section>

        <section className="ops-card">
          <h3>Custom Training</h3>
          <div className="form-row">
            <label>Incremental max new pairs</label>
            <input value={incrementalPairs} onChange={(e) => setIncrementalPairs(e.target.value)} type="number" min="1" />
            <button onClick={runIncrementalCustom} disabled={busy}>Run Incremental</button>
          </div>
          <div className="form-row">
            <label>JSONL max steps (optional)</label>
            <input value={jsonlMaxSteps} onChange={(e) => setJsonlMaxSteps(e.target.value)} type="number" min="1" placeholder="empty = full run" />
            <label className="checkbox-row">
              <input type="checkbox" checked={jsonlNoReload} onChange={(e) => setJsonlNoReload(e.target.checked)} />
              Skip reload
            </label>
            <button onClick={runJsonlCustom} disabled={busy}>Run JSONL Training</button>
          </div>
          <div className="form-row">
            <label>CNN/DailyMail article count</label>
            <input value={cnnArticles} onChange={(e) => setCnnArticles(e.target.value)} type="number" min="1" placeholder="e.g. 500" />
            <label>CNN epochs</label>
            <input value={cnnEpochs} onChange={(e) => setCnnEpochs(e.target.value)} type="number" min="1" />
            <button onClick={runCnnCustom} disabled={busy}>Run CNN Training</button>
          </div>
        </section>
      </div>

      <section className="ops-card job-section">
        <div className="card-title-row">
          <h3>Job Monitor</h3>
          <button onClick={refreshJobs} disabled={busy}>Refresh Jobs</button>
        </div>
        {jobsError && <div className="error-text">{jobsError}</div>}

        <div className="job-list">
          {jobs.length === 0 && <div className="muted">No jobs yet.</div>}
          {jobs.map((job) => (
            <button
              key={job.id}
              className={`job-item ${selectedJobId === job.id ? "selected" : ""}`}
              onClick={() => {
                setSelectedJobId(job.id);
                refreshSelectedJob(job.id);
              }}
            >
              <div className="job-item-top">
                <strong>{job.type}</strong>
                <span className={`pill ${job.status}`}>{job.status}</span>
              </div>
              <div className="job-item-meta">
                <span>{job.id.slice(0, 8)}</span>
                <span>{job.exit_code !== null ? `exit ${job.exit_code}` : "running"}</span>
              </div>
            </button>
          ))}
        </div>

        {selectedJob && (
          <div className="job-detail">
            <div className="card-title-row">
              <h4>Selected Job</h4>
              <div className="inline-actions">
                <span className="live-indicator">
                  {runningJobIds.has(selectedJob.id) ? "● Live output" : "● Static output"}
                </span>
                <button
                  onClick={handleStopSelected}
                  disabled={!runningJobIds.has(selectedJob.id) || busy}
                >
                  Stop Job
                </button>
              </div>
            </div>
            <div className="job-meta-grid">
              <div><span>ID</span><strong className="mono">{selectedJob.id}</strong></div>
              <div><span>Status</span><strong>{selectedJob.status}</strong></div>
              <div><span>Started</span><strong>{selectedJob.started_at || "-"}</strong></div>
              <div><span>Ended</span><strong>{selectedJob.ended_at || "-"}</strong></div>
            </div>
            <div className="log-box" ref={logBoxRef}>
              {(selectedJob.output || []).length === 0 ? (
                <div className="log-empty">No output yet.</div>
              ) : (
                <div className="log-lines">
                  {(selectedJob.output || []).map((line, index) => (
                    <div className="log-line" key={`${index}-${line.slice(0, 20)}`}>
                      <span className="log-number">{index + 1}</span>
                      <span className="log-text">{line}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      <section className="ops-card job-section">
        <div className="card-title-row">
          <h3>Quality Scores</h3>
          <button onClick={refreshEvaluation} disabled={busy}>Refresh Scores</button>
        </div>

        {!latestEval ? (
          <div className="muted">No evaluation results yet. Run "Evaluate Active Model".</div>
        ) : (
          <div className="score-grid">
            <div><span>Current Model Score</span><strong>{toPercent(currentModelScore)}</strong></div>
            <div><span>Token F1</span><strong>{toPercent(latestEval.metrics?.token_f1)}</strong></div>
            <div><span>ROUGE-L F1</span><strong>{toPercent(latestEval.metrics?.rouge_l_f1)}</strong></div>
            <div><span>Samples</span><strong>{latestEval.metrics?.samples}</strong></div>
            <div><span>Timestamp</span><strong>{latestEval.timestamp}</strong></div>
          </div>
        )}

        {previousScores.length > 0 && (
          <div className="score-history">
            <div className="muted">Previous scores</div>
            {previousScores.map((row, idx) => (
              <div className="score-row" key={`${row.timestamp}-${idx}`}>
                <span>{row.timestamp}</span>
                <span>F1 {toPercent(row.metrics?.token_f1)}</span>
                <span>ROUGE-L {toPercent(row.metrics?.rouge_l_f1)}</span>
                <span>{row.metrics?.samples} samples</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
