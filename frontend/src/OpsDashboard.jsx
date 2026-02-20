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

const quickActions = [
  { label: "Verify Backend", type: "verify_backend", args: [] },
  { label: "Submit Test Feedback", type: "submit_test_feedback", args: [] },
  { label: "Run Training Smoke Test", type: "run_training_test", args: [] },
  { label: "Fast Incremental Training", type: "train_incremental", args: ["--max-new-pairs", "1"] },
  { label: "JSONL Supervised Training", type: "train_from_jsonl", args: [] },
  { label: "Evaluate Active Model", type: "evaluate_model", args: ["--max-samples", "20"] },
];

const cardClass = "rounded-lg border border-[var(--border-color)] bg-[#26282d] p-3";
const buttonClass = "rounded-md border border-[var(--border-color)] bg-[#303238] px-3 py-2 text-xs font-medium text-[#ece9e2] transition hover:border-[#5b5e66] hover:bg-[#373940] disabled:cursor-not-allowed disabled:opacity-60";
const inputClass = "w-full rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2 text-sm text-[#ece9e2] outline-none focus:border-[#6a6d75]";

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

  const refreshFeedbackStats = async () => {
    try {
      const data = await getFeedbackStats();
      setFeedbackStats(data?.stats || null);
    } catch {
      setFeedbackStats(null);
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
    refreshFeedbackStats();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      refreshStatus();
      refreshJobs();
      refreshEvaluation();
      refreshFeedbackStats();
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
    <div className="space-y-3">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className="m-0 text-lg font-semibold text-[#f0eee7]">Control Center</h2>
          <p className="mt-1 text-xs text-[#a8a295]">Run tests, training jobs, and monitor backend status in one place.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 2xl:grid-cols-12">
        <section className={`${cardClass} 2xl:col-span-5`}>
          <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
            <h3 className="m-0 text-sm font-semibold text-[#f0eee7]">Unified Model Status</h3>
            <div className="flex flex-wrap gap-1.5">
              <button onClick={refreshStatus} disabled={busy} className={buttonClass}>Refresh</button>
              <button onClick={handleReload} disabled={busy} className={buttonClass}>Reload Model</button>
              <button onClick={handleExportModel} disabled={busy} className={buttonClass}>Export Full Model</button>
            </div>
          </div>

          {statusError ? (
            <div className="rounded-md border border-[rgba(211,107,105,0.36)] bg-[rgba(211,107,105,0.12)] px-2 py-1.5 text-xs font-semibold text-[#e28c8a]">
              {statusError}
            </div>
          ) : (
            <>
              <div className="inline-flex rounded-md border border-[rgba(75,184,146,0.36)] bg-[rgba(75,184,146,0.12)] px-2 py-1 text-xs font-semibold text-[#78d2b2]">
                Backend Online
              </div>
              <div className="mt-2 rounded-md border border-[#464a52] bg-[#2f3136] px-2 py-1.5 text-xs text-[#b9b2a6]">
                All training modes (feedback DPO, JSONL SFT, CNN) continuously update one active unified model.
              </div>
              <div className="mt-2 grid grid-cols-1 gap-2 lg:grid-cols-2">
                <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2">
                  <div className="text-[11px] text-[#a59f93]">Device</div>
                  <div className="text-xs font-semibold">{status?.device || "-"}</div>
                </div>
                <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2">
                  <div className="text-[11px] text-[#a59f93]">Loaded Model</div>
                  <div className="break-all font-mono text-[11px] text-[#ece9e2]">{status?.model_path || "-"}</div>
                </div>
              </div>
              {modelMessage && <div className="mt-2 text-xs text-[#d89b83]">{modelMessage}</div>}
            </>
          )}
        </section>

        <section className={`${cardClass} 2xl:col-span-3`}>
          <h3 className="m-0 mb-2 text-sm font-semibold text-[#f0eee7]">Quick Actions</h3>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 2xl:grid-cols-1">
            {quickActions.map((action) => (
              <button
                key={action.label}
                onClick={() => handleRunJob(action.type, action.args)}
                disabled={busy}
                className={`${buttonClass} text-left`}
              >
                {action.label}
              </button>
            ))}
          </div>
        </section>

        <section className={`${cardClass} 2xl:col-span-4`}>
          <h3 className="m-0 mb-2 text-sm font-semibold text-[#f0eee7]">Custom Training</h3>
          <div className="mb-2 rounded-md border border-[#464a52] bg-[#2f3136] px-2 py-1.5 text-xs text-[#b9b2a6]">
            Feedback available: <strong>{feedbackStats?.total_feedback_rows ?? "-"}</strong> rows · trainable unique pairs: <strong>{feedbackStats?.unique_pairs ?? "-"}</strong> · new pairs not yet trained: <strong>{feedbackStats?.new_pairs ?? "-"}</strong>
          </div>

          <div className="grid grid-cols-1 gap-2">
            <div className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_120px_auto] md:items-center">
              <label className="text-xs text-[#a7a194]">Incremental max new pairs</label>
              <input value={incrementalPairs} onChange={(e) => setIncrementalPairs(e.target.value)} type="number" min="1" className={inputClass} />
              <button onClick={runIncrementalCustom} disabled={busy} className={buttonClass}>Run Incremental</button>
            </div>

            <div className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_120px_auto_auto] md:items-center">
              <label className="text-xs text-[#a7a194]">JSONL max steps (optional)</label>
              <input value={jsonlMaxSteps} onChange={(e) => setJsonlMaxSteps(e.target.value)} type="number" min="1" placeholder="full run" className={inputClass} />
              <label className="flex items-center gap-2 whitespace-nowrap text-xs text-[#b7b1a5]">
                <input type="checkbox" checked={jsonlNoReload} onChange={(e) => setJsonlNoReload(e.target.checked)} />
                Skip reload
              </label>
              <button onClick={runJsonlCustom} disabled={busy} className={buttonClass}>Run JSONL Training</button>
            </div>

            <div className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_120px_minmax(0,1fr)_120px_auto] md:items-center">
              <label className="text-xs text-[#a7a194]">CNN article count</label>
              <input value={cnnArticles} onChange={(e) => setCnnArticles(e.target.value)} type="number" min="1" className={inputClass} />
              <label className="text-xs text-[#a7a194]">CNN epochs</label>
              <input value={cnnEpochs} onChange={(e) => setCnnEpochs(e.target.value)} type="number" min="1" className={inputClass} />
              <button onClick={runCnnCustom} disabled={busy} className={buttonClass}>Run CNN Training</button>
            </div>
          </div>
        </section>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1.7fr_1fr]">
        <section className={cardClass}>
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="m-0 text-sm font-semibold text-[#f0eee7]">Job Monitor</h3>
            <button onClick={refreshJobs} disabled={busy} className={buttonClass}>Refresh Jobs</button>
          </div>
          {jobsError && <div className="mb-2 text-xs text-[#eca2a0]">{jobsError}</div>}

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-[300px_minmax(0,1fr)]">
            <div className="max-h-[460px] space-y-2 overflow-auto pr-1">
              {jobs.length === 0 && <div className="text-xs text-[#a59f93]">No jobs yet.</div>}
              {jobs.map((job) => (
                <button
                  key={job.id}
                  className={`w-full rounded-md border px-2 py-2 text-left transition ${selectedJobId === job.id ? "border-[#7d5f54] bg-[#373239]" : "border-[var(--border-color)] bg-[#2f3136] hover:border-[#5a5e67]"}`}
                  onClick={() => {
                    setSelectedJobId(job.id);
                    refreshSelectedJob(job.id);
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <strong className="text-xs text-[#ece9e2]">{job.type}</strong>
                    <span className={`rounded-full border px-2 py-[2px] text-[10px] uppercase ${job.status === "running" || job.status === "stopping" ? "border-[#d29c53] text-[#f1c98f]" : ""} ${job.status === "completed" ? "border-[#4bb892] text-[#8de0c2]" : ""} ${job.status === "failed" ? "border-[#d36b69] text-[#eca2a0]" : ""} ${job.status === "stopped" ? "border-[#6b6e75] text-[#b6b0a3]" : ""}`}>
                      {job.status}
                    </span>
                  </div>
                  <div className="mt-1 flex justify-between text-[11px] text-[#a7a194]">
                    <span>{job.id.slice(0, 8)}</span>
                    <span>{job.exit_code !== null ? `exit ${job.exit_code}` : "running"}</span>
                  </div>
                </button>
              ))}
            </div>

            {selectedJob ? (
              <div className="min-w-0 border-t border-[var(--border-color)] pt-2 lg:border-l lg:border-t-0 lg:pl-3 lg:pt-0">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <h4 className="m-0 text-sm font-semibold text-[#f0eee7]">Selected Job</h4>
                  <div className="flex items-center gap-2">
                    <span className="rounded-full border border-[var(--border-color)] bg-[#2f3136] px-2 py-[3px] text-[11px] text-[#b7b1a5]">
                      {runningJobIds.has(selectedJob.id) ? "● Live output" : "● Static output"}
                    </span>
                    <button onClick={handleStopSelected} disabled={!runningJobIds.has(selectedJob.id) || busy} className={buttonClass}>Stop Job</button>
                  </div>
                </div>

                <div className="mb-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                  <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2">
                    <div className="text-[11px] text-[#a59f93]">ID</div>
                    <div className="break-all font-mono text-[11px] text-[#ece9e2]">{selectedJob.id}</div>
                  </div>
                  <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2">
                    <div className="text-[11px] text-[#a59f93]">Status</div>
                    <div className="text-xs font-semibold text-[#ece9e2]">{selectedJob.status}</div>
                  </div>
                  <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2">
                    <div className="text-[11px] text-[#a59f93]">Started</div>
                    <div className="text-xs text-[#ece9e2]">{selectedJob.started_at || "-"}</div>
                  </div>
                  <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] px-2 py-2">
                    <div className="text-[11px] text-[#a59f93]">Ended</div>
                    <div className="text-xs text-[#ece9e2]">{selectedJob.ended_at || "-"}</div>
                  </div>
                </div>

                <div ref={logBoxRef} className="max-h-[360px] min-h-[210px] overflow-auto rounded-md border border-[var(--border-color)] bg-[#202227] p-2 font-mono text-xs leading-5 text-[#d8d4cc]">
                  {(selectedJob.output || []).length === 0 ? (
                    <div className="italic text-[#aaa395]">No output yet.</div>
                  ) : (
                    <div className="space-y-[2px]">
                      {(selectedJob.output || []).map((line, index) => (
                        <div className="grid grid-cols-[44px_minmax(0,1fr)] gap-2" key={`${index}-${line.slice(0, 20)}`}>
                          <span className="select-none text-right text-[#868074]">{index + 1}</span>
                          <span className="break-words whitespace-pre-wrap text-[#dedad1]">{line}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex min-h-[220px] items-center justify-center rounded-md border border-dashed border-[var(--border-color)] text-xs text-[#a59f93]">
                Select a job to view details.
              </div>
            )}
          </div>
        </section>

        <section className={cardClass}>
          <div className="mb-2 flex items-center justify-between gap-2">
            <h3 className="m-0 text-sm font-semibold text-[#f0eee7]">Quality Scores</h3>
            <button onClick={refreshEvaluation} disabled={busy} className={buttonClass}>Refresh Scores</button>
          </div>

          {!latestEval ? (
            <div className="text-xs text-[#a59f93]">No evaluation results yet. Run "Evaluate Active Model".</div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] p-2">
                <div className="text-[11px] text-[#a59f93]">Current Model Score</div>
                <div className="text-sm font-semibold text-[#ece9e2]">{toPercent(currentModelScore)}</div>
              </div>
              <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] p-2">
                <div className="text-[11px] text-[#a59f93]">Token F1</div>
                <div className="text-sm font-semibold text-[#ece9e2]">{toPercent(latestEval.metrics?.token_f1)}</div>
              </div>
              <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] p-2">
                <div className="text-[11px] text-[#a59f93]">ROUGE-L F1</div>
                <div className="text-sm font-semibold text-[#ece9e2]">{toPercent(latestEval.metrics?.rouge_l_f1)}</div>
              </div>
              <div className="rounded-md border border-[var(--border-color)] bg-[#2f3136] p-2">
                <div className="text-[11px] text-[#a59f93]">Samples</div>
                <div className="text-sm font-semibold text-[#ece9e2]">{latestEval.metrics?.samples}</div>
              </div>
              <div className="col-span-2 rounded-md border border-[var(--border-color)] bg-[#2f3136] p-2">
                <div className="text-[11px] text-[#a59f93]">Timestamp</div>
                <div className="text-xs text-[#ece9e2]">{latestEval.timestamp}</div>
              </div>
            </div>
          )}

          {previousScores.length > 0 && (
            <div className="mt-2 space-y-1.5">
              <div className="text-xs text-[#a59f93]">Previous scores</div>
              {previousScores.map((row, idx) => (
                <div className="grid grid-cols-[1.3fr_1fr_1fr] gap-2 rounded-md border border-[var(--border-color)] bg-[#202227] px-2 py-1.5 text-[11px] text-[#d6d2ca]" key={`${row.timestamp}-${idx}`}>
                  <span>{row.timestamp}</span>
                  <span>F1 {toPercent(row.metrics?.token_f1)}</span>
                  <span>ROUGE-L {toPercent(row.metrics?.rouge_l_f1)}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
