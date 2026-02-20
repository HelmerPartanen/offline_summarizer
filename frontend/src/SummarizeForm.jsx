import { useState } from "react";
import { getSummary, sendFeedback } from "./api";

export default function SummarizeForm() {
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedbackMode, setFeedbackMode] = useState(null);
  const [correctedSummary, setCorrectedSummary] = useState("");
  const [feedbackSent, setFeedbackSent] = useState(false);

  const handleSummarize = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setSummary("");
    setFeedbackMode(null);
    setFeedbackSent(false);
    try {
      const result = await getSummary(text);
      setSummary(result);
      setCorrectedSummary(result);
    } catch {
      setSummary("An error occurred.");
    }
    setLoading(false);
  };

  const handleFeedback = async (type) => {
    if (type === "good") {
      await sendFeedback(text, summary, summary);
      setFeedbackSent(true);
      setFeedbackMode("good");
    } else {
      setFeedbackMode("bad");
    }
  };

  const submitCorrection = async () => {
    await sendFeedback(text, summary, correctedSummary);
    setFeedbackSent(true);
    setFeedbackMode(null);
  };

  return (
    <div className="flex flex-col gap-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={6}
        placeholder="Paste text to summarize…"
        disabled={loading}
        className="w-full resize-y rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2.5 text-sm text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none disabled:opacity-40"
      />

      <div className="flex justify-end">
        <button
          onClick={handleSummarize}
          disabled={loading || !text.trim()}
          className="rounded-lg bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 transition-colors hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Summarizing…" : "Summarize"}
        </button>
      </div>

      {summary && (
        <div className="flex flex-col gap-3 border-t border-neutral-800 pt-4">
          <div className="text-[11px] font-medium uppercase tracking-wider text-neutral-500">
            Result
          </div>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-3 text-sm leading-relaxed text-neutral-300">
            {summary}
          </div>

          {!feedbackSent && !feedbackMode && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-neutral-500">Rate this output</span>
              <div className="flex gap-2">
                <button
                  onClick={() => handleFeedback("good")}
                  className="rounded-md border border-neutral-700 px-3 py-1 text-xs text-neutral-400 transition-colors hover:border-neutral-500 hover:text-neutral-200"
                >
                  Good
                </button>
                <button
                  onClick={() => handleFeedback("bad")}
                  className="rounded-md border border-neutral-700 px-3 py-1 text-xs text-neutral-400 transition-colors hover:border-neutral-500 hover:text-neutral-200"
                >
                  Bad
                </button>
              </div>
            </div>
          )}

          {feedbackMode === "bad" && !feedbackSent && (
            <div className="flex flex-col gap-2">
              <textarea
                value={correctedSummary}
                onChange={(e) => setCorrectedSummary(e.target.value)}
                rows={3}
                className="w-full resize-y rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm text-neutral-200 focus:border-neutral-600 focus:outline-none"
              />
              <div className="flex justify-end">
                <button
                  onClick={submitCorrection}
                  className="rounded-md bg-neutral-100 px-3 py-1.5 text-xs font-medium text-neutral-900 hover:bg-white"
                >
                  Submit Correction
                </button>
              </div>
            </div>
          )}

          {feedbackSent && (
            <div className="flex items-center gap-1.5 text-xs text-emerald-400">
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Feedback recorded
            </div>
          )}
        </div>
      )}
    </div>
  );
}
