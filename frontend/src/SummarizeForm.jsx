import { useState } from "react";
import { getSummary, sendFeedback } from "./api";
import "./SummarizeForm.css";

export default function SummarizeForm() {
    const [text, setText] = useState("");
    const [summary, setSummary] = useState("");
    const [loading, setLoading] = useState(false);
    const [feedbackMode, setFeedbackMode] = useState(null); // 'good' or 'bad'
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
        } catch (e) {
            setSummary("An error occurred.");
        }
        setLoading(false);
    };

    const handleFeedback = async (type) => {
        if (type === 'good') {
            await sendFeedback(text, summary, summary);
            setFeedbackSent(true);
            setFeedbackMode('good');
        } else {
            setFeedbackMode('bad');
        }
    };

    const submitCorrection = async () => {
        await sendFeedback(text, summary, correctedSummary);
        setFeedbackSent(true);
        setFeedbackMode(null);
    };

    return (
        <div className="summarizer-container">
            <div className="input-section">
                <textarea
                    value={text}
                    onChange={e => setText(e.target.value)}
                    rows={12}
                    placeholder="Paste your text here to summarize..."
                    disabled={loading}
                    className="text-input"
                />
                <div className="actions">
                    <button onClick={handleSummarize} disabled={loading || !text.trim()} className="summarize-btn">
                        {loading ? "Summarizing..." : "Summarize Text"}
                    </button>
                </div>
            </div>

            {summary && (
                <div className="output-section">
                    <h3>Summary Result</h3>
                    <div className="summary-content">
                        <p>{summary}</p>
                    </div>

                    {!feedbackSent && !feedbackMode && (
                        <div className="feedback-prompt">
                            <span>How was the summary?</span>
                            <div className="feedback-buttons">
                                <button className="btn-good" onClick={() => handleFeedback('good')}>👍 Good</button>
                                <button className="btn-bad" onClick={() => handleFeedback('bad')}>👎 Bad</button>
                            </div>
                        </div>
                    )}

                    {feedbackMode === 'bad' && !feedbackSent && (
                        <div className="correction-section">
                            <h4>Improve this summary:</h4>
                            <textarea
                                value={correctedSummary}
                                onChange={e => setCorrectedSummary(e.target.value)}
                                rows={6}
                                className="correction-input"
                            />
                            <button className="submit-feedback-btn" onClick={submitCorrection}>
                                Submit Improved Summary
                            </button>
                        </div>
                    )}

                    {feedbackSent && (
                        <div className="feedback-thanks">
                            🎉 Thank you! This helps the AI learn.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
