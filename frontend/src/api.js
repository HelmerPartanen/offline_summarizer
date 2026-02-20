const API_BASE = "http://localhost:8000";


export async function getSummary(text) {
    try {
        const response = await fetch(`${API_BASE}/summarize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }
        const data = await response.json();
        return data.summary;
    } catch (error) {
        console.error("Failed to fetch summary:", error);
        return "Error fetching summary. Please ensure backend is running.";
    }
}

export async function sendFeedback(text, rejected_summary, chosen_summary) {
    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, rejected_summary, chosen_summary })
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to send feedback:", error);
        return { error: "Failed to send feedback" };
    }
}


export async function getSystemStatus() {
    const response = await fetch(`${API_BASE}/status`);
    if (!response.ok) {
        throw new Error("Failed to fetch status");
    }
    return response.json();
}


export async function reloadActiveModel(modelPath = null) {
    const response = await fetch(`${API_BASE}/reload_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath })
    });
    return response.json();
}


export async function runJob(jobType, args = []) {
    const response = await fetch(`${API_BASE}/jobs/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_type: jobType, args })
    });
    return response.json();
}


export async function listJobs() {
    const response = await fetch(`${API_BASE}/jobs`);
    if (!response.ok) {
        throw new Error("Failed to fetch jobs");
    }
    return response.json();
}


export async function getJob(jobId) {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!response.ok) {
        throw new Error("Failed to fetch job details");
    }
    return response.json();
}


export async function stopJob(jobId) {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/stop`, {
        method: "POST"
    });
    return response.json();
}


export async function exportActiveModel() {
    const response = await fetch(`${API_BASE}/model/export`);
    const contentType = response.headers.get("content-type") || "";

    if (!response.ok || contentType.includes("application/json")) {
        const fallback = await response.json().catch(() => ({ error: "Export failed" }));
        throw new Error(fallback.error || "Export failed");
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `offline_summarizer_model_${new Date().toISOString().replace(/[:.]/g, "-")}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}


export async function getLatestEvaluation() {
    const response = await fetch(`${API_BASE}/evaluation/latest`);
    if (!response.ok) {
        throw new Error("Failed to fetch latest evaluation");
    }
    return response.json();
}


export async function getEvaluationHistory(limit = 10) {
    const response = await fetch(`${API_BASE}/evaluation/history?limit=${limit}`);
    if (!response.ok) {
        throw new Error("Failed to fetch evaluation history");
    }
    return response.json();
}


export async function getFeedbackStats() {
    const response = await fetch(`${API_BASE}/feedback/stats`);
    if (!response.ok) {
        throw new Error("Failed to fetch feedback stats");
    }
    return response.json();
}
