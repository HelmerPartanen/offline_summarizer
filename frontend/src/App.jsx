import SummarizeForm from "./SummarizeForm";
import OpsDashboard from "./OpsDashboard";
import "./App.css";

function App() {
  return (
    <div className="mx-auto min-h-screen w-full max-w-[1500px] px-4 py-5 text-[var(--text-primary)] lg:px-6">
      <header className="mb-4 rounded-xl border border-[var(--border-color)] bg-[var(--panel-bg)] px-4 py-4 shadow-[0_10px_30px_rgba(0,0,0,0.28)] lg:px-5">
        <div className="mb-1 text-[0.68rem] uppercase tracking-[0.14em] text-[#9f9788]">MODEL OPS WORKSPACE</div>
        <h1 className="font-serif text-2xl tracking-tight text-[#f3f1eb] lg:text-4xl">Summarization Training Console</h1>
        <p className="mt-1 text-sm text-[var(--text-secondary)] lg:text-[0.95rem]">
          Run inference, collect feedback, train, evaluate, and export one unified model.
        </p>
      </header>
      <main className="grid w-full grid-cols-1 gap-4 xl:grid-cols-[1.08fr_1.52fr]">
        <section className="min-w-0 rounded-xl border border-[var(--border-color)] bg-[var(--panel-bg)] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.28)]">
          <div className="mb-3 flex items-baseline justify-between border-b border-[var(--border-color)] pb-2">
            <h2 className="m-0 font-serif text-lg">Inference & Feedback</h2>
            <span className="text-[0.72rem] uppercase tracking-[0.09em] text-[#9f9788]">Interactive playground</span>
          </div>
          <SummarizeForm />
        </section>
        <section className="min-w-0 rounded-xl border border-[var(--border-color)] bg-[var(--panel-bg)] p-4 shadow-[0_10px_30px_rgba(0,0,0,0.28)]">
          <div className="mb-3 flex items-baseline justify-between border-b border-[var(--border-color)] pb-2">
            <h2 className="m-0 font-serif text-lg">Training & Evaluation</h2>
            <span className="text-[0.72rem] uppercase tracking-[0.09em] text-[#9f9788]">Model operations</span>
          </div>
          <OpsDashboard />
        </section>
      </main>
    </div>
  );
}

export default App;
