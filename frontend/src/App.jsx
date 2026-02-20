import SummarizeForm from "./SummarizeForm";
import OpsDashboard from "./OpsDashboard";

function App() {
  return (
    <div className="min-h-screen bg-neutral-700 text-neutral-200 antialiased">
      <div className="mx-auto max-w-[1440px] px-5 py-5 lg:px-8 lg:py-6">
        <header className="mb-6 flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-neutral-100">
            <svg className="h-4 w-4 text-neutral-900" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 0-6.23.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
            </svg>
          </div>
          <div>
            <h1 className="text-base font-medium text-neutral-100 leading-none">Summarizer</h1>
            <p className="text-[11px] text-neutral-500 mt-0.5">Training &amp; evaluation workspace</p>
          </div>
        </header>

        <main className="grid grid-cols-1 gap-5 lg:grid-cols-12">
          <section className="lg:col-span-4">
            <div className="rounded-xl border border-neutral-800 bg-neutral-800">
              <div className="border-b border-neutral-800 px-4 py-3">
                <h2 className="text-[13px] font-medium text-neutral-200">Playground</h2>
              </div>
              <div className="p-4">
                <SummarizeForm />
              </div>
            </div>
          </section>
          <section className="lg:col-span-8">
            <div className="rounded-xl border border-neutral-800 bg-neutral-800">
              <div className="border-b border-neutral-800 px-4 py-3">
                <h2 className="text-[13px] font-medium text-neutral-200">Operations</h2>
              </div>
              <div className="p-4">
                <OpsDashboard />
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
