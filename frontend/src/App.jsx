import SummarizeForm from "./SummarizeForm";
import OpsDashboard from "./OpsDashboard";
import "./App.css";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="app-kicker">MODEL OPS WORKSPACE</div>
        <h1>Summarization Training Console</h1>
        <p>Run inference, collect feedback, train, evaluate, and export one unified model.</p>
      </header>
      <main className="app-main">
        <section className="panel">
          <div className="panel-head">
            <h2>Inference & Feedback</h2>
            <span>Interactive playground</span>
          </div>
          <SummarizeForm />
        </section>
        <section className="panel">
          <div className="panel-head">
            <h2>Training & Evaluation</h2>
            <span>Model operations</span>
          </div>
          <OpsDashboard />
        </section>
      </main>
    </div>
  );
}

export default App;
