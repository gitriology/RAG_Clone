import {
  ChevronDown,
  FileText,
  GitBranch,
  Network,
  Sparkles,
} from "lucide-react";
import { useState } from "react";

function RightPanel({ answer }) {
  const [tab, setTab] = useState("Retrieval");
  const sources = answer?.sources || [];
  const retrievalConfidence =
    answer?.retrievalConfidence ?? answer?.confidence ?? 0;

  return (
    <aside className="right-panel">
      <div className="panel-tabs">
        {["Retrieval", "Evidence Graph", "State Vector"].map((name) => (
          <button
            key={name}
            className={tab === name ? "active" : ""}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>

      {tab === "Retrieval" && (
        <div className="panel-content">
          <div className="panel-heading-row">
            <h2>
              Retrieved Documents <span>(Top-K)</span>
            </h2>
            <span>{sources.length || 0} sources</span>
          </div>
          <div className="source-list">
            {(sources.length ? sources : ["No sources returned yet"]).map(
              (source, index) => (
                <div className="source-card" key={`${source}-${index}`}>
                  <div className="source-top">
                    <div className="source-rank">{index + 1}</div>
                    <div className="source-main">
                      <h3>{source}</h3>
                      <div className="source-meta">
                        <FileText size={12} /> Retrieved source
                      </div>
                      <p>
                        Source returned by the RAG retrieval pipeline. Open the
                        Evidence Graph or State Vector tab for additional
                        pipeline signals.
                      </p>
                    </div>
                    <span className="source-score">Retrieved</span>
                  </div>
                </div>
              ),
            )}
          </div>

          <div className="retrieval-state">
            <div className="state-title">
              <Sparkles size={16} />
              <span>Retrieval State</span>
              <ChevronDown size={16} />
            </div>
            <div className="state-grid">
              <div>
                <span>Agreement</span>
                <strong>
                  {answer?.agreement != null
                    ? Number(answer.agreement).toFixed(2)
                    : "—"}
                </strong>
              </div>
              <div>
                <span>Complexity</span>
                <strong>{answer?.complexity || "—"}</strong>
              </div>
              <div>
                <span>Margin</span>
                <strong>
                  {answer?.margin != null
                    ? Number(answer.margin).toFixed(2)
                    : "—"}
                </strong>
              </div>
              <div>
                <span>Stability</span>
                <strong>{answer?.stability || "—"}</strong>
              </div>
              <div>
                <span>Recommended Top-K</span>
                <strong>
                  {(answer?.recommendedTopK ?? sources.length) || "—"}
                </strong>
              </div>
              <div>
                <span>Retrieval Confidence</span>
                <strong>{Number(retrievalConfidence).toFixed(2)}</strong>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === "Evidence Graph" && (
        <div className="empty-panel">
          <Network size={28} />
          <h2>Evidence Graph</h2>
          <p>
            The graph returned by your backend will be surfaced here without
            changing the RAG pipeline.
          </p>
          <div className="metric-row">
            <span>Nodes</span>
            <strong>{answer?.evidenceGraph?.node_count ?? "—"}</strong>
          </div>
          <div className="metric-row">
            <span>Edges</span>
            <strong>{answer?.evidenceGraph?.edge_count ?? "—"}</strong>
          </div>
        </div>
      )}

      {tab === "State Vector" && (
        <div className="empty-panel">
          <GitBranch size={28} />
          <h2>Evidence State Vector</h2>
          <p>
            Backend evidence-state features can be visualized here as the next
            UI layer.
          </p>
          <div className="metric-row">
            <span>Features</span>
            <strong>{answer?.evidenceState?.feature_count ?? "—"}</strong>
          </div>
          <div className="metric-row">
            <span>Evidence score</span>
            <strong>
              {answer?.evidenceState?.evidence_score != null
                ? Number(answer.evidenceState.evidence_score).toFixed(2)
                : "—"}
            </strong>
          </div>
        </div>
      )}
    </aside>
  );
}

export default RightPanel;
