import { Check, Copy, RotateCcw, Sparkles } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import RightPanel from "./RightPanel";

function MessageBubble({
  msg,
  sourceQuery = "",
  detailsOpen = false,
  onToggleDetails,
  onRegenerate,
  regenerating = false,
}) {
  const [copied, setCopied] = useState(false);
  const isUser = msg.type === "user";

  if (isUser) {
    return (
      <div className="message-row user-row">
        <div className="user-message">{msg.text}</div>
      </div>
    );
  }

  if (msg.loading) {
    return (
      <div className="assistant-loading">
        <Sparkles size={17} />
        <span />
        <span />
        <span />
      </div>
    );
  }

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(msg.text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard access can be unavailable in some browser contexts.
    }
  };

  const handleDetails = () => {
    onToggleDetails?.(msg.id);
  };

  return (
    <article className="answer-card">
      <div className="answer-header">
        <div className="answer-label">
          <Sparkles size={17} />
          <span>Final Answer</span>
          <span className="enhanced-badge">
            <span />
            RAG Enhanced
          </span>
        </div>
      </div>

      <div className="answer-content">
        <h1>{msg.title || "Research Answer"}</h1>

        <div className="answer-markdown">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
          >
            {msg.text || ""}
          </ReactMarkdown>
        </div>
      </div>

      <div className="answer-actions">
        <button type="button" onClick={copyAnswer} disabled={regenerating}>
          {copied ? <Check size={15} /> : <Copy size={15} />}
          {copied ? "Copied" : "Copy"}
        </button>

        <button
          type="button"
          onClick={() => onRegenerate?.(msg, sourceQuery)}
          disabled={regenerating || !sourceQuery}
        >
          <RotateCcw size={15} />
          {regenerating ? "Regenerating…" : "Regenerate"}
        </button>

        <button
          type="button"
          className={`details-action ${detailsOpen ? "selected" : ""}`}
          onClick={handleDetails}
          disabled={regenerating}
          aria-expanded={detailsOpen}
        >
          <Sparkles size={15} />
          Details {detailsOpen ? "−" : "+"}
        </button>
      </div>

      <div
        className={`message-details-mobile ${detailsOpen ? "open" : ""}`}
        aria-hidden={!detailsOpen}
      >
        {detailsOpen && <RightPanel answer={msg} />}
      </div>

      {msg.confidence !== undefined && (
        <div className="answer-meta-mobile">
          <span>Confidence {(Number(msg.confidence) * 100).toFixed(0)}%</span>
          <span>{msg.sources?.length || 0} sources</span>
        </div>
      )}
    </article>
  );
}

export default MessageBubble;
