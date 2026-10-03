import { Bot, Check, Copy, RotateCcw, Share2 } from "lucide-react";
import { useState } from "react";

function MessageBubble({ msg }) {
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
        <Bot size={17} />
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

  return (
    <article className="answer-card">
      <div className="answer-header">
        <div className="answer-label">
          <Bot size={17} />
          <span>Final Answer</span>
          <span className="enhanced-badge">
            <span />
            RAG Enhanced
          </span>
        </div>
      </div>

      <div className="answer-content">
        <h1>{msg.title || "Research Answer"}</h1>
        <p className="answer-lead">{msg.text}</p>

        {msg.sections?.map((section, index) => (
          <section className="answer-section" key={`${section.title}-${index}`}>
            <div className="section-number">{index + 1}</div>
            <div>
              <h2>{section.title}</h2>
              <ul>
                {section.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            </div>
          </section>
        ))}

        {msg.summary && (
          <div className="summary-card">
            <div className="summary-icon">✦</div>
            <div>
              <h3>In Summary</h3>
              <p>{msg.summary}</p>
            </div>
          </div>
        )}
      </div>

      <div className="answer-actions">
        <button onClick={copyAnswer}>
          {copied ? <Check size={15} /> : <Copy size={15} />}{" "}
          {copied ? "Copied" : "Copy"}
        </button>
        <button>
          <RotateCcw size={15} /> Regenerate
        </button>
        <button>
          <Share2 size={15} /> Share
        </button>
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
