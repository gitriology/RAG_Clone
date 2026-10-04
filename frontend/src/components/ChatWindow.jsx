import MessageBubble from "./MessageBubble";
import RightPanel from "./RightPanel";
import { useState } from "react";

function ChatWindow({ messages, chatEndRef }) {
  const [mobileRetrievalOpen, setMobileRetrievalOpen] = useState(false);
  const assistantMessages = messages.filter(
    (message) => message.type === "bot" && !message.loading,
  );
  const latestAnswer = assistantMessages.at(-1);

  return (
    <div className="workspace-grid">
      <main className="chat-column">
        {messages.length === 0 ? (
          <div className="welcome-state">
            <div className="welcome-mark">✦</div>
            <h1>Ask deeper questions.</h1>
            <p>
              Search your knowledge base, inspect retrieved evidence, and follow
              how the answer was formed.
            </p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <MessageBubble key={msg.id || index} msg={msg} />
          ))
        )}
        <div ref={chatEndRef} />
      </main>
      <div className="right-column">
        <div className="mobile-retrieval-shell">
          <button
            type="button"
            className="mobile-retrieval-toggle"
            onClick={() => setMobileRetrievalOpen((open) => !open)}
            aria-expanded={mobileRetrievalOpen}
          >
            <span>Retrieval Details</span>
            <span className="mobile-retrieval-chevron" aria-hidden="true">
              {mobileRetrievalOpen ? "−" : "+"}
            </span>
          </button>
          <div
            className={`mobile-retrieval-content ${mobileRetrievalOpen ? "open" : ""}`}
          >
            <RightPanel answer={latestAnswer} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatWindow;
