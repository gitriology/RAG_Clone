import { useEffect, useMemo, useState } from "react";
import { ChevronDown, PanelRight } from "lucide-react";
import MessageBubble from "./MessageBubble";
import RightPanel from "./RightPanel";

function toMillis(value) {
  if (!value) return 0;
  if (typeof value === "number") return value;
  if (value instanceof Date) return value.getTime();
  if (typeof value.toMillis === "function") return value.toMillis();
  if (typeof value.getTime === "function") return value.getTime();
  return 0;
}

function getQueryTimestamp(messages, assistantIndex) {
  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.type === "user") {
      return toMillis(messages[index].createdAt);
    }
  }
  return toMillis(messages[assistantIndex]?.createdAt);
}

function DetailsEmptyState() {
  return (
    <div className="details-empty-state">
      <div className="details-empty-icon">
        <PanelRight size={18} />
      </div>
      <h2>Retrieval details</h2>
      <p>
        Select <strong>Details</strong> on any answer to inspect its retrieval,
        evidence graph, and state vector.
      </p>
    </div>
  );
}

function ChatWindow({
  messages,
  chatEndRef,
  onRegenerate,
  regeneratingMessageId,
}) {
  // A Set lets multiple responses stay open at the same time.
  const [openDetailsIds, setOpenDetailsIds] = useState(() => new Set());

  const assistantMessages = useMemo(
    () =>
      messages.filter((message) => message.type === "bot" && !message.loading),
    [messages],
  );

  const assistantWithOrder = useMemo(
    () =>
      assistantMessages.map((message) => {
        const messageIndex = messages.findIndex(
          (item) => item.id === message.id,
        );
        return {
          message,
          queryTimestamp: getQueryTimestamp(messages, messageIndex),
        };
      }),
    [assistantMessages, messages],
  );

  // Details are intentionally closed whenever a different conversation/new
  // answer set is loaded. New answers never open automatically.
  useEffect(() => {
    setOpenDetailsIds(new Set());
  }, [messages.length, assistantMessages[0]?.id]);

  const toggleDetails = (messageId) => {
    setOpenDetailsIds((current) => {
      const next = new Set(current);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  const openDetails = useMemo(
    () =>
      assistantWithOrder
        .filter(({ message }) => openDetailsIds.has(message.id))
        .sort((a, b) => a.queryTimestamp - b.queryTimestamp),
    [assistantWithOrder, openDetailsIds],
  );

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
          messages.map((msg, index) => {
            const sourceQuery =
              msg.type === "bot"
                ? [...messages]
                    .slice(0, index)
                    .reverse()
                    .find((item) => item.type === "user")?.text || ""
                : "";

            const detailsOpen =
              msg.type === "bot" && openDetailsIds.has(msg.id);

            return (
              <MessageBubble
                key={msg.id || index}
                msg={msg}
                sourceQuery={sourceQuery}
                detailsOpen={detailsOpen}
                onToggleDetails={toggleDetails}
                onRegenerate={onRegenerate}
                regenerating={regeneratingMessageId === msg.id}
              />
            );
          })
        )}
        <div ref={chatEndRef} />
      </main>

      <aside
        className={`right-column ${
          openDetails.length ? "has-open-details" : "details-closed"
        }`}
        aria-label="Retrieval details"
      >
        {openDetails.length === 0 ? (
          <DetailsEmptyState />
        ) : (
          <div className="desktop-details-stack">
            {openDetails.map(({ message, queryTimestamp }) => (
              <section
                className="desktop-detail-card"
                key={message.id}
                data-query-timestamp={queryTimestamp}
              >
                <div className="desktop-detail-heading">
                  <div>
                    <span>Response details</span>
                    <strong>{message.title || "Research Answer"}</strong>
                  </div>
                  <button
                    type="button"
                    className="desktop-detail-close"
                    onClick={() => toggleDetails(message.id)}
                    aria-label="Close response details"
                    title="Close details"
                  >
                    <ChevronDown size={16} />
                  </button>
                </div>
                <RightPanel answer={message} />
              </section>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}

export default ChatWindow;
