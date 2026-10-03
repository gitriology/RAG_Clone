import {
  Database,
  Home,
  MessageSquare,
  Plus,
  Settings2,
  Trash2,
} from "lucide-react";
import LogoMark from "./LogoMark";
import graphIcon from "../assets/branding/graphs-icon.png";
import darkLeaf from "../assets/branding/ragar-leafmark-dark.png";
import lightLeaf from "../assets/branding/ragar-leafmark-light.png";

function Sidebar({
  onNewChat,
  recentThreads = [],
  currentThreadId,
  onSelectThread,
  onDeleteThread,
  historyLoading = false,
  historyError = "",
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <LogoMark className="brand-mark" />
        <div>
          <div className="brand-wordmark">RAGar</div>
          <div className="brand-tagline">Search · Reason · Generate</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <button className="nav-item active" type="button">
          <Home size={19} />
          <span>Home</span>
        </button>

        <button className="nav-item" type="button" onClick={onNewChat}>
          <Plus size={20} />
          <span>New Query</span>
        </button>

        <button className="nav-item" type="button">
          <Database size={19} />
          <span>Knowledge Base</span>
        </button>

        <button className="nav-item" type="button">
          <img
            className="graphs-nav-icon"
            src={graphIcon}
            alt=""
            aria-hidden="true"
          />
          <span>Graphs</span>
        </button>

        <button className="nav-item" type="button">
          <Settings2 size={19} />
          <span>Settings</span>
        </button>
      </nav>

      <div className="sidebar-divider" />

      <div className="recent-heading">Recent Queries</div>

      <div className="recent-list">
        {historyLoading ? (
          <div className="recent-status">Loading conversations…</div>
        ) : historyError ? (
          <div className="recent-status recent-error">{historyError}</div>
        ) : recentThreads.length === 0 ? (
          <div className="recent-empty">No conversations yet.</div>
        ) : (
          recentThreads.map((thread) => (
            <div
              className={`recent-item-wrap ${currentThreadId === thread.id ? "selected" : ""}`}
              key={thread.id}
            >
              <button
                className="recent-item"
                type="button"
                onClick={() => onSelectThread(thread.id)}
                disabled={thread.loading}
              >
                <MessageSquare size={13} />
                <span className="recent-copy">
                  <span className="recent-title">{thread.title}</span>
                  <span className="recent-time">{thread.timeLabel}</span>
                </span>
              </button>

              <button
                className="recent-delete"
                type="button"
                title="Delete conversation"
                aria-label={`Delete ${thread.title}`}
                onClick={() => onDeleteThread(thread.id)}
                disabled={thread.loading}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))
        )}
      </div>

      <div className="sidebar-note">
        <picture className="sidebar-leaf-wrap" aria-hidden="true">
          <source srcSet={darkLeaf} media="(prefers-color-scheme: dark)" />
          <img
            className="sidebar-leaf sidebar-leaf-light"
            src={lightLeaf}
            alt=""
          />
          <img
            className="sidebar-leaf sidebar-leaf-dark"
            src={darkLeaf}
            alt=""
          />
        </picture>

        <div className="sidebar-note-copy">
          <strong>Better questions.</strong>
          <strong>Deeper context.</strong>
          <strong>Smarter answers.</strong>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
