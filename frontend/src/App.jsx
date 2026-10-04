import { useEffect, useMemo, useRef, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import Header from "./components/Header";
import InputBox from "./components/InputBox";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { sendQuery } from "./services/api";
import {
  addMessage,
  buildAssistantMessageData,
  createConversation,
  deleteConversation,
  getConversationMessages,
  getUserConversations,
} from "./services/chatService";
import "./App.css";

function formatRelativeTime(timestamp) {
  const milliseconds =
    timestamp?.toMillis?.() ?? timestamp?.getTime?.() ?? null;

  if (!milliseconds) return "Just now";

  const diff = Math.max(0, Date.now() - milliseconds);
  const minutes = Math.floor(diff / 60_000);
  const hours = Math.floor(diff / 3_600_000);
  const days = Math.floor(diff / 86_400_000);

  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;

  return new Date(milliseconds).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
  });
}

function makeThread(conversation) {
  return {
    id: conversation.id,
    title: conversation.title || "New Query",
    timeLabel: formatRelativeTime(
      conversation.updatedAt || conversation.createdAt,
    ),
    persistent: true,
  };
}

function mapStoredMessage(message) {
  if (message.role === "user") {
    return {
      id: message.id,
      type: "user",
      text: message.text || "",
    };
  }

  return {
    id: message.id,
    type: "bot",
    text: message.text || "",
    title: message.title || "Research Answer",
    confidence: Number(message.confidence ?? 0),
    sources: Array.isArray(message.sources) ? message.sources : [],
    pipelineVersion: message.pipelineVersion,
    answerValid: Boolean(message.answerValid),
    retrievalConfidence: Number(message.retrievalConfidence ?? 0),
    answerConfidence: Number(message.answerConfidence ?? 0),
    evidenceGraph: message.evidenceGraph || {},
    evidenceState: message.evidenceState || {},
    agreement: message.agreement,
    complexity: message.complexity,
    margin: message.margin,
    stability: message.stability,
    recommendedTopK: message.recommendedTopK,
  };
}

function ResearchApp() {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const { user, loading } = useAuth();

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [recentThreads, setRecentThreads] = useState([]);
  const [currentThreadId, setCurrentThreadId] = useState(null);

  const [historyLoading, setHistoryLoading] = useState(false);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [sending, setSending] = useState(false);

  const [theme, setTheme] = useState(
    () => localStorage.getItem("ragar-theme") || "light",
  );

  const chatEndRef = useRef(null);
  const currentThreadIdRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ragar-theme", theme);
  }, [theme]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }, [messages]);

  // Load this authenticated user's conversations from Firestore.
  useEffect(() => {
    let cancelled = false;

    const loadHistory = async () => {
      if (!user?.uid) {
        currentThreadIdRef.current = null;
        setMessages([]);
        setRecentThreads([]);
        setCurrentThreadId(null);
        setHistoryError("");
        return;
      }

      setHistoryLoading(true);
      setHistoryError("");
      setMessages([]);
      setCurrentThreadId(null);
      currentThreadIdRef.current = null;

      try {
        const conversations = await getUserConversations(user.uid);

        if (cancelled) return;
        setRecentThreads(conversations.map(makeThread));
      } catch (error) {
        if (cancelled) return;
        console.error("Failed to load conversations:", error);
        setRecentThreads([]);
        setHistoryError("Could not load saved conversations.");
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    };

    loadHistory();

    return () => {
      cancelled = true;
    };
  }, [user?.uid]);

  const newChat = () => {
    if (sending || conversationLoading) return;
    setMobileSidebarOpen(false);

    currentThreadIdRef.current = null;
    setCurrentThreadId(null);
    setMessages([]);
    setInput("");
    setHistoryError("");
  };

  const selectConversation = async (conversationId) => {
    if (conversationId === currentThreadIdRef.current) return;
    setMobileSidebarOpen(false);
    if (sending) return;

    setConversationLoading(true);
    setHistoryError("");

    try {
      const storedMessages = await getConversationMessages(conversationId);

      currentThreadIdRef.current = conversationId;
      setCurrentThreadId(conversationId);
      setMessages(storedMessages.map(mapStoredMessage));
      setInput("");
    } catch (error) {
      console.error("Failed to load conversation:", error);
      setHistoryError("Could not open that conversation.");
    } finally {
      setConversationLoading(false);
    }
  };

  const deleteThread = async (conversationId) => {
    if (sending) return;

    const thread = recentThreads.find((item) => item.id === conversationId);
    const confirmed = window.confirm(
      `Delete “${thread?.title || "this conversation"}”? This cannot be undone.`,
    );

    if (!confirmed) return;

    try {
      if (thread?.persistent !== false) {
        await deleteConversation(conversationId);
      }

      setRecentThreads((prev) =>
        prev.filter((item) => item.id !== conversationId),
      );

      if (currentThreadIdRef.current === conversationId) {
        currentThreadIdRef.current = null;
        setCurrentThreadId(null);
        setMessages([]);
        setInput("");
      }
    } catch (error) {
      console.error("Failed to delete conversation:", error);
      setHistoryError("Could not delete that conversation.");
    }
  };

  const touchThread = (conversationId, fallbackTitle, persistent = true) => {
    setRecentThreads((prev) => {
      const existing = prev.find((thread) => thread.id === conversationId);
      const updated = {
        id: conversationId,
        title: existing?.title || fallbackTitle,
        timeLabel: "Just now",
        persistent: existing?.persistent ?? persistent,
      };

      return [
        updated,
        ...prev.filter((thread) => thread.id !== conversationId),
      ];
    });
  };

  const sendMessage = async () => {
    const queryText = input.trim();

    if (!queryText || !user?.uid || sending || conversationLoading) return;

    setInput("");
    setSending(true);
    setHistoryError("");

    const requestId = `${Date.now()}-${Math.random()}`;
    let conversationId = currentThreadIdRef.current;
    let persistentConversation = true;

    try {
      // Create the Firestore conversation only for the first message in a new chat.
      if (!conversationId) {
        try {
          conversationId = await createConversation(user.uid, queryText);
        } catch (firestoreError) {
          // Authentication and RAG should remain usable even if Firestore is temporarily unavailable.
          console.error(
            "Could not create Firestore conversation:",
            firestoreError,
          );
          persistentConversation = false;
          conversationId = `local-${requestId}`;
          setHistoryError(
            "Conversation history is temporarily unavailable. This chat will not persist.",
          );
        }

        currentThreadIdRef.current = conversationId;
        setCurrentThreadId(conversationId);
        setRecentThreads((prev) => [
          {
            id: conversationId,
            title:
              queryText.length > 34 ? `${queryText.slice(0, 31)}…` : queryText,
            timeLabel: "Just now",
            persistent: persistentConversation,
          },
          ...prev,
        ]);
      }

      // Render immediately; Firestore should never block the RAG UI.
      setMessages((prev) => [
        ...prev,
        {
          id: `${requestId}-user`,
          type: "user",
          text: queryText,
        },
        {
          id: `${requestId}-loading`,
          type: "bot",
          loading: true,
        },
      ]);

      if (persistentConversation) {
        try {
          await addMessage(conversationId, {
            role: "user",
            text: queryText,
          });
        } catch (firestoreError) {
          console.error("Could not save user message:", firestoreError);
          setHistoryError(
            "The question was sent, but conversation history could not be saved.",
          );
        }
      }

      // Existing FastAPI RAG pipeline remains unchanged.
      const data = await sendQuery(queryText, {
        topKDocuments: 5,
        maxSentences: 3,
      });

      const evidenceGraph = data.evidence_graph || {};
      const evidenceState = data.evidence_state || {};

      const botMsg = {
        id: `${requestId}-assistant`,
        type: "bot",
        text: String(data.answer ?? ""),
        title: queryText.length > 74 ? `${queryText.slice(0, 71)}…` : queryText,
        confidence: Number(data.confidence ?? 0),
        sources: Array.isArray(data.sources) ? data.sources : [],
        pipelineVersion: data.pipeline_version,
        answerValid: Boolean(data.meta?.answer_valid),
        retrievalConfidence: Number(data.meta?.retrieval_confidence ?? 0),
        answerConfidence: Number(data.meta?.answer_confidence ?? 0),
        evidenceGraph: {
          node_count: evidenceGraph.node_count,
          edge_count: evidenceGraph.edge_count,
        },
        evidenceState: {
          feature_count: evidenceState.feature_count,
          evidence_score: evidenceState.evidence_score,
        },
        agreement: data.meta?.answer_agreement,
        complexity: evidenceState.complexity,
        margin: evidenceState.margin,
        stability: evidenceState.stability,
        recommendedTopK: evidenceState.recommended_top_k,
      };

      // Always show the RAG answer, even if the history write fails.
      setMessages((prev) =>
        prev.map((message) =>
          message.id === `${requestId}-loading` ? botMsg : message,
        ),
      );

      if (persistentConversation) {
        try {
          // Store only the metadata needed to reconstruct the UI.
          // The full evidence graph is intentionally not stored because Firestore
          // documents have a 1 MiB size limit.
          await addMessage(conversationId, buildAssistantMessageData(botMsg));
        } catch (firestoreError) {
          console.error("Could not save assistant message:", firestoreError);
          setHistoryError(
            "The answer is available, but the response could not be saved to history.",
          );
        }
      }

      touchThread(conversationId, queryText, persistentConversation);
    } catch (error) {
      console.error("RAG query error:", error);

      setMessages((prev) => [
        ...prev.filter((message) => message.id !== `${requestId}-loading`),
        {
          id: `${requestId}-error`,
          type: "bot",
          text:
            error?.response?.data?.detail ||
            "Answer not present in knowledge base.",
        },
      ]);

      if (persistentConversation) {
        try {
          await addMessage(conversationId, {
            role: "assistant",
            text:
              error?.response?.data?.detail ||
              "Answer not present in knowledge base.",
            title:
              queryText.length > 74 ? `${queryText.slice(0, 71)}…` : queryText,
            answerValid: false,
            confidence: 0,
            sources: [],
          });
        } catch (firestoreError) {
          console.error("Could not save error response:", firestoreError);
        }
      }

      touchThread(conversationId, queryText, persistentConversation);
    } finally {
      setSending(false);
    }
  };

  const appClass = useMemo(() => `app-shell ${theme}`, [theme]);

  if (loading) {
    return <div className="boot-screen">Loading your workspace…</div>;
  }

  if (!user) {
    return <Login />;
  }

  return (
    <div className={appClass}>
      <Sidebar
        isMobileOpen={mobileSidebarOpen}
        onNewChat={newChat}
        recentThreads={recentThreads}
        currentThreadId={currentThreadId}
        onSelectThread={selectConversation}
        onDeleteThread={deleteThread}
        historyLoading={historyLoading}
        historyError={historyError}
      />

      {mobileSidebarOpen && (
        <button
          type="button"
          className="sidebar-overlay"
          onClick={() => setMobileSidebarOpen(false)}
          aria-label="Close sidebar"
        />
      )}

      <div className="main-area">
        <Header
          onMenuClick={() => setMobileSidebarOpen((open) => !open)}
          theme={theme}
          onToggleTheme={() =>
            setTheme((value) => (value === "dark" ? "light" : "dark"))
          }
        />

        <InputBox
          input={input}
          setInput={setInput}
          sendMessage={sendMessage}
          disabled={historyLoading || conversationLoading || sending}
        />

        {conversationLoading && (
          <div className="conversation-loading">Loading conversation…</div>
        )}

        <ChatWindow messages={messages} chatEndRef={chatEndRef} />
      </div>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <ResearchApp />
    </AuthProvider>
  );
}

export default App;
