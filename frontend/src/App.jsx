import { useState, useEffect, useRef } from "react";
import ChatWindow from "./components/ChatWindow";
import InputBox from "./components/InputBox";
import Header from "./components/Header";
import { sendQuery } from "./services/api";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const chatEndRef = useRef(null);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { type: "user", text: input };
    const loadingMsg = { type: "bot", text: "Typing..." };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);

    try {
      const queryText = input.trim();
      const data = await sendQuery(queryText, {
        topKDocuments: 5,
        maxSentences: 3,
      });

      // Backend is the single source of truth. Never recompute, truncate,
      // clamp, or rescore the returned answer or confidence.
      const botMsg = {
        type: "bot",
        text: String(data.answer ?? ""),
        confidence: Number(data.confidence ?? 0),
        sources: Array.isArray(data.sources) ? data.sources : [],
        pipelineVersion: data.pipeline_version,
        confidenceCalibration: data.meta?.confidence_calibration ?? {},
        answerValid: Boolean(data.meta?.answer_valid),
      };

      setMessages((prev) => {
        const updated = [...prev];
        updated.pop(); // remove typing
        updated.push(botMsg);
        return updated;
      });
    } catch {
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "Answer not present in knowledge base." },
      ]);
    }

    setInput("");
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <Header />
      <ChatWindow messages={messages} chatEndRef={chatEndRef} />
      <InputBox input={input} setInput={setInput} sendMessage={sendMessage} />
    </div>
  );
}

export default App;
