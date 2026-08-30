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
      const data = await sendQuery(input);

      // The backend is the source of truth. Do not truncate or
      // otherwise rewrite the answer returned by the RAG pipeline.
      const botMsg = {
        type: "bot",
        text: data.answer ?? "",
        confidence:
          typeof data.confidence === "number"
            ? data.confidence
            : Number(data.confidence ?? 0),
        sources: data.sources ?? [],
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
