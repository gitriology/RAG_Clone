import { ArrowUp, Sparkles } from "lucide-react";

function InputBox({ input, setInput, sendMessage, disabled }) {
  const handleSend = () => {
    if (!input.trim() || disabled) return;
    sendMessage();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="query-bar-wrap">
      <div className="query-bar">
        <Sparkles size={18} className="query-sparkle" />
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="What are the latest advancements in retrieval-augmented generation?"
          onKeyDown={handleKeyDown}
          disabled={disabled}
          aria-label="Research question"
        />
        <button
          className="send-button"
          onClick={handleSend}
          disabled={disabled || !input.trim()}
          aria-label="Send query"
        >
          <ArrowUp size={21} />
        </button>
      </div>
    </div>
  );
}

export default InputBox;
