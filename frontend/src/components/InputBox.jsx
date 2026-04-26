function InputBox({ input, setInput, sendMessage }) {
  return (
    <div className="p-4 bg-white flex gap-2 border-t">
      <input
        className="flex-1 border rounded-lg px-3 py-2"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask anything..."
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
      />

      <button
        onClick={sendMessage}
        className="bg-blue-600 text-white px-4 py-2 rounded-lg"
      >
        Send
      </button>
    </div>
  );
}

export default InputBox;
