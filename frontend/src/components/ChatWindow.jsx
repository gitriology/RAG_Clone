import MessageBubble from "./MessageBubble";

function ChatWindow({ messages, chatEndRef }) {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg, index) => (
        <MessageBubble key={index} msg={msg} />
      ))}
      <div ref={chatEndRef}></div>
    </div>
  );
}

export default ChatWindow;
