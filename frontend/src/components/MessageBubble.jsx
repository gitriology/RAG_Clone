function MessageBubble({ msg }) {
  const isUser = msg.type === "user";

  return (
    <div
      className={`max-w-xl px-4 py-2 rounded-lg ${
        isUser
          ? "bg-blue-500 text-white ml-auto"
          : "bg-white text-black mr-auto shadow"
      }`}
    >
      <p>{msg.text}</p>

      {!isUser && msg.confidence !== undefined && (
        <div className="text-xs mt-2">
          <span
            className={`font-semibold ${
              msg.confidence > 0.7 ? "text-green-600" : "text-red-500"
            }`}
          >
            Confidence: {msg.confidence}
          </span>

          <div className="text-gray-500 mt-1">
            Sources: {msg.sources?.slice(0, 2).join(", ")}
          </div>
        </div>
      )}
    </div>
  );
}

export default MessageBubble;
