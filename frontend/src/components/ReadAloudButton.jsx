import React, { useState, useEffect } from 'react';

export function ReadAloudButton() {
  const [isPlaying, setIsPlaying] = useState(false);

  // Clean up speech synthesis if component unmounts while playing
  useEffect(() => {
    return () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const handleTogglePlay = () => {
    if (!('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported in this browser.');
      return;
    }

    if (isPlaying) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
      return;
    }

    // Grab text content from the conversation workspace or latest answer nodes
    const answerNodes = document.querySelectorAll('.answer-markdown, .chat-message-content, .user-message');
    if (answerNodes.length === 0) {
      alert('No text found to read aloud.');
      return;
    }

    const latestAnswer = answerNodes[answerNodes.length - 1];
    const textToRead = latestAnswer.innerText;

    if (!textToRead || !textToRead.trim()) {
      alert('The selected message is empty.');
      return;
    }

    const utterance = new SpeechSynthesisUtterance(textToRead);
    
    // Automatically select a natural English or fallback voice if available
    const voices = window.speechSynthesis.getVoices();
    const defaultVoice = voices.find(v => v.lang.includes('en')) || voices[0];
    if (defaultVoice) {
      utterance.voice = defaultVoice;
    }
    
    utterance.rate = 1.0;

    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);

    window.speechSynthesis.speak(utterance);
    setIsPlaying(true);
  };

  return (
    <div className="flex items-center w-full">
      <button
        onClick={handleTogglePlay}
        className={`flex items-center justify-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all shadow-xs cursor-pointer ${
          isPlaying 
            ? 'bg-red-500 text-white hover:bg-red-600' 
            : 'bg-[var(--accent)] text-white hover:opacity-90'
        }`}
      >
        {isPlaying ? (
          <>
            <svg className="w-4 h-4 fill-current animate-pulse" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" /></svg>
            Stop Reading
          </>
        ) : (
          <>
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/></svg>
            Read Aloud
          </>
        )}
      </button>
    </div>
  );
}