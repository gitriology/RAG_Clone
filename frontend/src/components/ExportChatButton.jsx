import React from 'react';

export function ExportChatButton() {
  const handleExportMarkdown = () => {
    const messageElements = document.querySelectorAll('.user-message, .answer-markdown');
    
    if (messageElements.length === 0) {
      alert('No conversation messages found to export.');
      return;
    }

    let markdownContent = `# RAG Assistant Conversation Export\n*Exported on: ${new Date().toLocaleString()}*\n\n---\n\n`;

    messageElements.forEach((el) => {
      const isUser = el.classList.contains('user-message');
      const prefix = isUser ? '### **User:**\n' : '### **Assistant:**\n';
      const text = el.innerText.trim();
      
      markdownContent += `${prefix}\n${text}\n\n---\n\n`;
    });

    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `chat-export-${Date.now()}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={handleExportMarkdown}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors shadow-2xs"
      title="Download conversation as Markdown"
    >
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      Export Chat (.md)
    </button>
  );
}