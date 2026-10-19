import React, { useEffect, useState } from 'react';

const ACCENT_PRESETS = {
  Sage: { light: '#4B6B57', dark: '#739E82', softLight: '#E8F0EC', softDark: '#1E2E23' },
  Ocean: { light: '#2563EB', dark: '#60A5FA', softLight: '#EFF6FF', softDark: '#1E3A8A' },
  Lavender: { light: '#7C3AED', dark: '#A78BFA', softLight: '#F5F3FF', softDark: '#2E1065' },
  Rose: { light: '#E11D48', dark: '#FB7185', softLight: '#FFF1F2', softDark: '#4C0519' },
  Amber: { light: '#D97706', dark: '#FBBF24', softLight: '#FFFBEB', softDark: '#451A03' }
};

export function SettingsToolbar({ isOpen, onClose }) {
  const [accent, setAccent] = useState('Ocean');
  const [textSize, setTextSize] = useState('M'); // S, M, L
  const [fontFamily, setFontFamily] = useState('Sans'); // Sans, Serif, Mono

  // Apply Colour Mood variables to <html>
  useEffect(() => {
    const root = document.documentElement;
    const isDark = root.classList.contains('dark') || window.matchMedia('(prefers-color-scheme: dark)').matches;
    const preset = ACCENT_PRESETS[accent];
    
    root.style.setProperty('--accent', isDark ? preset.dark : preset.light);
    root.style.setProperty('--accent-soft', isDark ? preset.softDark : preset.softLight);
  }, [accent]);

  // Apply Reading Comfort attributes and inject global rules targeting answers and markdown
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-text-size', textSize);
    root.setAttribute('data-font-family', fontFamily.toLowerCase());

    let styleEl = document.getElementById('reading-comfort-styles');
    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.id = 'reading-comfort-styles';
      document.head.appendChild(styleEl);
    }

    const sizeMap = { S: '0.875rem', M: '1rem', L: '1.125rem' };
    const fontMap = {
      Sans: 'ui-sans-serif, system-ui, sans-serif, "Inter", "Segoe UI"',
      Serif: 'ui-serif, Georgia, Cambria, "Times New Roman", serif',
      Mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'
    };

    styleEl.innerHTML = `
      .answer-markdown, .chat-message-content, [class*="message"], .bot-answer {
        font-size: ${sizeMap[textSize]} !important;
        font-family: ${fontMap[fontFamily]} !important;
      }
    `;
  }, [textSize, fontFamily]);

  if (!isOpen) return null;

  return (
    <div className="absolute top-16 left-64 z-50 w-80 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl shadow-xl p-4 text-xs animate-in fade-in zoom-in-95 duration-150">
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-gray-100 dark:border-gray-800">
        <span className="font-semibold text-gray-900 dark:text-gray-100 text-sm">Appearance & Readability</span>
        <button 
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 font-bold px-1.5 py-0.5 rounded"
        >
          ✕
        </button>
      </div>

      {/* Colour Mood Picker */}
      <div className="mb-4">
        <label className="block text-gray-500 dark:text-gray-400 font-medium mb-1.5">Colour Mood</label>
        <div className="grid grid-cols-5 gap-1">
          {Object.keys(ACCENT_PRESETS).map((presetName) => (
            <button
              key={presetName}
              onClick={() => setAccent(presetName)}
              className={`py-1.5 rounded-md font-semibold text-center transition-all ${
                accent === presetName 
                  ? 'bg-[var(--accent)] text-white shadow-xs' 
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200'
              }`}
            >
              {presetName}
            </button>
          ))}
        </div>
      </div>

      {/* Reading Comfort: Text Size */}
      <div className="mb-4">
        <label className="block text-gray-500 dark:text-gray-400 font-medium mb-1.5">Text Size</label>
        <div className="flex space-x-2">
          {['S', 'M', 'L'].map((size) => (
            <button
              key={size}
              onClick={() => setTextSize(size)}
              className={`flex-1 py-1.5 rounded font-bold transition-colors ${
                textSize === size 
                  ? 'bg-[var(--accent-soft)] text-[var(--accent)] border border-[var(--accent)]' 
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
              }`}
            >
              {size}
            </button>
          ))}
        </div>
      </div>

      {/* Reading Comfort: Font Family */}
      <div>
        <label className="block text-gray-500 dark:text-gray-400 font-medium mb-1.5">Font Family</label>
        <div className="flex space-x-2">
          {['Sans', 'Serif', 'Mono'].map((font) => (
            <button
              key={font}
              onClick={() => setFontFamily(font)}
              className={`flex-1 py-1.5 rounded font-medium transition-colors ${
                fontFamily === font 
                  ? 'bg-[var(--accent-soft)] text-[var(--accent)] border border-[var(--accent)] font-semibold' 
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
              }`}
            >
              {font}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}