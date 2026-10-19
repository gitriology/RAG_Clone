import React, { useEffect, useState } from 'react';
import { ReadAloudButton } from './ReadAloudButton';
import { ExportChatButton } from './ExportChatButton';

const ACCENT_PRESETS = {
  Sage: { 
    light: '#4B6B57', dark: '#739E82', 
    softLight: '#E8F0EC', softDark: '#1E2E23',
    bgLight: '#F3F7F4', bgDark: '#111813',
    panelLight: '#FFFFFF', panelDark: '#1A241C',
    panelSoftLight: '#EBF2EE', panelSoftDark: '#222F24',
    lineLight: '#D5E2D9', lineDark: '#2B3B2E'
  },
  Ocean: { 
    light: '#2563EB', dark: '#60A5FA', 
    softLight: '#EFF6FF', softDark: '#1E3A8A',
    bgLight: '#F1F5F9', bgDark: '#0A0F1D',
    panelLight: '#FFFFFF', panelDark: '#111827',
    panelSoftLight: '#E2E8F0', panelSoftDark: '#1F2937',
    lineLight: '#CBD5E1', lineDark: '#374151'
  },
  Lavender: { 
    light: '#7C3AED', dark: '#A78BFA', 
    softLight: '#F5F3FF', softDark: '#2E1065',
    bgLight: '#F6F3FA', bgDark: '#140D1C',
    panelLight: '#FFFFFF', panelDark: '#1E1428',
    panelSoftLight: '#EDE8F5', panelSoftDark: '#2A1C38',
    lineLight: '#E9E2F2', lineDark: '#322244'
  },
  Rose: { 
    light: '#E11D48', dark: '#FB7185', 
    softLight: '#FFF1F2', softDark: '#4C0519',
    bgLight: '#FAEDF0', bgDark: '#1C0D10',
    panelLight: '#FFFFFF', panelDark: '#2A1317',
    panelSoftLight: '#F7D7DE', panelSoftDark: '#3D1A21',
    lineLight: '#F5D0D7', lineDark: '#442027'
  },
  Amber: { 
    light: '#D97706', dark: '#FBBF24', 
    softLight: '#FFFBEB', softDark: '#451A03',
    bgLight: '#FAF5ED', bgDark: '#1C160C',
    panelLight: '#FFFFFF', panelDark: '#292013',
    panelSoftLight: '#F5EBDD', panelSoftDark: '#3B2A18',
    lineLight: '#F2E2C9', lineDark: '#42331D'
  }
};

export function SettingsModal({ isOpen, onClose }) {
  const [accent, setAccent] = useState('Ocean');
  const [textSize, setTextSize] = useState('M'); // S, M, L
  const [fontFamily, setFontFamily] = useState('Sans'); // Sans, Serif, Mono

  // Synchronize Colour Mood variables with Header theme toggle (data-theme / class)
  useEffect(() => {
    const root = document.documentElement;
    
    const applyMood = () => {
      const isDark = root.getAttribute('data-theme') === 'dark' || root.classList.contains('dark');
      const preset = ACCENT_PRESETS[accent];
      
      root.style.setProperty('--accent', isDark ? preset.dark : preset.light);
      root.style.setProperty('--accent-soft', isDark ? preset.softDark : preset.softLight);
      root.style.setProperty('--bg', isDark ? preset.bgDark : preset.bgLight);
      root.style.setProperty('--panel', isDark ? preset.panelDark : preset.panelLight);
      root.style.setProperty('--panel-solid', isDark ? preset.panelDark : preset.panelLight);
      root.style.setProperty('--panel-soft', isDark ? preset.panelSoftDark : preset.panelSoftLight);
      root.style.setProperty('--line', isDark ? preset.lineDark : preset.lineLight);
    };

    applyMood();

    // Observe changes from the header theme toggle
    const observer = new MutationObserver(applyMood);
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme', 'class'] });

    return () => observer.disconnect();
  }, [accent]);

  // Apply Reading Comfort globally across the entire project
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

    const sizeMap = { S: '0.85rem', M: '1rem', L: '1.15rem' };
    const fontMap = {
      Sans: 'ui-sans-serif, system-ui, sans-serif, "Inter", "Segoe UI"',
      Serif: 'ui-serif, Georgia, Cambria, "Times New Roman", serif',
      Mono: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'
    };

    styleEl.innerHTML = `
      body, .app-shell, input, button, textarea, p, span, div, .chat-message-content, .answer-markdown, .user-message {
        font-size: ${sizeMap[textSize]} !important;
        font-family: ${fontMap[fontFamily]} !important;
      }
    `;
  }, [textSize, fontFamily]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-xs p-4">
      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-800">
          <h2 className="text-base font-bold text-gray-900 dark:text-gray-100">Settings & Customization</h2>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 p-1 rounded-lg transition-colors cursor-pointer"
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="px-6 py-5 space-y-6 max-h-[75vh] overflow-y-auto text-sm">
          
          {/* 1. Colour Mood */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 font-semibold mb-2">Colour Mood</label>
            <div className="grid grid-cols-5 gap-2">
              {Object.keys(ACCENT_PRESETS).map((presetName) => (
                <button
                  key={presetName}
                  onClick={() => setAccent(presetName)}
                  className={`py-2 rounded-lg font-medium text-xs text-center transition-all cursor-pointer ${
                    accent === presetName 
                      ? 'bg-[var(--accent)] text-white shadow-sm ring-2 ring-offset-2 ring-[var(--accent)]' 
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                >
                  {presetName}
                </button>
              ))}
            </div>
          </div>

          {/* 2. Reading Comfort: Text Size */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 font-semibold mb-2">Global Text Size</label>
            <div className="flex space-x-3">
              {['S', 'M', 'L'].map((size) => (
                <button
                  key={size}
                  onClick={() => setTextSize(size)}
                  className={`flex-1 py-2 rounded-lg font-bold text-xs transition-colors cursor-pointer ${
                    textSize === size 
                      ? 'bg-[var(--accent-soft)] text-[var(--accent)] border-2 border-[var(--accent)]' 
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-transparent'
                  }`}
                >
                  {size}
                </button>
              ))}
            </div>
          </div>

          {/* 3. Reading Comfort: Font Family */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 font-semibold mb-2">Global Font Family</label>
            <div className="flex space-x-3">
              {['Sans', 'Serif', 'Mono'].map((font) => (
                <button
                  key={font}
                  onClick={() => setFontFamily(font)}
                  className={`flex-1 py-2 rounded-lg font-medium text-xs transition-colors cursor-pointer ${
                    fontFamily === font 
                      ? 'bg-[var(--accent-soft)] text-[var(--accent)] border-2 border-[var(--accent)] font-semibold' 
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-transparent'
                  }`}
                >
                  {font}
                </button>
              ))}
            </div>
          </div>

          <hr className="border-gray-100 dark:border-gray-800" />

          {/* 4. Read Aloud Controls */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 font-semibold mb-2">Text-to-Speech (Read Aloud) a Chat</label>
            <ReadAloudButton />
          </div>

          {/* 5. Export Chat Controls */}
          <div>
            <label className="block text-gray-700 dark:text-gray-300 font-semibold mb-2">Conversation Export a Chat</label>
            <ExportChatButton />
          </div>

        </div>

        {/* Modal Footer */}
        <div className="flex justify-end px-6 py-3 bg-gray-50 dark:bg-gray-900/50 border-t border-gray-100 dark:border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[var(--accent)] text-white text-xs font-semibold rounded-lg hover:opacity-90 transition-opacity cursor-pointer"
          >
            Save & Close
          </button>
        </div>

      </div>
    </div>
  );
}