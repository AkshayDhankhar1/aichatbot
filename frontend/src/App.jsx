import { useEffect, useRef, useState } from 'react';
import './App.css';
import { streamChat } from './api';
import Message from './components/Message';

const WELCOME = {
  role: 'assistant',
  content:
    "Hi! I answer questions strictly from the CRM documents that were ingested. " +
    "Try “What was total closed-won revenue?” or ask me to “plot deals by stage as a bar chart”.",
  sources: [],
  chart: null,
};

export default function App() {
  // Session-only state — nothing is persisted (no DB, cleared on refresh).
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const listRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;

    const userMsg = { role: 'user', content: text };
    // History sent to backend = everything so far (excluding the welcome note).
    const history = messages
      .filter((m) => m !== WELCOME)
      .map(({ role, content }) => ({ role, content }));

    const botMsg = { role: 'assistant', content: '', sources: [], chart: null, streaming: true };
    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput('');
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    const updateBot = (patch) =>
      setMessages((prev) => {
        const next = [...prev];
        const i = next.length - 1;
        next[i] = { ...next[i], ...patch(next[i]) };
        return next;
      });

    await streamChat(
      text,
      history,
      {
        onToken: (chunk) => updateBot((m) => ({ content: m.content + chunk })),
        onDone: ({ sources, chart }) => updateBot(() => ({ sources, chart, streaming: false })),
        onError: (msg) =>
          updateBot((m) => ({
            content: m.content || `⚠️ ${msg}`,
            streaming: false,
          })),
      },
      controller.signal,
    );

    setBusy(false);
    abortRef.current = null;
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function clearChat() {
    abortRef.current?.abort();
    setMessages([WELCOME]);
    setBusy(false);
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-dot" />
          CRM RAG Chatbot
        </div>
        <button className="ghost-btn" onClick={clearChat}>New chat</button>
      </header>

      <main className="chat-list" ref={listRef}>
        {messages.map((m, i) => (
          <Message key={i} {...m} />
        ))}
      </main>

      <footer className="composer">
        <textarea
          className="composer-input"
          placeholder="Ask about the CRM documents…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
        />
        <button className="send-btn" onClick={send} disabled={busy || !input.trim()}>
          {busy ? '…' : 'Send'}
        </button>
      </footer>
    </div>
  );
}
