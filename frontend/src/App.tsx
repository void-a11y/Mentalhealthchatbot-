import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, sendChat } from "./api";
import { Composer } from "./Composer";
import { MessageView } from "./MessageView";
import { getSessionId, resetSessionId } from "./session";
import type { Message } from "./types";
import { SidePanel } from "./SidePanel";

const STARTERS = [
  "What is a panic attack?",
  "How can I tell stress from anxiety?",
  "What happens in cognitive behavioral therapy?",
  "How do I talk to my doctor about my mood?",
];

let counter = 0;
const nextId = () => `m${++counter}`;

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ text: string; retry: string } | null>(null);
  const sessionId = useRef(getSessionId());
  const abort = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    endRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "end" });
  }, [messages, loading, error]);

  const send = useCallback(async (text: string, isRetry = false) => {
    setError(null);
    if (!isRetry) setMessages((m) => [...m, { id: nextId(), role: "user", text }]);
    setLoading(true);
    abort.current = new AbortController();
    try {
      const r = await sendChat(sessionId.current, text, { signal: abort.current.signal });
      setMessages((m) => [...m, { id: nextId(), role: "assistant", text: r.response, route: r.route, sources: r.sources }]);
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError({ text: e instanceof ApiError ? e.message : "Something went wrong. Try again.", retry: text });
    } finally {
      setLoading(false);
    }
  }, []);

  const newChat = () => {
    abort.current?.abort();
    sessionId.current = resetSessionId();
    setMessages([]);
    setError(null);
    setLoading(false);
  };

  return (
    <div className="app">
     <SidePanel />
     <img className="corner" src="/brain.jpeg" alt="" />
      <header className="brand">
        <h1>HAVEN </h1>
        <p>a safe and <br/>judgment free <br/>space</p>
        {messages.length > 0 && (
          <button type="button" className="quiet newchat" onClick={newChat}>
            New chat
          </button>
        )}
      </header>

      <main className="thread" aria-live="polite" aria-busy={loading}>
        {messages.length === 0 && !loading ? (
          <div className="empty">
            <h2>Ask a question. Answers come only from the NIMH documents behind this app, with the passages shown.</h2>
            <ul className="starters">
              {STARTERS.map((s) => (
                <li key={s}>
                  <button type="button" onClick={() => send(s)}>
                    {s}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          messages.map((m) => <MessageView key={m.id} message={m} />)
        )}

        {loading && (
          <div className="msg assistant pending" role="status">
            <span className="dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            Searching the documents
          </div>
        )}

        {error && (
          <div className="msg error" role="alert">
            <p>{error.text}</p>
            <button type="button" className="quiet" onClick={() => send(error.retry, true)}>
              Try again
            </button>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <footer className="bottom">
        <Composer disabled={loading} onSend={(t) => send(t)} />
        <p className="disclaimer">Educational project, not medical advice. In an emergency call 112 (India) or 988 (US).</p>
      </footer>
    </div>
  );
}
