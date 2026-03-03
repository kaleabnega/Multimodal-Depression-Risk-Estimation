import { useMemo, useState } from "react";
import ChatSidebar from "./components/ChatSidebar";
import MessageList from "./components/MessageList";
import Composer from "./components/Composer";
import { sendMessage } from "./lib/api";

let counter = 3;

const initialMessages = [
  {
    id: 1,
    role: "assistant",
    text: "I am here to help you reflect on text, audio, and visual cues. What would you like to check today?",
  },
  {
    id: 2,
    role: "assistant",
    text: "If this is urgent or safety-related, contact local emergency services.",
  },
];

export default function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const canSend = useMemo(
    () => input.trim().length > 0 && !isSending,
    [input, isSending],
  );

  async function handleSend(event) {
    event.preventDefault();
    if (!canSend) return;

    const userMessage = { id: counter++, role: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsSending(true);

    const result = await sendMessage({ text: userMessage.text });

    setMessages((prev) => [
      ...prev,
      {
        id: counter++,
        role: "assistant",
        text: result.text,
        meta: result.meta
      },
    ]);
    setIsSending(false);
  }

  return (
    <div className="app-shell">
      <ChatSidebar />
      <main className="chat-main">
        <header className="chat-header">
          <h1 className="chat-title">
            Multimodal Depression Support Assistant
          </h1>
          <p className="chat-subtitle">
            Text, audio, and video-aware safety-grounded conversation
          </p>
        </header>

        <section className="chat-thread" aria-live="polite">
          <MessageList messages={messages} />
        </section>

        <footer className="chat-footer">
          <Composer
            value={input}
            isSending={isSending}
            onChange={setInput}
            onSend={handleSend}
          />
          <p className="footer-note">
            Use this as supportive guidance only. It is not a diagnostic tool.
          </p>
        </footer>
      </main>
    </div>
  );
}
