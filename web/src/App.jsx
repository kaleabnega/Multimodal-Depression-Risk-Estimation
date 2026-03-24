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
  const [videoFile, setVideoFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [isSending, setIsSending] = useState(false);

  const canSend = useMemo(
    () => (input.trim().length > 0 || Boolean(videoFile) || Boolean(audioFile)) && !isSending,
    [input, videoFile, audioFile, isSending],
  );

  async function handleSend(event) {
    event.preventDefault();
    if (!canSend) return;

    const conversationHistory = messages
      .slice(initialMessages.length)
      .map((message) => ({
        role: message.role,
        text: message.text,
      }));

    const userMessage = {
      id: counter++,
      role: "user",
      text: input.trim() || "(media attached)",
      meta: {
        ...(videoFile ? { attached_video: videoFile.name } : {}),
        ...(audioFile ? { attached_audio: audioFile.name } : {}),
      }
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsSending(true);

    const result = await sendMessage({
      text: input.trim(),
      videoFile,
      audioFile,
      conversationHistory,
      asr_from_audio: Boolean(audioFile),
      debug: true
    });

    setMessages((prev) => [
      ...prev,
      {
        id: counter++,
        role: "assistant",
        text: result.text,
        meta: result.meta
      },
    ]);
    setVideoFile(null);
    setAudioFile(null);
    setIsSending(false);
  }

  function handleVideoChange(event) {
    const file = event.target.files?.[0] ?? null;
    setVideoFile(file);
    // Reset the native input so selecting the same file again triggers onChange.
    event.target.value = "";
  }

  function handleAudioChange(event) {
    const file = event.target.files?.[0] ?? null;
    setAudioFile(file);
    // Reset the native input so selecting the same file again triggers onChange.
    event.target.value = "";
  }

  function handleClearChats() {
    setMessages(initialMessages);
    setInput("");
    setVideoFile(null);
    setAudioFile(null);
    setIsSending(false);
  }

  return (
    <div className="app-shell">
      <ChatSidebar onClearChats={handleClearChats} />
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
            selectedVideo={videoFile}
            selectedAudio={audioFile}
            isSending={isSending}
            canSend={canSend}
            onChange={setInput}
            onVideoChange={handleVideoChange}
            onAudioChange={handleAudioChange}
            onClearVideo={() => setVideoFile(null)}
            onClearAudio={() => setAudioFile(null)}
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
