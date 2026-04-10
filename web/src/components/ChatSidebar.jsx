export default function ChatSidebar({ onClearChats }) {
  return (
    <aside className="chat-sidebar" aria-label="Navigation">
      <div className="brand-lockup">
        <div className="brand-mark" />
        <div>
          <p className="brand-title">MDE Assistant</p>
        </div>
      </div>

      <button className="clear-chats-btn" type="button" onClick={onClearChats}>
        Clear Chats
      </button>

   
    </aside>
  );
}
