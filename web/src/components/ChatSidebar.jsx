export default function ChatSidebar({ onClearChats }) {
  return (
    <aside className="chat-sidebar" aria-label="Navigation">
      <div className="brand-lockup">
        <div className="brand-mark" />
        <div>
          <p className="brand-title">MDE Assistant</p>
          <p className="brand-sub">Multimodal support</p>
        </div>
      </div>

      <button className="clear-chats-btn" type="button" onClick={onClearChats}>
        Clear Chats
      </button>

      {/* <nav className="sidebar-section" aria-label="Recent">
        <p className="section-title">Recent</p>
        <button className="history-item" type="button">Facial expression check</button>
        <button className="history-item" type="button">Audio + text reflection</button>
      </nav> */}

      <div className="sidebar-footer">
        <p className="safety-note">
          For support and screening only, not diagnosis.
        </p>
      </div>
    </aside>
  );
}
