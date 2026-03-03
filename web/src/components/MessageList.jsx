export default function MessageList({ messages }) {
  return (
    <div className="message-list">
      {messages.map((msg) => (
        <article key={msg.id} className={`message-row ${msg.role}`}>
          <div className="message-avatar" aria-hidden="true">{msg.role === 'user' ? 'U' : 'A'}</div>
          <div className="message-body">
            <p className="message-role">{msg.role === 'user' ? 'You' : 'Assistant'}</p>
            <p className="message-text">{msg.text}</p>
            {msg.role === 'assistant' && msg.meta?.response_source ? (
              <p className="message-meta">
                source: {msg.meta.response_source}
                {msg.meta.response_model ? ` | model: ${msg.meta.response_model}` : ''}
              </p>
            ) : null}
          </div>
        </article>
      ))}
    </div>
  )
}
