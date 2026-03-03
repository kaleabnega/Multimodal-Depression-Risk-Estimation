export default function Composer({ value, isSending, onChange, onSend }) {
  return (
    <form className="composer" onSubmit={onSend}>
      <textarea
        className="composer-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Message MDE Assistant"
        rows={1}
      />
      <button className="composer-send" type="submit" disabled={isSending || !value.trim()}>
        {isSending ? 'Sending...' : 'Send'}
      </button>
    </form>
  )
}
