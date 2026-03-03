export default function Composer({ value, selectedVideo, isSending, onChange, onVideoChange, onClearVideo, onSend }) {
  return (
    <form className="composer" onSubmit={onSend}>
      <div className="composer-main">
        <textarea
          className="composer-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Message MDE Assistant"
          rows={1}
        />
        <div className="composer-tools">
          <label className="attach-btn">
            Attach Video
            <input type="file" accept="video/*" onChange={onVideoChange} />
          </label>
          {selectedVideo ? (
            <span className="attach-name">
              {selectedVideo.name}
              <button type="button" className="clear-attach" onClick={onClearVideo}>x</button>
            </span>
          ) : null}
        </div>
      </div>
      <button className="composer-send" type="submit" disabled={isSending || !value.trim()}>
        {isSending ? 'Sending...' : 'Send'}
      </button>
    </form>
  )
}
