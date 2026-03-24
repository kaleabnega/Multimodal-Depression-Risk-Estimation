export default function Composer({
  value,
  selectedVideo,
  selectedAudio,
  isSending,
  canSend,
  onChange,
  onVideoChange,
  onAudioChange,
  onClearVideo,
  onClearAudio,
  onSend
}) {
  function handleKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }

    event.preventDefault();
    if (!isSending && canSend) {
      event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <form className="composer" onSubmit={onSend}>
      <div className="composer-main">
        <textarea
          className="composer-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message MDE Assistant"
          rows={1}
        />
        <div className="composer-tools">
          <label className="attach-btn">
            Attach Video
            <input type="file" accept="video/*" onChange={onVideoChange} />
          </label>
          <label className="attach-btn">
            Attach Audio (only ".wav" files)
            <input type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg" onChange={onAudioChange} />
          </label>
          {selectedVideo ? (
            <span className="attach-name">
              Video: {selectedVideo.name}
              <button type="button" className="clear-attach" onClick={onClearVideo}>x</button>
            </span>
          ) : null}
          {selectedAudio ? (
            <span className="attach-name">
              Audio: {selectedAudio.name}
              <button type="button" className="clear-attach" onClick={onClearAudio}>x</button>
            </span>
          ) : null}
        </div>
      </div>
      <button className="composer-send" type="submit" disabled={isSending || !canSend}>
        {isSending ? 'Sending...' : 'Send'}
      </button>
    </form>
  )
}
