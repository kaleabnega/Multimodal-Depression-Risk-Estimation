export async function sendMessage(payload) {
  try {
    let res;
    if (payload.videoFile || payload.audioFile) {
      const form = new FormData();
      form.append("text", payload.text ?? "");
      form.append(
        "conversation_history_json",
        JSON.stringify(Array.isArray(payload.conversationHistory) ? payload.conversationHistory : []),
      );
      form.append("asr_from_audio", String(Boolean(payload.asr_from_audio)));
      if (payload.videoFile) {
        form.append("video_file", payload.videoFile);
      }
      if (payload.audioFile) {
        form.append("audio_file", payload.audioFile);
        form.append("asr_model", String(payload.asr_model ?? "openai/whisper-large-v3"));
      }
      form.append("debug", String(Boolean(payload.debug)));
      form.append("video_fps", String(payload.video_fps ?? 2));
      form.append("max_frames", String(payload.max_frames ?? 8));
      res = await fetch(
        "https://multimodal-depression-risk-estimation-production.up.railway.app/api/chat-upload",

        // "/api/chat-upload",

        {
          method: "POST",
          body: form,
        },
      );
    } else {
      res = await fetch(
        "https://multimodal-depression-risk-estimation-production.up.railway.app/api/chat",

        // "/api/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...payload,
            conversation_history: Array.isArray(payload.conversationHistory) ? payload.conversationHistory : [],
          }),
        },
      );
    }

    if (!res.ok) {
      let detail = `API error: ${res.status}`;
      try {
        const errData = await res.json();
        if (errData?.detail) detail = String(errData.detail);
      } catch (_) {
        // Ignore JSON parse errors and keep the status-only message.
      }
      throw new Error(detail);
    }

    const data = await res.json();
    return {
      text: data.response ?? "No response returned.",
      meta: data,
    };
  } catch (err) {
    return {
      text: `Request failed: ${err instanceof Error ? err.message : "unknown error"}`,
      meta: { request_failed: true },
    };
  }
}
