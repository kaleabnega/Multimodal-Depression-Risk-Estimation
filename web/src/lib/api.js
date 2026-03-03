export async function sendMessage(payload) {
  try {
    let res;
    if (payload.videoFile) {
      const form = new FormData();
      form.append("text", payload.text ?? "");
      form.append("video_file", payload.videoFile);
      form.append("debug", String(Boolean(payload.debug)));
      form.append("video_fps", String(payload.video_fps ?? 2));
      form.append("max_frames", String(payload.max_frames ?? 8));
      res = await fetch(
        "https://multimodal-depression-risk-estimation-production.up.railway.app/api/chat-upload",
        {
          method: "POST",
          body: form,
        },
      );
    } else {
      res = await fetch(
        "https://multimodal-depression-risk-estimation-production.up.railway.app/api/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
    }

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    const data = await res.json();
    return {
      text: data.response ?? "No response returned.",
      meta: data,
    };
  } catch (_) {
    return {
      text: "Demo mode: backend is not connected yet. Your frontend is ready; connect /api/chat to scripts/run_demo.py via a server wrapper.",
      meta: { demo_mode: true },
    };
  }
}
