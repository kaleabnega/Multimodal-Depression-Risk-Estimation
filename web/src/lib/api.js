export async function sendMessage(payload) {
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`)
    }

    const data = await res.json()
    return {
      text: data.response ?? 'No response returned.',
      meta: data
    }
  } catch (_) {
    return {
      text: 'Demo mode: backend is not connected yet. Your frontend is ready; connect /api/chat to scripts/run_demo.py via a server wrapper.',
      meta: { demo_mode: true }
    }
  }
}
