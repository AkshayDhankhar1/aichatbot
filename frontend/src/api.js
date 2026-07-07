// Thin client for the backend /chat SSE stream.
// Base URL comes from env (never hardcoded) so it can point at the deployed
// Railway/Render backend in production.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Stream a chat answer. Calls the provided callbacks as events arrive.
 *
 * @param {string} message                 - user question
 * @param {Array<{role,content}>} history  - prior session turns (no server persistence)
 * @param {object} handlers
 * @param {(text:string)=>void} handlers.onToken  - incremental text chunk
 * @param {(payload:{sources:string[], chart:object|null})=>void} handlers.onDone
 * @param {(message:string)=>void} handlers.onError
 * @param {AbortSignal} [signal]
 */
export async function streamChat(message, history, handlers, signal) {
  const { onToken, onDone, onError } = handlers;

  let resp;
  try {
    resp = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history }),
      signal,
    });
  } catch (e) {
    onError?.(`Could not reach the server. Is the backend running? (${e.message})`);
    return;
  }

  if (!resp.ok || !resp.body) {
    onError?.(`Server error: ${resp.status} ${resp.statusText}`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buffer.split('\n\n');
      buffer = frames.pop() || '';

      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!line) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        let event;
        try {
          event = JSON.parse(json);
        } catch {
          continue;
        }
        if (event.type === 'token') onToken?.(event.content);
        else if (event.type === 'done') onDone?.({ sources: event.sources || [], chart: event.chart || null });
        else if (event.type === 'error') onError?.(event.message);
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') onError?.(e.message);
  }
}
