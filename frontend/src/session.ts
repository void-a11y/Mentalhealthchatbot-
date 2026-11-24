const KEY = "mhrag.session";

function fresh(): string {
  return globalThis.crypto?.randomUUID?.() ?? `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

let memory: string | null = null;

/** Session id lives in sessionStorage when available (per tab), otherwise in memory. */
export function getSessionId(): string {
  try {
    const stored = sessionStorage.getItem(KEY);
    if (stored) return stored;
    const id = fresh();
    sessionStorage.setItem(KEY, id);
    return id;
  } catch {
    return (memory ??= fresh());
  }
}

export function resetSessionId(): string {
  const id = fresh();
  memory = id;
  try {
    sessionStorage.setItem(KEY, id);
  } catch {
    /* storage unavailable */
  }
  return id;
}
