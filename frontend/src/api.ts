import type { ChatResponse } from "./types";

export const API_URL: string = (import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

export type ErrorKind = "setup" | "busy" | "auth" | "server" | "invalid" | "network";

export class ApiError extends Error {
  constructor(public kind: ErrorKind, message: string, public status?: number) {
    super(message);
  }
}

/** Turn an HTTP status (and the FastAPI `detail`, if any) into a message the reader can act on. */
export function errorFor(status: number, detail?: string): ApiError {
  switch (status) {
    case 503:
      return new ApiError("setup", `The assistant isn't ready yet. ${detail ?? "Build the index and set GROQ_API_KEY on the backend."}`.trim(), status);
    case 429:
      return new ApiError("busy", "The language model is busy. Wait a few seconds and send it again.", status);
    case 401:
      return new ApiError("auth", "The language model rejected the API key. Check GROQ_API_KEY in the backend's .env file.", status);
    case 422:
      return new ApiError("invalid", "That message couldn't be sent. Keep it between 1 and 2000 characters.", status);
    default:
      return new ApiError("server", "The language model service returned an error. Try again in a moment.", status);
  }
}

export async function sendChat(
  sessionId: string,
  query: string,
  opts: { signal?: AbortSignal; baseUrl?: string } = {},
): Promise<ChatResponse> {
  const base = opts.baseUrl ?? API_URL;
  let res: Response;
  try {
    res = await fetch(`${base}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query }),
      signal: opts.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("network", `Can't reach the backend at ${base}. Check that it is running.`);
  }
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* no JSON body */
    }
    throw errorFor(res.status, detail);
  }
  return (await res.json()) as ChatResponse;
}
