import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, errorFor, sendChat } from "./api";

afterEach(() => vi.unstubAllGlobals());

const ok = { response: "hi [1]", route: "answered", abstained: false, sources: [] };

describe("errorFor", () => {
  it.each([
    [503, "setup"],
    [429, "busy"],
    [401, "auth"],
    [422, "invalid"],
    [502, "server"],
    [500, "server"],
  ])("maps %i to %s", (status, kind) => {
    expect(errorFor(status).kind).toBe(kind);
  });

  it("includes the backend detail for setup errors", () => {
    expect(errorFor(503, "Index not found").message).toContain("Index not found");
  });
});

describe("sendChat", () => {
  it("posts session_id and query as JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ok });
    vi.stubGlobal("fetch", fetchMock);
    const r = await sendChat("abc", "hello", { baseUrl: "http://x" });
    expect(r.route).toBe("answered");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://x/chat");
    expect(JSON.parse(init.body)).toEqual({ session_id: "abc", query: "hello" });
  });

  it("throws a typed error with the backend detail on 503", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: "Index not found" }) }));
    await expect(sendChat("a", "b", { baseUrl: "http://x" })).rejects.toMatchObject({ kind: "setup", status: 503 });
  });

  it("copes with a non-JSON error body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => { throw new Error("no json"); } }));
    await expect(sendChat("a", "b", { baseUrl: "http://x" })).rejects.toBeInstanceOf(ApiError);
  });

  it("reports an unreachable backend", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(sendChat("a", "b", { baseUrl: "http://x" })).rejects.toMatchObject({ kind: "network" });
  });
});
