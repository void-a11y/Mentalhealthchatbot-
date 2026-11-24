import { describe, expect, it } from "vitest";
import { parseBlocks, parseInline } from "./parse";

describe("parseInline", () => {
  it("extracts citation markers and bold", () => {
    expect(parseInline("Rest helps [1][2]. **Avoid** caffeine [3].")).toEqual([
      { t: "text", v: "Rest helps " },
      { t: "cite", n: 1 },
      { t: "cite", n: 2 },
      { t: "text", v: ". " },
      { t: "bold", v: "Avoid" },
      { t: "text", v: " caffeine " },
      { t: "cite", n: 3 },
      { t: "text", v: "." },
    ]);
  });

  it("leaves HTML as plain text", () => {
    expect(parseInline("<img src=x onerror=alert(1)>")).toEqual([{ t: "text", v: "<img src=x onerror=alert(1)>" }]);
  });

  it("does not treat non-numeric brackets as citations", () => {
    expect(parseInline("see [note]")).toEqual([{ t: "text", v: "see [note]" }]);
  });
});

describe("parseBlocks", () => {
  it("splits paragraphs and keeps single line breaks inside one", () => {
    const b = parseBlocks("Call now.\nTele-MANAS 14416\n\nYou matter.");
    expect(b).toHaveLength(2);
    expect(b[0]).toMatchObject({ t: "p" });
  });

  it("parses bullet and numbered lists separately", () => {
    const b = parseBlocks("Try:\n- walk [1]\n- breathe\n\n1. sleep\n2. eat");
    expect(b.map((x) => x.t)).toEqual(["p", "list", "list"]);
    expect(b[1]).toMatchObject({ ordered: false });
    expect(b[2]).toMatchObject({ ordered: true });
    expect((b[1] as { items: unknown[] }).items).toHaveLength(2);
  });

  it("returns nothing for empty text", () => {
    expect(parseBlocks("  \n ")).toEqual([]);
  });
});
