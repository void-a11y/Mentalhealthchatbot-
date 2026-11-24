/** A tiny, safe formatter: **bold**, [n] citation markers, bullet and numbered lists.
 *  It returns data, never HTML, so nothing from the model is injected into the page. */

export type Inline = { t: "text"; v: string } | { t: "bold"; v: string } | { t: "cite"; n: number };
export type Block =
  | { t: "p"; inlines: Inline[] }
  | { t: "list"; ordered: boolean; items: Inline[][] };

const INLINE = /\*\*(.+?)\*\*|\[(\d{1,3})\]/g;

export function parseInline(s: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of s.matchAll(INLINE)) {
    const i = m.index ?? 0;
    if (i > last) out.push({ t: "text", v: s.slice(last, i) });
    if (m[1] !== undefined) out.push({ t: "bold", v: m[1] });
    else out.push({ t: "cite", n: Number(m[2]) });
    last = i + m[0].length;
  }
  if (last < s.length) out.push({ t: "text", v: s.slice(last) });
  return out;
}

const BULLET = /^\s*[-*•]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;

export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: Inline[][] } | null = null;

  const flushPara = () => {
    if (para.length) blocks.push({ t: "p", inlines: parseInline(para.join("\n")) });
    para = [];
  };
  const flushList = () => {
    if (list) blocks.push({ t: "list", ...list });
    list = null;
  };

  for (const line of text.replace(/\r\n?/g, "\n").split("\n")) {
    const b = BULLET.exec(line);
    const n = b ? null : NUMBERED.exec(line);
    const item = b ?? n;
    if (item) {
      flushPara();
      const ordered = !!n;
      if (list && list.ordered !== ordered) flushList();
      list ??= { ordered, items: [] };
      list.items.push(parseInline(item[1]));
    } else if (!line.trim()) {
      flushPara();
      flushList();
    } else {
      flushList();
      para.push(line.trim());
    }
  }
  flushPara();
  flushList();
  return blocks;
}
