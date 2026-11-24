import type { Inline } from "./parse";
import { parseBlocks } from "./parse";

interface Props {
  text: string;
  validCites?: Set<number>;
  openCites?: Set<number>;
  onCite?: (n: number) => void;
}

function Inlines({ items, validCites, openCites, onCite }: { items: Inline[] } & Omit<Props, "text">) {
  return (
    <>
      {items.map((it, i) => {
        if (it.t === "text") return <span key={i}>{it.v}</span>;
        if (it.t === "bold") return <strong key={i}>{it.v}</strong>;
        if (!validCites?.has(it.n)) return null;
        return (
          <button
            key={i}
            type="button"
            className="cite"
            aria-label={`Source ${it.n}`}
            aria-expanded={openCites?.has(it.n) ?? false}
            aria-controls={`source-${it.n}`}
            onClick={() => onCite?.(it.n)}
          >
            {it.n}
          </button>
        );
      })}
    </>
  );
}

export function RichText({ text, validCites, openCites, onCite }: Props) {
  const shared = { validCites, openCites, onCite };
  return (
    <>
      {parseBlocks(text).map((b, i) =>
        b.t === "p" ? (
          <p key={i}>
            <Inlines items={b.inlines} {...shared} />
          </p>
        ) : b.ordered ? (
          <ol key={i}>
            {b.items.map((it, j) => (
              <li key={j}>
                <Inlines items={it} {...shared} />
              </li>
            ))}
          </ol>
        ) : (
          <ul key={i}>
            {b.items.map((it, j) => (
              <li key={j}>
                <Inlines items={it} {...shared} />
              </li>
            ))}
          </ul>
        ),
      )}
    </>
  );
}
