import type { Source } from "./types";

interface Props {
  sources: Source[];
  open: Set<number>;
  onToggle: (n: number) => void;
}

export function Sources({ sources, open, onToggle }: Props) {
  if (!sources.length) return null;
  return (
    <section className="sources" aria-label="Sources">
      <p className="sources-title">{sources.length === 1 ? "1 source" : `${sources.length} sources`}</p>
      <ul>
        {sources.map((s) => {
          const isOpen = open.has(s.n);
          return (
            <li key={s.n} id={`source-${s.n}`} className={isOpen ? "source open" : "source"}>
              <button type="button" className="source-head" aria-expanded={isOpen} onClick={() => onToggle(s.n)}>
                <span className="source-n">{s.n}</span>
                <span className="source-name">
                  {s.title}
                  {s.section ? <span className="source-section"> / {s.section}</span> : null}
                </span>
                <svg className="chev" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {isOpen && (
                <div className="source-body">
                  <blockquote>{s.snippet.trim()}{s.snippet.length >= 240 ? "…" : ""}</blockquote>
                  <p className="source-meta">
                    From {s.source}
                    {s.url ? (
                      <>
                        {" "}
                        <a href={s.url} target="_blank" rel="noreferrer noopener">
                          Open the original page
                        </a>
                      </>
                    ) : null}
                  </p>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
