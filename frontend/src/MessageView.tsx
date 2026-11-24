import { useState } from "react";
import { RichText } from "./RichText";
import { Sources } from "./Sources";
import type { Message } from "./types";

export function MessageView({ message }: { message: Message }) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const toggle = (n: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });
  const openOnly = (n: number) => {
    toggle(n);
    requestAnimationFrame(() => document.getElementById(`source-${n}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  };

  if (message.role === "user") {
    return (
      <div className="msg user">
        <p>{message.text}</p>
      </div>
    );
  }

  const sources = message.sources ?? [];
  const valid = new Set(sources.map((s) => s.n));
  const route = message.route ?? "answered";
  const cls = `msg assistant ${route}`;
  return (
    <div className={cls} role={route === "crisis" ? "alert" : undefined}>
      {route === "crisis" && <p className="notice-title">Please reach out for support now</p>}
      {route === "not_found" && <p className="notice-title">Not in the documents</p>}
      {route === "out_of_scope" && <p className="notice-title">Better for a professional</p>}
      <div className="prose">
        <RichText text={message.text} validCites={valid} openCites={open} onCite={openOnly} />
      </div>
      <Sources sources={sources} open={open} onToggle={toggle} />
    </div>
  );
}
