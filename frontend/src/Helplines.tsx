const LINES = [
  { name: "Tele-MANAS (India, 24/7, free)", number: "14416", tel: "14416" },
  { name: "KIRAN helpline (India)", number: "1800-599-0019", tel: "18005990019" },
  { name: "Emergency (India)", number: "112", tel: "112" },
  { name: "Call or text (US)", number: "988", tel: "988" },
];

export function Helplines() {
  return (
    <section className="side-block" aria-label="Helplines">
      <h2>Need to talk to someone?</h2>
      <ul className="lines">
        {LINES.map((l) => (
          <li key={l.name}>
            <span>{l.name}</span>
            <a href={`tel:${l.tel}`}>{l.number}</a>
          </li>
        ))}
      </ul>
    </section>
  );
}
