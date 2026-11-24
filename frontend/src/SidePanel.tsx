import "./side.css";
import { Breathing } from "./Breathing";
import { Helplines } from "./Helplines";

/** Sits in the green strip to the right of the chat. Hidden on narrow screens. */
export function SidePanel() {
  return (
    <aside className="side">
      <Helplines />
      <Breathing />
    </aside>
  );
}
