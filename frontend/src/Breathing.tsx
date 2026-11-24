import { useEffect, useState } from "react";
import { PHASES, nextPhase } from "./phases";

export function Breathing() {
  const [running, setRunning] = useState(false);
  const [i, setI] = useState(0);

  useEffect(() => {
    if (!running) return;
    const t = setTimeout(() => setI(nextPhase), PHASES[i].seconds * 1000);
    return () => clearTimeout(t);
  }, [running, i]);

  const toggle = () => {
    setI(0);
    setRunning((r) => !r);
  };

  const phase = PHASES[i];
  const scale = running ? phase.scale : 0.5;
  const ms = running && phase.label !== "Hold" ? phase.seconds * 1000 : 0;

  return (
    <section className="side-block" aria-label="Breathing exercise">
      <h2>Take a minute</h2>
      <div className="breath">
        <div className="breath-ring" aria-hidden="true">
          <div className="breath-dot" style={{ transform: `scale(${scale})`, transitionDuration: `${ms}ms` }} />
        </div>
        <p className="breath-label">{running ? phase.label : "Breathe with the circle"}</p>
      </div>
      <button type="button" className="side-btn" onClick={toggle}>
        {running ? "Stop" : "Start"}
      </button>
    </section>
  );
}
