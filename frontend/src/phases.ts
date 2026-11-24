export interface Phase {
  label: string;
  seconds: number;
  /** Circle size at the end of the phase: 1 = full, small = resting. */
  scale: number;
}

export const PHASES: Phase[] = [
  { label: "Breathe in", seconds: 4, scale: 1 },
  { label: "Hold", seconds: 4, scale: 1 },
  { label: "Breathe out", seconds: 6, scale: 0.5 },
];

export const nextPhase = (i: number): number => (i + 1) % PHASES.length;
