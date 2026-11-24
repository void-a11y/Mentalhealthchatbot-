import { describe, expect, it } from "vitest";
import { PHASES, nextPhase } from "./phases";

describe("breathing phases", () => {
  it("cycles in, hold, out and wraps around", () => {
    expect(PHASES.map((p) => p.label)).toEqual(["Breathe in", "Hold", "Breathe out"]);
    expect(nextPhase(0)).toBe(1);
    expect(nextPhase(2)).toBe(0);
  });

  it("breathes out longer than in", () => {
    expect(PHASES[2].seconds).toBeGreaterThan(PHASES[0].seconds);
  });
});
