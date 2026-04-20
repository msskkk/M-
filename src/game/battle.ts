import { Character, Element, Match } from "./types";

const ATTR_CHART: Record<Element, { strong: Element[]; weak: Element[] }> = {
  fire: { strong: ["wood"], weak: ["water"] },
  water: { strong: ["fire"], weak: ["wood"] },
  wood: { strong: ["water"], weak: ["fire"] },
  light: { strong: ["dark"], weak: ["dark"] },
  dark: { strong: ["light"], weak: ["light"] },
  heart: { strong: [], weak: [] },
};

// Per-match multiplier rises with orb count (4 connected = x1.25, 5 = x1.5, ...)
function matchMultiplier(count: number): number {
  return 1 + (Math.max(count, 3) - 3) * 0.25;
}

// Combo multiplier (PaD style): 1 + (combo - 1) * 0.25
function comboMultiplier(combo: number): number {
  return 1 + (combo - 1) * 0.25;
}

export function computeDamageAndHeal(
  party: Character[],
  matches: Match[],
  enemyElement: Element,
) {
  const damage: Record<Element, number> = {
    fire: 0,
    water: 0,
    wood: 0,
    light: 0,
    dark: 0,
    heart: 0,
  };
  let heal = 0;
  const combos = matches.length;
  const cMul = comboMultiplier(combos);

  for (const m of matches) {
    if (m.element === "heart") {
      const baseRcv = party.reduce((s, c) => s + c.rcv, 0);
      heal += Math.floor(baseRcv * matchMultiplier(m.cells.length) * cMul);
      continue;
    }
    // All alive members whose element matches attack.
    for (const ch of party) {
      if (ch.hp <= 0) continue;
      let mul = matchMultiplier(m.cells.length) * cMul;
      if (ch.element === m.element) {
        let dmg = ch.atk * mul;
        if (ATTR_CHART[ch.element].strong.includes(enemyElement)) dmg *= 2;
        if (ATTR_CHART[ch.element].weak.includes(enemyElement)) dmg *= 0.5;
        damage[m.element] += Math.floor(dmg);
      }
    }
  }
  const totalDamage = Object.values(damage).reduce((a, b) => a + b, 0);
  return { totalDamage, damageByElement: damage, heal, combos };
}
