import {
  Board,
  COLS,
  ELEMENTS,
  Element,
  Match,
  Orb,
  ROWS,
} from "./types";

let nextId = 1;
const newId = () => nextId++;

const randElement = (): Element =>
  ELEMENTS[Math.floor(Math.random() * ELEMENTS.length)];

export function createBoard(): Board {
  const b: Board = [];
  for (let r = 0; r < ROWS; r++) {
    const row: (Orb | null)[] = [];
    for (let c = 0; c < COLS; c++) {
      row.push({ id: newId(), element: randElement() });
    }
    b.push(row);
  }
  // Regenerate until no initial matches, for a clean start.
  while (findMatches(b).length > 0) {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        b[r][c] = { id: newId(), element: randElement() };
      }
    }
  }
  return b;
}

export function cloneBoard(b: Board): Board {
  return b.map((row) => row.map((o) => (o ? { ...o } : null)));
}

export function swap(b: Board, a: [number, number], c: [number, number]): Board {
  const nb = cloneBoard(b);
  const tmp = nb[a[0]][a[1]];
  nb[a[0]][a[1]] = nb[c[0]][c[1]];
  nb[c[0]][c[1]] = tmp;
  return nb;
}

// Find connected runs of 3+ same-element orbs horizontally or vertically,
// then merge overlapping runs into match groups (T/L shapes count as one).
export function findMatches(b: Board): Match[] {
  const runs: { cells: Set<string>; element: Element }[] = [];

  const key = (r: number, c: number) => `${r},${c}`;

  // horizontal
  for (let r = 0; r < ROWS; r++) {
    let c = 0;
    while (c < COLS) {
      const orb = b[r][c];
      if (!orb) {
        c++;
        continue;
      }
      let end = c + 1;
      while (end < COLS && b[r][end]?.element === orb.element) end++;
      if (end - c >= 3) {
        const cells = new Set<string>();
        for (let k = c; k < end; k++) cells.add(key(r, k));
        runs.push({ cells, element: orb.element });
      }
      c = end;
    }
  }

  // vertical
  for (let c = 0; c < COLS; c++) {
    let r = 0;
    while (r < ROWS) {
      const orb = b[r][c];
      if (!orb) {
        r++;
        continue;
      }
      let end = r + 1;
      while (end < ROWS && b[end][c]?.element === orb.element) end++;
      if (end - r >= 3) {
        const cells = new Set<string>();
        for (let k = r; k < end; k++) cells.add(key(k, c));
        runs.push({ cells, element: orb.element });
      }
      r = end;
    }
  }

  // merge overlapping runs of the same element
  const merged: { cells: Set<string>; element: Element }[] = [];
  for (const run of runs) {
    let absorbed = false;
    for (const m of merged) {
      if (m.element !== run.element) continue;
      let overlaps = false;
      for (const k of run.cells) {
        if (m.cells.has(k)) {
          overlaps = true;
          break;
        }
      }
      if (overlaps) {
        for (const k of run.cells) m.cells.add(k);
        absorbed = true;
        break;
      }
    }
    if (!absorbed) merged.push({ cells: new Set(run.cells), element: run.element });
  }

  return merged.map((m) => ({
    element: m.element,
    cells: [...m.cells].map((s) => {
      const [r, c] = s.split(",").map(Number);
      return [r, c] as [number, number];
    }),
  }));
}

export function removeMatches(b: Board, matches: Match[]): Board {
  const nb = cloneBoard(b);
  for (const m of matches) {
    for (const [r, c] of m.cells) nb[r][c] = null;
  }
  return nb;
}

// Gravity: orbs fall, new ones spawn at top.
export function applyGravity(b: Board): Board {
  const nb = cloneBoard(b);
  for (let c = 0; c < COLS; c++) {
    const stack: Orb[] = [];
    for (let r = ROWS - 1; r >= 0; r--) {
      const o = nb[r][c];
      if (o) stack.push(o);
    }
    for (let r = ROWS - 1; r >= 0; r--) {
      nb[r][c] = stack.shift() ?? null;
    }
    for (let r = 0; r < ROWS; r++) {
      if (!nb[r][c]) nb[r][c] = { id: newId(), element: randElement() };
    }
  }
  return nb;
}
