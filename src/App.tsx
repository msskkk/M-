import { useEffect, useMemo, useRef, useState } from "react";
import {
  applyGravity,
  createBoard,
  findMatches,
  removeMatches,
  swap,
} from "./game/board";
import { computeDamageAndHeal } from "./game/battle";
import {
  Board,
  COLS,
  Character,
  ELEMENT_COLOR,
  ELEMENT_LABEL,
  Enemy,
  Match,
  ROWS,
} from "./game/types";
import { OrbCell } from "./components/OrbCell";

type Phase = "idle" | "dragging" | "resolving" | "enemyTurn" | "win" | "lose";

const INITIAL_PARTY: Character[] = [
  {
    name: "炎竜",
    element: "fire",
    maxHp: 1800,
    hp: 1800,
    atk: 680,
    rcv: 120,
    color: "#ff4d3b",
  },
  {
    name: "水蛇",
    element: "water",
    maxHp: 2000,
    hp: 2000,
    atk: 540,
    rcv: 180,
    color: "#3ab0ff",
  },
  {
    name: "木亀",
    element: "wood",
    maxHp: 2400,
    hp: 2400,
    atk: 480,
    rcv: 220,
    color: "#4ed36b",
  },
];

const createEnemy = (): Enemy => ({
  name: "氷塔ドラゴン",
  maxHp: 12000,
  hp: 12000,
  atk: 1200,
  turnMax: 3,
  turn: 3,
  element: "water",
});

export default function App() {
  const [board, setBoard] = useState<Board>(() => createBoard());
  const [party, setParty] = useState<Character[]>(INITIAL_PARTY);
  const [enemy, setEnemy] = useState<Enemy>(createEnemy);
  const [phase, setPhase] = useState<Phase>("idle");
  const [flash, setFlash] = useState<Set<string> | null>(null);
  const [message, setMessage] = useState<string>("ドロップを動かそう！");
  const [comboCount, setComboCount] = useState(0);

  const boardRef = useRef<HTMLDivElement | null>(null);
  const cellSizeRef = useRef<number>(0);
  const dragRef = useRef<{
    held: [number, number] | null;
    lastPos: [number, number] | null;
  }>({ held: null, lastPos: null });

  const totalHp = useMemo(
    () => party.reduce((s, c) => s + c.maxHp, 0),
    [party],
  );
  const currentHp = useMemo(
    () => party.reduce((s, c) => s + Math.max(c.hp, 0), 0),
    [party],
  );

  useEffect(() => {
    const onResize = () => {
      if (!boardRef.current) return;
      cellSizeRef.current = boardRef.current.clientWidth / COLS;
    };
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // ----- drag handling -----
  const getCellFromPoint = (clientX: number, clientY: number) => {
    const el = boardRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const size = rect.width / COLS;
    const c = Math.floor(x / size);
    const r = Math.floor(y / size);
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return null;
    return [r, c] as [number, number];
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (phase !== "idle") return;
    const cell = getCellFromPoint(e.clientX, e.clientY);
    if (!cell) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    dragRef.current.held = cell;
    dragRef.current.lastPos = cell;
    setPhase("dragging");
    setMessage("動かしてルートを作ろう");
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (phase !== "dragging") return;
    const cell = getCellFromPoint(e.clientX, e.clientY);
    if (!cell) return;
    const prev = dragRef.current.lastPos;
    if (!prev) return;
    if (cell[0] === prev[0] && cell[1] === prev[1]) return;
    // Only allow adjacent (including diagonal) moves per step.
    const dr = Math.abs(cell[0] - prev[0]);
    const dc = Math.abs(cell[1] - prev[1]);
    if (dr > 1 || dc > 1) {
      dragRef.current.lastPos = cell;
      return;
    }
    setBoard((b) => swap(b, prev, cell));
    dragRef.current.lastPos = cell;
  };

  const onPointerUp = async () => {
    if (phase !== "dragging") return;
    dragRef.current.held = null;
    dragRef.current.lastPos = null;
    await resolveBoard();
  };

  // ----- resolution loop -----
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const resolveBoard = async () => {
    setPhase("resolving");
    let current = board;
    let combo = 0;
    const allMatches: Match[] = [];

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const matches = findMatches(current);
      if (matches.length === 0) break;
      combo += matches.length;
      allMatches.push(...matches);
      setComboCount(combo);
      const keys = new Set<string>();
      for (const m of matches) for (const [r, c] of m.cells) keys.add(`${r},${c}`);
      setFlash(keys);
      setBoard(current);
      await sleep(350);
      current = removeMatches(current, matches);
      setBoard(current);
      setFlash(null);
      await sleep(200);
      current = applyGravity(current);
      setBoard(current);
      await sleep(300);
    }

    if (allMatches.length === 0) {
      setMessage("マッチなし…");
      setPhase("idle");
      setTimeout(() => setMessage("ドロップを動かそう！"), 800);
      return;
    }

    // Apply damage / heal
    const { totalDamage, heal, combos } = computeDamageAndHeal(
      party,
      allMatches,
      enemy.element,
    );
    setMessage(`${combos} コンボ！ ${totalDamage} ダメージ`);

    const newEnemyHp = Math.max(0, enemy.hp - totalDamage);
    setEnemy({ ...enemy, hp: newEnemyHp });

    if (heal > 0) {
      setParty((p) =>
        p.map((c) => ({
          ...c,
          hp: Math.min(c.maxHp, c.hp + Math.floor(heal / p.length)),
        })),
      );
    }

    await sleep(700);
    setComboCount(0);

    if (newEnemyHp <= 0) {
      setMessage("しょうり！");
      setPhase("win");
      return;
    }

    // Enemy turn
    await enemyTurn();
  };

  const enemyTurn = async () => {
    setPhase("enemyTurn");
    const nextTurn = enemy.turn - 1;
    if (nextTurn > 0) {
      setEnemy({ ...enemy, turn: nextTurn });
      setMessage(`あと ${nextTurn} ターンで攻撃`);
      await sleep(500);
      setPhase("idle");
      setMessage("ドロップを動かそう！");
      return;
    }

    // Attack: split damage across alive members (frontmost takes hit).
    setMessage(`${enemy.name} の攻撃！`);
    await sleep(600);
    const aliveIdx = party.findIndex((c) => c.hp > 0);
    if (aliveIdx >= 0) {
      const updated = party.map((c, i) =>
        i === aliveIdx ? { ...c, hp: Math.max(0, c.hp - enemy.atk) } : c,
      );
      setParty(updated);
      const newTotal = updated.reduce((s, c) => s + Math.max(c.hp, 0), 0);
      if (newTotal <= 0) {
        setEnemy({ ...enemy, turn: enemy.turnMax });
        setMessage("ゲームオーバー…");
        setPhase("lose");
        return;
      }
    }
    setEnemy({ ...enemy, turn: enemy.turnMax });
    await sleep(400);
    setPhase("idle");
    setMessage("ドロップを動かそう！");
  };

  const reset = () => {
    setBoard(createBoard());
    setParty(INITIAL_PARTY.map((c) => ({ ...c, hp: c.maxHp })));
    setEnemy(createEnemy());
    setPhase("idle");
    setMessage("ドロップを動かそう！");
    setComboCount(0);
  };

  const held = dragRef.current.lastPos;

  return (
    <div className="app">
      <header className="top-bar">
        <div className="stage-title">{message}</div>
        {comboCount > 0 && (
          <div className="combo">{comboCount} COMBO</div>
        )}
      </header>

      <div className="enemy-panel">
        <div className="enemy-name">
          {enemy.name}{" "}
          <span
            className="elem-chip"
            style={{ background: ELEMENT_COLOR[enemy.element] }}
          >
            {ELEMENT_LABEL[enemy.element]}
          </span>
        </div>
        <div className="enemy-sprite" aria-hidden>
          🐉
        </div>
        <div className="hp-bar enemy-hp">
          <div
            className="hp-fill"
            style={{
              width: `${(enemy.hp / enemy.maxHp) * 100}%`,
              background: "#ff5555",
            }}
          />
          <span className="hp-text">
            {enemy.hp} / {enemy.maxHp}
          </span>
        </div>
        <div className="enemy-turn">
          攻撃まで: <b>{enemy.turn}</b>
        </div>
      </div>

      <div className="party">
        {party.map((c, i) => (
          <div key={i} className={`member ${c.hp <= 0 ? "dead" : ""}`}>
            <div
              className="portrait"
              style={{ background: c.color }}
              title={c.name}
            >
              <span className="portrait-label">{ELEMENT_LABEL[c.element]}</span>
            </div>
            <div className="hp-bar">
              <div
                className="hp-fill"
                style={{
                  width: `${Math.max(0, (c.hp / c.maxHp) * 100)}%`,
                  background: "#4ed36b",
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="total-hp">
        HP {currentHp} / {totalHp}
      </div>

      <div
        className="board"
        ref={boardRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        style={{
          gridTemplateColumns: `repeat(${COLS}, 1fr)`,
          gridTemplateRows: `repeat(${ROWS}, 1fr)`,
        }}
      >
        {board.map((row, r) =>
          row.map((orb, c) => (
            <OrbCell
              key={orb ? orb.id : `empty-${r}-${c}`}
              orb={orb}
              flashing={flash?.has(`${r},${c}`) ?? false}
              held={
                held ? held[0] === r && held[1] === c && phase === "dragging" : false
              }
            />
          )),
        )}
      </div>

      {(phase === "win" || phase === "lose") && (
        <div className="overlay">
          <div className="overlay-card">
            <h2>{phase === "win" ? "クリア！" : "ゲームオーバー"}</h2>
            <button onClick={reset}>もう一度</button>
          </div>
        </div>
      )}
    </div>
  );
}
