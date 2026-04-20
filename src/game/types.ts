export type Element = "fire" | "water" | "wood" | "light" | "dark" | "heart";

export const ELEMENTS: Element[] = [
  "fire",
  "water",
  "wood",
  "light",
  "dark",
  "heart",
];

export const ELEMENT_LABEL: Record<Element, string> = {
  fire: "火",
  water: "水",
  wood: "木",
  light: "光",
  dark: "闇",
  heart: "回",
};

export const ELEMENT_COLOR: Record<Element, string> = {
  fire: "#ff4d3b",
  water: "#3ab0ff",
  wood: "#4ed36b",
  light: "#ffd54a",
  dark: "#b166ff",
  heart: "#ff78b2",
};

export const ROWS = 5;
export const COLS = 6;

export type Orb = {
  id: number;
  element: Element;
};

export type Board = (Orb | null)[][]; // [row][col]

export type Match = {
  cells: [number, number][]; // (row,col)
  element: Element;
};

export type Character = {
  name: string;
  element: Element;
  subElement?: Element;
  maxHp: number;
  hp: number;
  atk: number;
  rcv: number;
  color: string;
};

export type Enemy = {
  name: string;
  maxHp: number;
  hp: number;
  atk: number;
  turnMax: number;
  turn: number;
  element: Element;
};
