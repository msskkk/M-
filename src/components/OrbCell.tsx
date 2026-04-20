import { ELEMENT_COLOR, ELEMENT_LABEL, Orb } from "../game/types";

type Props = {
  orb: Orb | null;
  flashing: boolean;
  held: boolean;
};

export function OrbCell({ orb, flashing, held }: Props) {
  if (!orb) {
    return <div className="cell empty" />;
  }
  return (
    <div
      className={`cell ${flashing ? "flash" : ""} ${held ? "held" : ""}`}
      data-element={orb.element}
    >
      <div
        className="orb"
        style={{
          background: `radial-gradient(circle at 35% 30%, #ffffffcc 0%, ${ELEMENT_COLOR[orb.element]} 55%, #00000055 100%)`,
        }}
      >
        <span className="orb-label">{ELEMENT_LABEL[orb.element]}</span>
      </div>
    </div>
  );
}
