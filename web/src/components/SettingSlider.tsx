import type { TrackerSetting } from "../types";

/** The radius in source pixels a setting's value corresponds to. */
export function radiusOf(setting: TrackerSetting, value: number): number {
  if (setting.unit === "side") return Math.floor(value / 2);
  if (setting.unit === "radius") return value;
  return setting.radius; // not a length (pyramid depth); its own radius stands
}

/**
 * How a value reads to the operator: in pixels they can measure on the frame.
 *
 * A pattern is stated as the square it covers, because that is the box drawn on
 * the canvas; a search area as the distance the point may travel from where it
 * was, because that is the question being answered ("does it move more than
 * this between frames?"). A pyramid depth is neither, so it is stated as
 * itself.
 */
export function describeSetting(setting: TrackerSetting, value: number): string {
  if (setting.unit === "count") return `${value} levels`;
  if (setting.role === "search") return `±${radiusOf(setting, value)}px`;
  const side = setting.unit === "side" ? value : value * 2 + 1;
  return `${side}px square`;
}

/**
 * One tracker setting as a labelled slider with its value in pixels.
 *
 * Shared by the patch panel (settings of an existing track) and the new-track
 * card (settings of a track that does not exist yet), so the two can never
 * disagree about what a value means. `extra` is drawn after the value for the
 * caller's own marks — a "preview" tag, a reset button.
 */
export function SettingSlider({
  setting,
  value,
  disabled,
  onChange,
  extra,
}: {
  setting: TrackerSetting;
  value: number;
  disabled?: boolean;
  onChange: (value: number) => void;
  extra?: React.ReactNode;
}) {
  return (
    <label className="patch-setting" title={setting.help}>
      <span className="patch-setting__label">{setting.label}</span>
      <input
        type="range"
        min={setting.minimum}
        max={setting.maximum}
        step={setting.step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="patch-setting__value">
        {describeSetting(setting, value)}
        {extra}
      </span>
    </label>
  );
}
