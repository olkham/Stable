// Which key presses belong to the focused control, and which are the app's.
//
// Every shortcut here (frame stepping, play, the tools, In/Out, anchors, undo,
// zoom, Delete) is a window-level listener, so it sees every key — including
// the ones meant for whatever has focus. Each listener used to step aside for
// *any* focused button or input, and a click leaves focus on the control it
// clicked: after pressing Play, picking a tool, ticking "Overlays" or a track's
// ◀/▶, the arrows, I/O, A, V/N/D/H and Delete all silently did nothing until
// the canvas was clicked. A control owns the keys it actually uses and no
// others, so that is all a shortcut defers to.

/** Input types where every key is typing — the shortcut letters included. */
const TEXT_INPUT_TYPES = new Set([
  "text",
  "search",
  "email",
  "url",
  "tel",
  "password",
  "number",
  "date",
  "datetime-local",
  "month",
  "time",
  "week",
]);

/** Keys a focused slider moves itself with. */
const SLIDER_KEYS = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  "Home",
  "End",
  "PageUp",
  "PageDown",
]);

/** Keys a focused radio button moves through its group with. */
const RADIO_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

/** Keys that press a button, tick a box or open a picker. */
const ACTIVATION_KEYS = new Set([" ", "Enter"]);

/**
 * Whether the focused element will act on this key itself.
 *
 * True for every key in a text field, a text area, a select (whose arrows
 * change its value) or editable content; for the movement keys on a slider or
 * a radio; and for Space/Enter on a button, a checkbox or any other control
 * they activate — answering those here as well would, for instance, toggle a
 * focused Play button twice. Everything else is the app's.
 */
export function keyBelongsToControl(event: KeyboardEvent): boolean {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  switch (target.tagName) {
    case "TEXTAREA":
    case "SELECT":
      return true;
    case "INPUT": {
      const type = (target as HTMLInputElement).type;
      if (TEXT_INPUT_TYPES.has(type)) return true;
      if (type === "range") return SLIDER_KEYS.has(event.key);
      if (type === "radio" && RADIO_KEYS.has(event.key)) return true;
      return ACTIVATION_KEYS.has(event.key);
    }
    case "BUTTON":
    case "SUMMARY":
      return ACTIVATION_KEYS.has(event.key);
    case "A":
      return event.key === "Enter" && (target as HTMLAnchorElement).href !== "";
    default:
      return false;
  }
}
