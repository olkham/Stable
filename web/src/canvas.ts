// Sizing a canvas for the device's pixel ratio, without paying for it twice.

/**
 * Give a canvas a backing store of `cssWidth × cssHeight` CSS pixels at `dpr`.
 *
 * Only touches the canvas when the size actually changed. Assigning `width` or
 * `height` — even the value it already has — discards the backing store and
 * allocates a new one, and the canvases that call this redraw on every frame
 * of playback.
 *
 * Returns whether the canvas was resized (its context state is reset if so).
 */
export function fitCanvas(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): boolean {
  const width = Math.max(1, Math.floor(cssWidth * dpr));
  const height = Math.max(1, Math.floor(cssHeight * dpr));
  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  return true;
}
