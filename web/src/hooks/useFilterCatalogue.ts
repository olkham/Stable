// The smoothing filter catalogue, fetched once for the whole app.
//
// `GET /api/filters` is the server's own registry (stable/smoothing/filters.py)
// with every filter's parameter schema, and it is what both smoothing controls
// build themselves from: the Smoothing panel's four camera channels, and the
// per-track "smooth this path" block on a track card (todo 1, tenth round).
// Nothing about a filter is hardcoded in either, so adding one to the registry
// makes it appear in both and a rename can never desync them.
//
// The request is memoised at module scope rather than per component. A track
// card renders one of these per track, and the catalogue is a constant for the
// lifetime of the server process — asking for it once per open card would be
// one request per track for an answer that cannot have changed.

import { useEffect, useState } from "react";
import * as api from "../api/client";
import type { FilterInfo } from "../types";

let pending: Promise<FilterInfo[]> | null = null;

function load(): Promise<FilterInfo[]> {
  // A failed fetch is not cached: the catalogue is worth asking for again the
  // next time something needs it, rather than leaving the app permanently
  // convinced there are no filters.
  if (!pending) {
    pending = api.getFilters().catch((error) => {
      pending = null;
      throw error;
    });
  }
  return pending;
}

/** The filter catalogue, or an empty list until it arrives (or if it cannot). */
export function useFilterCatalogue(): FilterInfo[] {
  const [catalogue, setCatalogue] = useState<FilterInfo[]>([]);
  useEffect(() => {
    let cancelled = false;
    load()
      .then((rows) => {
        if (!cancelled) setCatalogue(rows);
      })
      .catch(() => {
        if (!cancelled) setCatalogue([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return catalogue;
}
