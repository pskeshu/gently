/**
 * Store context — pipes the Zustand store down to children so they can
 * subscribe to their own slices via `useTuiSelector` and re-render only
 * when their slice changes (not on every state mutation).
 *
 * For composite (object/array) slices that change identity each tick but
 * are shallowly equal, wrap the selector in `useShallow` from
 * `zustand/react/shallow`.
 */

import { createContext, useContext } from "react";
import { useStore } from "zustand";
import type { StoreApi } from "zustand/vanilla";
import type { TuiStore } from "./store.js";

export const StoreContext = createContext<StoreApi<TuiStore> | null>(null);

export function useTuiStoreApi(): StoreApi<TuiStore> {
  const api = useContext(StoreContext);
  if (!api) throw new Error("useTuiStoreApi must be used inside <StoreContext.Provider>");
  return api;
}

export function useTuiSelector<T>(selector: (s: TuiStore) => T): T {
  const api = useTuiStoreApi();
  return useStore(api, selector);
}
