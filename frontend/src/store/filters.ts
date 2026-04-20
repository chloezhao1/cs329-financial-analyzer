import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { FormType } from "@/types/api";

const CURRENT_YEAR = new Date().getFullYear();

interface FiltersState {
  ticker: string;
  year: number;
  reportTypes: FormType[];
  maxPerType: number;
  selectedLabels: string[];
  sideBySide: boolean;

  setTicker: (ticker: string) => void;
  setYear: (year: number) => void;
  setReportTypes: (types: FormType[]) => void;
  setMaxPerType: (value: number) => void;
  setSelectedLabels: (labels: string[]) => void;
  toggleLabel: (label: string) => void;
  setSideBySide: (value: boolean) => void;
}

export const useFiltersStore = create<FiltersState>()(
  persist(
    (set, get) => ({
      ticker: "AAPL",
      year: CURRENT_YEAR,
      reportTypes: ["10-K", "10-Q"],
      maxPerType: 2,
      selectedLabels: [],
      sideBySide: false,

      setTicker: (ticker) => set({ ticker }),
      setYear: (year) => set({ year }),
      setReportTypes: (reportTypes) => set({ reportTypes }),
      setMaxPerType: (maxPerType) => set({ maxPerType }),
      setSelectedLabels: (selectedLabels) => set({ selectedLabels }),
      toggleLabel: (label) => {
        const current = get().selectedLabels;
        set({
          selectedLabels: current.includes(label)
            ? current.filter((l) => l !== label)
            : [...current, label],
        });
      },
      setSideBySide: (sideBySide) => set({ sideBySide }),
    }),
    { name: "fra-filters" },
  ),
);

export function formatAnalysisLabel(a: {
  ticker: string;
  form_type: string;
  filing_date: string;
}): string {
  return `${a.ticker} | ${a.form_type} | ${a.filing_date}`;
}
