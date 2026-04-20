import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { fetchAnalyses } from "@/api/signals";
import { formatAnalysisLabel } from "@/store/filters";
import type { Analysis } from "@/types/api";

export interface AnalysesBundle {
  analyses: Analysis[];
  labels: string[];
  byLabel: Map<string, Analysis>;
}

export function useAnalyses() {
  const query = useQuery<Analysis[]>({
    queryKey: ["analyses"],
    queryFn: () => fetchAnalyses(false),
  });

  const bundle = useMemo<AnalysesBundle>(() => {
    const analyses = query.data ?? [];
    const labels = analyses.map(formatAnalysisLabel);
    const byLabel = new Map(analyses.map((a) => [formatAnalysisLabel(a), a]));
    return { analyses, labels, byLabel };
  }, [query.data]);

  return { ...query, ...bundle };
}
