import type {
  Analysis,
  BaselineStats,
  ComparisonRow,
  DataSource,
  HybridRescoreResponse,
} from "@/types/api";

import { apiRequest } from "./http";

export async function fetchAnalyses(refresh = false): Promise<Analysis[]> {
  return apiRequest<Analysis[]>("/api/signals/analyses", { query: { refresh: refresh ? 1 : 0 } });
}

export async function fetchDataSource(): Promise<{ data_source: DataSource }> {
  return apiRequest<{ data_source: DataSource }>("/api/signals/data-source");
}

export async function fetchComparison(labels: string[]): Promise<ComparisonRow[]> {
  return apiRequest<ComparisonRow[]>("/api/signals/comparison", {
    method: "POST",
    body: { labels },
  });
}

export async function fetchBaseline(): Promise<BaselineStats | null> {
  return apiRequest<BaselineStats | null>("/api/signals/baseline");
}

export async function runHybridRescore(
  label: string,
  maxSentences = 120,
): Promise<HybridRescoreResponse> {
  return apiRequest<HybridRescoreResponse>("/api/signals/hybrid", {
    method: "POST",
    body: { label, max_sentences: maxSentences },
  });
}
