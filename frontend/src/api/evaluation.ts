import type { EvaluationResults } from "@/types/api";

import { apiRequest } from "./http";

export async function fetchLatestEvaluation(): Promise<EvaluationResults> {
  return apiRequest<EvaluationResults>("/api/evaluation/latest");
}

export async function runEvaluation(threshold: number): Promise<EvaluationResults> {
  return apiRequest<EvaluationResults>("/api/evaluation/run", {
    method: "POST",
    body: { threshold },
  });
}
