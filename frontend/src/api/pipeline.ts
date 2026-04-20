import type { RunPipelineRequest, RunPipelineResponse } from "@/types/api";

import { apiRequest } from "./http";

export async function runPipeline(body: RunPipelineRequest): Promise<RunPipelineResponse> {
  return apiRequest<RunPipelineResponse>("/api/pipeline/run", {
    method: "POST",
    body,
  });
}
