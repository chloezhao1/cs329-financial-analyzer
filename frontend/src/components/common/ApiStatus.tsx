import { useQuery } from "@tanstack/react-query";
import { CircleDashed, ShieldAlert, ShieldCheck } from "lucide-react";

import { apiRequest, API_BASE } from "@/api/http";
import { cn } from "@/lib/utils";

interface HealthResponse {
  status: string;
  service: string;
  project_root: string;
}

export function ApiStatus() {
  const q = useQuery({
    queryKey: ["health"],
    queryFn: () => apiRequest<HealthResponse>("/api/health"),
    refetchInterval: 30_000,
    retry: 0,
  });

  const state: "loading" | "ok" | "down" = q.isLoading
    ? "loading"
    : q.data?.status === "ok"
      ? "ok"
      : "down";

  const Icon =
    state === "loading" ? CircleDashed : state === "ok" ? ShieldCheck : ShieldAlert;
  const label =
    state === "loading" ? "Contacting API…" : state === "ok" ? "API online" : "API offline";
  const tone =
    state === "ok"
      ? "border-[hsl(var(--growth)/0.4)] text-[hsl(var(--growth))]"
      : state === "down"
        ? "border-destructive/50 text-destructive"
        : "border-border text-muted-foreground";

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 rounded-full border bg-card/60 px-3 py-1 text-xs",
        tone,
      )}
      title={API_BASE || "same origin (Vite /api → FastAPI in dev)"}
    >
      <Icon className={cn("h-3.5 w-3.5", state === "loading" && "animate-spin")} />
      <span className="font-medium">{label}</span>
      <span className="hidden font-mono text-[0.65rem] text-muted-foreground md:inline">
        {API_BASE || "/api (proxied)"}
      </span>
    </div>
  );
}
