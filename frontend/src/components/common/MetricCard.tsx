import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export type MetricTone = "growth" | "risk" | "cost" | "net" | "neutral";

interface MetricCardProps {
  label: string;
  value: string;
  hint?: ReactNode;
  tone?: MetricTone;
  icon?: LucideIcon;
  compact?: boolean;
  className?: string;
}

const TONE_BG: Record<MetricTone, string> = {
  growth: "from-[hsl(var(--growth)/0.15)] to-transparent",
  risk: "from-[hsl(var(--risk)/0.15)] to-transparent",
  cost: "from-[hsl(var(--cost)/0.15)] to-transparent",
  net: "from-[hsl(var(--net)/0.15)] to-transparent",
  neutral: "from-muted/50 to-transparent",
};

const TONE_ACCENT: Record<MetricTone, string> = {
  growth: "text-[hsl(var(--growth))]",
  risk: "text-[hsl(var(--risk))]",
  cost: "text-[hsl(var(--cost))]",
  net: "text-[hsl(var(--net))]",
  neutral: "text-foreground",
};

export function MetricCard({
  label,
  value,
  hint,
  tone = "neutral",
  icon: Icon,
  compact = false,
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-border bg-card/60 backdrop-blur",
        compact ? "p-3" : "p-4",
        className,
      )}
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-0 bg-gradient-to-br opacity-90",
          TONE_BG[tone],
        )}
        aria-hidden="true"
      />
      <div className="relative flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-[0.7rem] font-semibold uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          <p
            className={cn(
              "mt-1 font-semibold tabular-nums",
              TONE_ACCENT[tone],
              compact ? "text-lg" : "text-2xl",
            )}
          >
            {value}
          </p>
          {hint ? (
            <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
          ) : null}
        </div>
        {Icon ? (
          <div className="rounded-lg bg-background/60 p-1.5 text-muted-foreground">
            <Icon className="h-4 w-4" />
          </div>
        ) : null}
      </div>
    </div>
  );
}
