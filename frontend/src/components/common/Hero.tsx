import type { Analysis } from "@/types/api";

import { cn } from "@/lib/utils";

interface HeroProps {
  analysis: Analysis;
  compact?: boolean;
}

export function Hero({ analysis, compact = false }: HeroProps) {
  const net = analysis.scores.net_operating_signal;
  const pillLabel =
    net >= 0
      ? "Growth exceeds risk in this sample"
      : "Risk outweighs growth in this sample";

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border border-border/60 text-primary-foreground shadow-lg",
        compact ? "p-4" : "p-6",
      )}
      style={{
        background:
          "linear-gradient(135deg, hsl(222 47% 11%) 0%, hsl(220 82% 40%) 55%, hsl(150 50% 35%) 100%)",
      }}
    >
      <div className="relative z-10">
        <p className="text-[0.7rem] font-bold uppercase tracking-[0.18em] text-white/80">
          Financial Report Analyzer
        </p>
        <h1
          className={cn(
            "mt-2 font-extrabold leading-tight text-white",
            compact ? "text-xl" : "text-3xl",
          )}
        >
          {analysis.company_name}{" "}
          <span className="font-mono text-white/80">({analysis.ticker})</span>
        </h1>
        <p className="mt-2 text-sm text-white/85">
          {analysis.form_type} · {analysis.filing_date} · {analysis.source}
        </p>
        <span className="mt-4 inline-flex items-center rounded-full border border-white/20 bg-white/15 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
          {pillLabel}
        </span>
      </div>
    </div>
  );
}
