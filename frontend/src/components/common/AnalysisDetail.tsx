import type { ColumnDef } from "@tanstack/react-table";
import { ArrowDownRight, ArrowUpRight, ShieldAlert, TrendingUp } from "lucide-react";
import { useMemo } from "react";

import { SectionBreakdown } from "@/components/charts/SectionBreakdown";
import { DataTable } from "@/components/common/DataTable";
import { Hero } from "@/components/common/Hero";
import { MetricCard, type MetricTone } from "@/components/common/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber, formatSignedNumber, truncate } from "@/lib/utils";
import type { Analysis, PhraseHit, SentenceScore } from "@/types/api";

interface AnalysisDetailProps {
  analysis: Analysis;
  compact?: boolean;
}

function scoreTone(value: number, kind: "growth" | "risk" | "cost" | "net"): MetricTone {
  if (kind === "growth" && value >= 0) return "growth";
  if (kind === "risk" && value >= 0) return "risk";
  if (kind === "cost" && value >= 0) return "cost";
  if (kind === "net") return value >= 0 ? "growth" : "risk";
  return "neutral";
}

export function AnalysisDetail({ analysis, compact = false }: AnalysisDetailProps) {
  const { scores, coverage, zscores } = analysis;

  const sentenceColumns = useMemo<ColumnDef<SentenceScore>[]>(
    () => [
      {
        accessorKey: "section",
        header: "Section",
        cell: (ctx) => (
          <Badge variant="muted" className="font-mono">
            {ctx.getValue<string>()}
          </Badge>
        ),
      },
      {
        accessorKey: "text",
        header: "Sentence",
        cell: (ctx) => (
          <span className="block max-w-[680px] text-sm leading-relaxed text-foreground">
            {truncate(ctx.getValue<string>(), 240)}
          </span>
        ),
      },
      {
        accessorKey: "net_score",
        header: "Net",
        cell: (ctx) => {
          const v = ctx.getValue<number>();
          const Icon = v >= 0 ? ArrowUpRight : ArrowDownRight;
          return (
            <span
              className={`inline-flex items-center gap-1 font-mono ${v >= 0 ? "text-[hsl(var(--growth))]" : "text-[hsl(var(--risk))]"}`}
            >
              <Icon className="h-3 w-3" />
              {formatSignedNumber(v)}
            </span>
          );
        },
      },
      {
        accessorKey: "cost_pressure",
        header: "Cost",
        cell: (ctx) => (
          <span className="font-mono text-[hsl(var(--cost))]">
            {formatSignedNumber(ctx.getValue<number>())}
          </span>
        ),
      },
    ],
    [],
  );

  const phraseColumns = useMemo<ColumnDef<PhraseHit>[]>(
    () => [
      { accessorKey: "term", header: "Term" },
      {
        accessorKey: "source",
        header: "Source",
        cell: (ctx) => (
          <Badge variant={ctx.getValue<string>() === "lm_word" ? "outline" : "default"}>
            {ctx.getValue<string>()}
          </Badge>
        ),
      },
      {
        accessorKey: "count",
        header: "Count",
        cell: (ctx) => (
          <span className="font-mono tabular-nums">{ctx.getValue<number>()}</span>
        ),
      },
    ],
    [],
  );

  const metricClass = compact
    ? "grid grid-cols-2 gap-3"
    : "grid grid-cols-2 gap-4 xl:grid-cols-4";

  const engineTitle =
    analysis.method?.engine_id === "v3"
      ? "Engine score (v3)"
      : "Engine scores";

  return (
    <div className="space-y-6">
      <Hero analysis={analysis} compact={compact} />

      <p className="text-sm font-medium text-foreground">
        {engineTitle}
        <span className="ml-2 text-xs font-normal text-muted-foreground">
          {analysis.method?.signal_engine
            ? `(${analysis.method.signal_engine})`
            : "document-level means · separate from the LLM-only block below"}
        </span>
      </p>

      <div className={metricClass}>
        <MetricCard
          label="Growth"
          value={formatNumber(scores.growth)}
          tone={scoreTone(scores.growth, "growth")}
          hint={zscores ? `z = ${formatSignedNumber(zscores.growth)}` : undefined}
          icon={TrendingUp}
          compact={compact}
        />
        <MetricCard
          label="Risk"
          value={formatNumber(scores.risk)}
          tone={scoreTone(scores.risk, "risk")}
          hint={zscores ? `z = ${formatSignedNumber(zscores.risk)}` : undefined}
          icon={ShieldAlert}
          compact={compact}
        />
        <MetricCard
          label="Cost Pressure"
          value={formatNumber(scores.cost_pressure)}
          tone={scoreTone(scores.cost_pressure, "cost")}
          compact={compact}
        />
        <MetricCard
          label="Net Signal"
          value={formatSignedNumber(scores.net_operating_signal)}
          tone={scoreTone(scores.net_operating_signal, "net")}
          hint={zscores ? `z = ${formatSignedNumber(zscores.net_operating_signal)}` : undefined}
          compact={compact}
        />
      </div>

      {zscores ? (
        <Card>
          <CardHeader>
            <CardTitle>Sector baseline</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-2 text-sm">
            <Badge variant={zscores.is_sector_specific ? "default" : "muted"}>
              {zscores.reference_label}
            </Badge>
            <span className="text-muted-foreground">n = {zscores.reference_n}</span>
            {!zscores.reference_reliable ? (
              <Badge variant="outline" className="text-muted-foreground">
                sample may be too small
              </Badge>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>Top signal sentences</CardTitle>
          </CardHeader>
          <CardContent>
            <DataTable
              columns={sentenceColumns}
              data={analysis.top_sentences}
              emptyLabel="No sentences were scored for this document."
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Sentences by section</CardTitle>
          </CardHeader>
          <CardContent>
            <SectionBreakdown sectionsByCount={coverage.sentences_by_section} />
            <p className="mt-2 text-xs text-muted-foreground">
              {coverage.scored_sentences} scored · {coverage.scored_with_hits} with
              explicit signal hits
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {[
          { title: "Top growth phrases", rows: analysis.top_growth_phrases },
          { title: "Top risk phrases", rows: analysis.top_risk_phrases },
          { title: "Top cost phrases", rows: analysis.top_cost_phrases },
        ].map((block) => (
          <Card key={block.title}>
            <CardHeader>
              <CardTitle>{block.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <DataTable
                columns={phraseColumns}
                data={block.rows}
                emptyLabel="No phrase hits."
              />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
