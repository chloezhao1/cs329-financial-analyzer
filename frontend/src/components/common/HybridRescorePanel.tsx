import { useMutation } from "@tanstack/react-query";
import { BarChart3, Sparkles, Zap } from "lucide-react";
import { useState } from "react";

import { runHybridRescore } from "@/api/signals";
import { ErrorState } from "@/components/common/ErrorState";
import { Loader } from "@/components/common/Loader";
import { MetricCard } from "@/components/common/MetricCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { formatNumber, formatSignedNumber } from "@/lib/utils";
import type { Analysis, HybridLabel, HybridRescoreResponse, HybridSentence } from "@/types/api";

interface HybridRescorePanelProps {
  label: string;
  originalAnalysis?: Analysis;
}

const LABEL_TO_BADGE: Record<HybridLabel, "growth" | "risk" | "muted"> = {
  positive: "growth",
  negative: "risk",
  neutral: "muted",
};

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDelta(hybrid: number, original: number | undefined): string | null {
  if (original === undefined) return null;
  const delta = hybrid - original;
  return `${formatSignedNumber(delta)} vs lexicon`;
}

export function HybridRescorePanel({ label, originalAnalysis }: HybridRescorePanelProps) {
  const [maxSentences, setMaxSentences] = useState(120);
  const [onlyLlm, setOnlyLlm] = useState(true);

  const mutation = useMutation<HybridRescoreResponse, Error, void>({
    mutationFn: () => runHybridRescore(label, maxSentences),
  });

  const data = mutation.data;
  const coverageBoost =
    data != null
      ? Math.max(0, data.hybrid_coverage_rate - data.lexicon_coverage_rate)
      : 0;

  const filteredSentences: HybridSentence[] =
    data?.sentences.filter((s) => (onlyLlm ? s.method === "llm" : true)) ?? [];

  const origScores = originalAnalysis?.scores;

  return (
    <Card className="border-primary/30">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-primary/15 p-2 text-primary">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="flex items-center gap-2">
                Hybrid LLM rescore
                <Badge variant="outline" className="uppercase">
                  Experimental
                </Badge>
              </CardTitle>
              <CardDescription>
                Runs the V2 lexicon scorer first, then sends every sentence
                with <em>zero</em> lexicon hits to Claude for a
                positive / negative / neutral classification. Uses your
                <code className="mx-1 rounded bg-muted px-1 py-0.5 text-[0.7rem]">
                  ANTHROPIC_API_KEY
                </code>
                from <code>.env</code>.
              </CardDescription>
            </div>
          </div>
          <div className="flex items-end gap-2">
            <div className="w-28">
              <Label htmlFor="max-sentences" className="text-xs">
                Max sentences
              </Label>
              <Input
                id="max-sentences"
                type="number"
                min={10}
                max={2000}
                step={10}
                value={maxSentences}
                onChange={(e) =>
                  setMaxSentences(
                    Math.max(10, Math.min(2000, Number(e.target.value) || 120)),
                  )
                }
              />
            </div>
            <Button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
            >
              <Zap className="h-4 w-4" />
              {mutation.isPending ? "Calling Claude…" : "Run hybrid rescore"}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {!data && !mutation.isPending && !mutation.isError ? (
          <p className="text-sm text-muted-foreground">
            Click <strong>Run hybrid rescore</strong> to classify neutral
            sentences with the LLM. Costs one Claude call per batch of ~10
            sentences; the first{" "}
            <code className="rounded bg-muted px-1">max_sentences</code> of
            this document will be scanned.
          </p>
        ) : null}

        {mutation.isPending ? (
          <Loader label="Running V2 lexicon, then batching zero-hit sentences to Claude…" />
        ) : null}

        {mutation.isError ? (
          <ErrorState
            title="Hybrid rescore failed"
            error={mutation.error}
            onRetry={() => mutation.mutate()}
          />
        ) : null}

        {data ? (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="Lexicon coverage"
                value={percent(data.lexicon_coverage_rate)}
                hint={`${data.lexicon_hits} / ${data.scanned_sentences} scanned`}
                tone="neutral"
              />
              <MetricCard
                label="Hybrid coverage"
                value={percent(data.hybrid_coverage_rate)}
                hint={`+${percent(coverageBoost)} vs lexicon`}
                tone="net"
              />
              <MetricCard
                label="LLM reclassified"
                value={`${data.llm_positive + data.llm_negative}`}
                hint={`${data.llm_positive} positive · ${data.llm_negative} negative`}
                tone="growth"
              />
              <MetricCard
                label="LLM left neutral"
                value={`${data.llm_neutral}`}
                hint={`of ${data.llm_fallback} fallback sentences`}
                tone="cost"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-muted-foreground">
                Scanned <strong>{data.scanned_sentences}</strong> of{" "}
                <strong>{data.total_sentences}</strong> sentences in this
                document.
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={onlyLlm}
                  onChange={(e) => setOnlyLlm(e.target.checked)}
                  className="h-3.5 w-3.5 rounded border-border accent-primary"
                />
                Show only LLM-classified sentences
              </label>
            </div>

            <div className="max-h-[460px] overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-card/90 backdrop-blur">
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="px-3 py-2">Method</th>
                    <th className="px-3 py-2">Label</th>
                    <th className="px-3 py-2">Sentence</th>
                    <th className="px-3 py-2">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredSentences.length === 0 ? (
                    <tr>
                      <td
                        colSpan={4}
                        className="px-3 py-6 text-center text-xs text-muted-foreground"
                      >
                        No sentences match the current filter.
                      </td>
                    </tr>
                  ) : (
                    filteredSentences.map((s, i) => (
                      <tr
                        key={`${i}-${s.text.slice(0, 32)}`}
                        className="border-b border-border/60 align-top hover:bg-muted/30"
                      >
                        <td className="px-3 py-2">
                          <Badge
                            variant={s.method === "llm" ? "default" : "muted"}
                            className="uppercase"
                          >
                            {s.method}
                          </Badge>
                        </td>
                        <td className="px-3 py-2">
                          <Badge variant={LABEL_TO_BADGE[s.label]}>
                            {s.label}
                          </Badge>
                        </td>
                        <td className="px-3 py-2 text-foreground">{s.text}</td>
                        <td className="px-3 py-2 text-xs text-muted-foreground">
                          {s.llm_reason ?? "—"}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* LLM-enhanced document scores section */}
            <div className="mt-6 rounded-lg border border-primary/20 bg-primary/5 p-4">
              <div className="mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">LLM-enhanced document scores</h3>
              </div>
              <p className="mb-4 text-sm text-muted-foreground">
                Aggregated scores from the hybrid classification (lexicon + LLM fallback).
                These incorporate the LLM's classifications for sentences that had zero lexicon hits.
              </p>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <MetricCard
                  label="Growth (hybrid)"
                  value={formatNumber(data.hybrid_growth_score)}
                  hint={formatDelta(data.hybrid_growth_score, origScores?.growth) ?? `${data.hybrid_positive_count} positive sentences`}
                  tone="growth"
                />
                <MetricCard
                  label="Risk (hybrid)"
                  value={formatNumber(data.hybrid_risk_score)}
                  hint={formatDelta(data.hybrid_risk_score, origScores?.risk) ?? `${data.hybrid_negative_count} negative sentences`}
                  tone="risk"
                />
                <MetricCard
                  label="Cost pressure (hybrid)"
                  value={formatNumber(data.hybrid_cost_score)}
                  hint={formatDelta(data.hybrid_cost_score, origScores?.cost_pressure) ?? "from lexicon scoring"}
                  tone="cost"
                />
                <MetricCard
                  label="Net signal (hybrid)"
                  value={formatSignedNumber(data.hybrid_net_score)}
                  hint={formatDelta(data.hybrid_net_score, origScores?.net_operating_signal) ?? "growth − risk"}
                  tone="net"
                />
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className="rounded-md bg-background/60 p-3 text-center">
                  <div className="text-2xl font-bold text-green-500">{data.hybrid_positive_count}</div>
                  <div className="text-xs text-muted-foreground">Positive sentences</div>
                </div>
                <div className="rounded-md bg-background/60 p-3 text-center">
                  <div className="text-2xl font-bold text-red-500">{data.hybrid_negative_count}</div>
                  <div className="text-xs text-muted-foreground">Negative sentences</div>
                </div>
                <div className="rounded-md bg-background/60 p-3 text-center">
                  <div className="text-2xl font-bold text-muted-foreground">{data.hybrid_neutral_count}</div>
                  <div className="text-xs text-muted-foreground">Neutral sentences</div>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
