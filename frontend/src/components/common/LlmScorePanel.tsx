import { useMutation } from "@tanstack/react-query";
import { BarChart3, Sparkles, Zap } from "lucide-react";
import { useState } from "react";

import { runLlmPureRescore } from "@/api/signals";
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
import type { LlmPureLabel, LlmPureRescoreResponse, LlmPureSentence } from "@/types/api";

const LABEL_TO_BADGE: Record<LlmPureLabel, "growth" | "risk" | "muted"> = {
  positive: "growth",
  negative: "risk",
  neutral: "muted",
};

export function LlmScorePanel({ label }: { label: string }) {
  const [maxSentences, setMaxSentences] = useState(120);

  const mutation = useMutation<LlmPureRescoreResponse, Error, void>({
    mutationFn: () => runLlmPureRescore(label, maxSentences),
  });

  const data = mutation.data;

  return (
    <Card className="border-primary/30">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-primary/15 p-2 text-primary">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="flex flex-wrap items-center gap-2">
                LLM score
                <Badge variant="outline" className="uppercase">
                  Pure LLM
                </Badge>
                <span className="text-xs font-normal text-muted-foreground">
                  Claude only · {data?.engine ?? "PureLLMSignalEngine"}{" "}
                  {data ? `v${data.engine_version}` : ""}
                </span>
              </CardTitle>
              <CardDescription>
                Classifies <em>every</em> sentence with the standalone LLM engine
                (no FinBERT, no Loughran–McDonald, no signal-engine blend). Uses{" "}
                <code className="mx-1 rounded bg-muted px-1 py-0.5 text-[0.7rem]">
                  ANTHROPIC_API_KEY
                </code>{" "}
                from <code className="rounded bg-muted px-1">.env</code>.
              </CardDescription>
            </div>
          </div>
          <div className="flex items-end gap-2">
            <div className="w-28">
              <Label htmlFor="llm-pure-max-sentences" className="text-xs">
                Max sentences
              </Label>
              <Input
                id="llm-pure-max-sentences"
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
              {mutation.isPending ? "Calling Claude…" : "Run LLM score"}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {!data && !mutation.isPending && !mutation.isError ? (
          <p className="text-sm text-muted-foreground">
            Click <strong>Run LLM score</strong> to score up to the first{" "}
            <code className="rounded bg-muted px-1">max_sentences</code> in this
            document with the pure LLM path only.
          </p>
        ) : null}

        {mutation.isPending ? (
          <Loader label="Scoring with Claude (pure LLM)…" />
        ) : null}

        {mutation.isError ? (
          <ErrorState
            title="LLM score failed"
            error={mutation.error}
            onRetry={() => mutation.mutate()}
          />
        ) : null}

        {data ? <LlmPureResult data={data} /> : null}
      </CardContent>
    </Card>
  );
}

function LlmPureResult({ data }: { data: LlmPureRescoreResponse }) {
  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground">
        Scanned <strong>{data.scanned_sentences}</strong> of{" "}
        <strong>{data.total_sentences}</strong> sentences.
      </div>

      <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 p-4">
        <div className="mb-3 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">LLM document scores (pure engine)</h3>
        </div>
        <p className="mb-4 text-sm text-muted-foreground">
          Averages are over all scored sentences. Separate from the main dashboard
          engine (v3) — not blended with lexicon or FinBERT.
        </p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Growth (LLM)"
            value={formatNumber(data.llm_pure_growth_score)}
            hint={`${data.llm_positive_count} positive-tagged lines`}
            tone="growth"
          />
          <MetricCard
            label="Risk (LLM)"
            value={formatNumber(data.llm_pure_risk_score)}
            hint={`${data.llm_negative_count} negative-tagged lines`}
            tone="risk"
          />
          <MetricCard
            label="Cost pressure (LLM)"
            value={formatNumber(data.llm_pure_cost_score)}
            hint="not split by LLM — typically 0"
            tone="cost"
          />
          <MetricCard
            label="Net (LLM)"
            value={formatSignedNumber(data.llm_pure_net_score)}
            hint="mean(growth) − mean(risk) on LLM lines"
            tone="net"
          />
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-md bg-background/60 p-3 text-center">
            <div className="text-2xl font-bold text-green-500">{data.llm_positive_count}</div>
            <div className="text-xs text-muted-foreground">Positive</div>
          </div>
          <div className="rounded-md bg-background/60 p-3 text-center">
            <div className="text-2xl font-bold text-red-500">{data.llm_negative_count}</div>
            <div className="text-xs text-muted-foreground">Negative</div>
          </div>
          <div className="rounded-md bg-background/60 p-3 text-center">
            <div className="text-2xl font-bold text-muted-foreground">
              {data.llm_neutral_count}
            </div>
            <div className="text-xs text-muted-foreground">Neutral</div>
          </div>
        </div>
      </div>

      <LlmPureSentenceTable sentences={data.sentences} />
    </div>
  );
}

function LlmPureSentenceTable({ sentences }: { sentences: LlmPureSentence[] }) {
  return (
    <div className="max-h-[360px] overflow-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-card/90 backdrop-blur">
          <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-2">Label</th>
            <th className="px-3 py-2">Net</th>
            <th className="px-3 py-2">Sentence</th>
            <th className="px-3 py-2">Reason</th>
          </tr>
        </thead>
        <tbody>
          {sentences.length === 0 ? (
            <tr>
              <td
                colSpan={4}
                className="px-3 py-6 text-center text-xs text-muted-foreground"
              >
                No sentences returned.
              </td>
            </tr>
          ) : (
            sentences.map((s, i) => (
              <tr
                key={`${i}-${s.text.slice(0, 32)}`}
                className="border-b border-border/60 align-top hover:bg-muted/30"
              >
                <td className="px-3 py-2">
                  <Badge variant={LABEL_TO_BADGE[s.label]}>
                    {s.label}
                  </Badge>
                </td>
                <td className="px-3 py-2 font-mono text-xs tabular-nums">
                  {formatSignedNumber(s.net_score)}
                </td>
                <td className="px-3 py-2 text-foreground">{s.text}</td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  {s.reason ?? "—"}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
