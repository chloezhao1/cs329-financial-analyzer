import { useMutation, useQueryClient } from "@tanstack/react-query";
import { PlayCircle } from "lucide-react";
import { useState } from "react";

import { runPipeline } from "@/api/pipeline";
import { ErrorState } from "@/components/common/ErrorState";
import { Loader } from "@/components/common/Loader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useFiltersStore } from "@/store/filters";
import type { FormType, RunPipelineResponse } from "@/types/api";

const CHOICES: FormType[] = ["10-K", "10-Q", "EARNINGS_CALL"];

/** Allowed filing year range for the pipeline date window (inclusive). */
const MIN_PIPELINE_YEAR = 2022;
const MAX_PIPELINE_YEAR = 2026;

/** Mirrors backend when the pipeline collects nothing (invalid ticker, empty date range, etc.). */
const PIPELINE_EMPTY_MESSAGE =
  "No filings or transcripts were collected. Check that tickers are valid SEC symbols, your date range includes filing dates, and the selected form types are not all skipped (e.g. transcripts require enabling the earnings-call option when applicable).";

function clampPipelineYear(y: number): number {
  return Math.min(MAX_PIPELINE_YEAR, Math.max(MIN_PIPELINE_YEAR, y));
}

export function FetchPage() {
  const queryClient = useQueryClient();
  const tickerStored = useFiltersStore((s) => s.ticker);
  const setTicker = useFiltersStore((s) => s.setTicker);
  const reportTypes = useFiltersStore((s) => s.reportTypes);
  const setReportTypes = useFiltersStore((s) => s.setReportTypes);
  const maxPerType = useFiltersStore((s) => s.maxPerType);
  const setMaxPerType = useFiltersStore((s) => s.setMaxPerType);

  const currentYear = new Date().getFullYear();
  const [tickersText, setTickersText] = useState(tickerStored);
  const [skipTranscripts, setSkipTranscripts] = useState(true);
  const [startYear, setStartYear] = useState<string>(
    String(clampPipelineYear(currentYear - 1)),
  );
  const [endYear, setEndYear] = useState<string>(
    String(clampPipelineYear(currentYear)),
  );

  const toYearStart = (year: string): string | undefined => {
    const n = Number(year);
    if (!Number.isFinite(n) || n < MIN_PIPELINE_YEAR || n > MAX_PIPELINE_YEAR) {
      return undefined;
    }
    return `${n}-01-01`;
  };
  const toYearEnd = (year: string): string | undefined => {
    const n = Number(year);
    if (!Number.isFinite(n) || n < MIN_PIPELINE_YEAR || n > MAX_PIPELINE_YEAR) {
      return undefined;
    }
    return `${n}-12-31`;
  };

  const mutation = useMutation<RunPipelineResponse, Error, void>({
    mutationFn: async () => {
      const tickers = tickersText
        .split(/[,\s]+/)
        .map((t) => t.trim().toUpperCase())
        .filter(Boolean);
      if (tickers.length === 0) {
        throw new Error("Enter at least one ticker.");
      }
      const start = Number(startYear);
      const end = Number(endYear);
      if (!Number.isInteger(start) || start < MIN_PIPELINE_YEAR || start > MAX_PIPELINE_YEAR) {
        throw new Error(
          `Start year must be a whole number between ${MIN_PIPELINE_YEAR} and ${MAX_PIPELINE_YEAR}.`,
        );
      }
      if (!Number.isInteger(end) || end < MIN_PIPELINE_YEAR || end > MAX_PIPELINE_YEAR) {
        throw new Error(
          `End year must be a whole number between ${MIN_PIPELINE_YEAR} and ${MAX_PIPELINE_YEAR}.`,
        );
      }
      if (start > end) {
        throw new Error("Start year cannot be after end year.");
      }
      const data = await runPipeline({
        tickers,
        form_types: reportTypes,
        max_per_type: maxPerType,
        skip_transcripts: skipTranscripts,
        start_date: toYearStart(startYear),
        end_date: toYearEnd(endYear),
      });
      // Defensive: treat 0 records as failure even if an older API still returns 200.
      if (data.n_records === 0) {
        throw new Error(PIPELINE_EMPTY_MESSAGE);
      }
      return data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
      queryClient.invalidateQueries({ queryKey: ["data-source"] });
    },
  });

  const toggleFormType = (t: FormType): void => {
    setReportTypes(
      reportTypes.includes(t)
        ? reportTypes.filter((x) => x !== t)
        : [...reportTypes, t],
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Fetch &amp; analyze</h2>
        <p className="text-sm text-muted-foreground">
          Runs the full scraping and preprocessing pipeline. This can take several
          minutes depending on filing sizes.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline configuration</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="tickers">Tickers</Label>
            <Input
              id="tickers"
              placeholder="AAPL, MSFT, NVDA"
              value={tickersText}
              onChange={(e) => {
                setTickersText(e.target.value);
                setTicker(e.target.value);
              }}
            />
            <p className="text-xs text-muted-foreground">Separate with commas or spaces.</p>
          </div>

          <div className="space-y-2">
            <Label>Form types</Label>
            <div className="flex flex-wrap gap-2">
              {CHOICES.map((c) => {
                const active = reportTypes.includes(c);
                return (
                  <button
                    key={c}
                    type="button"
                    onClick={() => toggleFormType(c)}
                    className={`rounded-full border px-3 py-1 text-xs font-semibold transition ${
                      active
                        ? "border-primary bg-primary/20 text-primary"
                        : "border-border bg-card/40 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {c}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="max-per-type">Max documents per form type</Label>
            <Input
              id="max-per-type"
              type="number"
              min={1}
              max={8}
              value={maxPerType}
              onChange={(e) => setMaxPerType(Math.max(1, Math.min(8, Number(e.target.value))))}
            />
          </div>

          <div className="flex items-end gap-3">
            <div className="flex flex-1 items-center gap-3 rounded-lg border border-border bg-card/40 px-3 py-2">
              <Switch
                id="skip-transcripts"
                checked={skipTranscripts}
                onCheckedChange={setSkipTranscripts}
              />
              <div>
                <Label htmlFor="skip-transcripts">Skip earnings transcripts</Label>
                <p className="text-[0.7rem] text-muted-foreground">
                  Recommended unless Selenium/Chrome are configured.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="start-year">Start year</Label>
            <Input
              id="start-year"
              type="number"
              min={MIN_PIPELINE_YEAR}
              max={MAX_PIPELINE_YEAR}
              step={1}
              placeholder={String(clampPipelineYear(currentYear - 1))}
              value={startYear}
              onChange={(e) => setStartYear(e.target.value)}
            />
            <p className="text-[0.7rem] text-muted-foreground">
              Allowed range: {MIN_PIPELINE_YEAR}–{MAX_PIPELINE_YEAR}. Filings from Jan 1 of this
              year onward.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="end-year">End year</Label>
            <Input
              id="end-year"
              type="number"
              min={MIN_PIPELINE_YEAR}
              max={MAX_PIPELINE_YEAR}
              step={1}
              placeholder={String(clampPipelineYear(currentYear))}
              value={endYear}
              onChange={(e) => setEndYear(e.target.value)}
            />
            <p className="text-[0.7rem] text-muted-foreground">
              Allowed range: {MIN_PIPELINE_YEAR}–{MAX_PIPELINE_YEAR}. Filings up to Dec 31 of this
              year.
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center gap-3">
        <Button
          size="lg"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
        >
          <PlayCircle className="h-4 w-4" />
          {mutation.isPending ? "Running pipeline…" : "Run pipeline"}
        </Button>
        {mutation.isPending ? (
          <span className="text-sm text-muted-foreground">
            Streaming filings · preprocessing · scoring…
          </span>
        ) : null}
      </div>

      {mutation.isPending ? <Loader label="Pipeline in flight — this can take a while." /> : null}

      {mutation.isError ? (
        <ErrorState
          title="Pipeline failed"
          error={mutation.error}
          onRetry={() => mutation.mutate()}
        />
      ) : null}

      {mutation.isSuccess && mutation.data ? (
        <Card>
          <CardHeader>
            <CardTitle>Pipeline run complete</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant="default">{mutation.data.n_records} records</Badge>
              <Badge variant="outline">{mutation.data.n_analyses} analyses</Badge>
              <Badge variant="muted">
                tickers: {mutation.data.tickers.join(", ") || "–"}
              </Badge>
              <Badge variant="muted">
                forms: {mutation.data.form_types.join(", ") || "–"}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              The dashboard has been refreshed — head to Dashboard or Compare to inspect
              the new results.
            </p>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
