import type { ColumnDef } from "@tanstack/react-table";
import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { fetchComparison } from "@/api/signals";
import { SignalBarChart } from "@/components/charts/SignalBarChart";
import { AnalysisDetail } from "@/components/common/AnalysisDetail";
import { DataTable } from "@/components/common/DataTable";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { Loader } from "@/components/common/Loader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAnalyses } from "@/hooks/useAnalyses";
import { formatNumber, formatSignedNumber } from "@/lib/utils";
import { useFiltersStore } from "@/store/filters";
import type { ComparisonRow } from "@/types/api";

export function ComparePage() {
  const analysesQuery = useAnalyses();
  const { analyses, labels, byLabel } = analysesQuery;
  const selectedLabels = useFiltersStore((s) => s.selectedLabels);
  const toggleLabel = useFiltersStore((s) => s.toggleLabel);
  const setSelectedLabels = useFiltersStore((s) => s.setSelectedLabels);
  const sideBySide = useFiltersStore((s) => s.sideBySide);
  const setSideBySide = useFiltersStore((s) => s.setSideBySide);

  const activeLabels = useMemo(
    () => selectedLabels.filter((l) => labels.includes(l)),
    [selectedLabels, labels],
  );

  const comparisonQuery = useQuery<ComparisonRow[]>({
    queryKey: ["comparison", activeLabels],
    queryFn: () => fetchComparison(activeLabels),
    enabled: activeLabels.length >= 1,
  });

  const columns = useMemo<ColumnDef<ComparisonRow>[]>(
    () => [
      { accessorKey: "label", header: "Document" },
      {
        accessorKey: "growth",
        header: "Growth",
        cell: (ctx) => (
          <span className="font-mono text-[hsl(var(--growth))]">
            {formatNumber(ctx.getValue<number>())}
          </span>
        ),
      },
      {
        accessorKey: "risk",
        header: "Risk",
        cell: (ctx) => (
          <span className="font-mono text-[hsl(var(--risk))]">
            {formatNumber(ctx.getValue<number>())}
          </span>
        ),
      },
      {
        accessorKey: "net_operating_signal",
        header: "Net Signal",
        cell: (ctx) => {
          const v = ctx.getValue<number>();
          return (
            <span
              className={`font-mono font-semibold ${v >= 0 ? "text-[hsl(var(--growth))]" : "text-[hsl(var(--risk))]"}`}
            >
              {formatSignedNumber(v)}
            </span>
          );
        },
      },
      {
        accessorKey: "scored_sentences",
        header: "Sentences",
        cell: (ctx) => (
          <span className="font-mono tabular-nums">{ctx.getValue<number>()}</span>
        ),
      },
    ],
    [],
  );

  if (analysesQuery.isLoading) return <Loader />;
  if (analysesQuery.isError)
    return (
      <ErrorState
        title="Could not load analyses"
        error={analysesQuery.error}
        onRetry={() => analysesQuery.refetch()}
      />
    );

  if (analyses.length === 0) {
    return (
      <EmptyState
        title="Nothing to compare yet"
        description="Run the pipeline first to populate the analysis set."
      />
    );
  }

  const sideBySideCandidates = activeLabels
    .map((l) => byLabel.get(l))
    .filter((x): x is NonNullable<typeof x> => Boolean(x));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Compare reports</h2>
          <p className="text-sm text-muted-foreground">
            Select two or more documents to chart their signals side by side.
          </p>
        </div>
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card/50 px-3 py-2">
          <Label htmlFor="side-by-side">Side-by-side detail</Label>
          <Switch
            id="side-by-side"
            checked={sideBySide}
            onCheckedChange={setSideBySide}
          />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Documents</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-3 flex items-center gap-2 text-xs">
            <button
              type="button"
              className="rounded-md border border-border px-2 py-1 text-muted-foreground hover:text-foreground"
              onClick={() => setSelectedLabels(labels)}
            >
              Select all
            </button>
            <button
              type="button"
              className="rounded-md border border-border px-2 py-1 text-muted-foreground hover:text-foreground"
              onClick={() => setSelectedLabels([])}
            >
              Clear
            </button>
            <span className="text-muted-foreground">
              {activeLabels.length} of {labels.length} selected
            </span>
          </div>
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {analyses.map((a) => {
              const label = `${a.ticker} | ${a.form_type} | ${a.filing_date}`;
              const checked = activeLabels.includes(label);
              return (
                <label
                  key={label}
                  className="flex cursor-pointer items-start gap-3 rounded-lg border border-border bg-card/40 px-3 py-2 hover:bg-muted/40"
                >
                  <Checkbox
                    checked={checked}
                    onChange={() => toggleLabel(label)}
                    className="mt-1"
                  />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold">
                      {a.ticker}{" "}
                      <span className="font-normal text-muted-foreground">
                        · {a.form_type}
                      </span>
                    </p>
                    <p className="truncate text-xs text-muted-foreground">
                      {a.filing_date} · {a.source}
                    </p>
                  </div>
                </label>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {activeLabels.length === 0 ? (
        <EmptyState
          title="Pick at least one document"
          description="Select documents above to see the comparison table and chart."
        />
      ) : comparisonQuery.isLoading ? (
        <Loader label="Building comparison…" />
      ) : comparisonQuery.isError ? (
        <ErrorState
          title="Could not build comparison"
          error={comparisonQuery.error}
          onRetry={() => comparisonQuery.refetch()}
        />
      ) : (
        <Tabs defaultValue="chart">
          <TabsList>
            <TabsTrigger value="chart">Chart</TabsTrigger>
            <TabsTrigger value="table">Table</TabsTrigger>
          </TabsList>

          <TabsContent value="chart">
            <Card>
              <CardHeader>
                <CardTitle>Raw signals</CardTitle>
              </CardHeader>
              <CardContent>
                <SignalBarChart rows={comparisonQuery.data ?? []} mode="raw" />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="table">
            <DataTable
              columns={columns}
              data={comparisonQuery.data ?? []}
              emptyLabel="No comparison rows yet."
            />
          </TabsContent>
        </Tabs>
      )}

      {sideBySide && sideBySideCandidates.length >= 2 ? (
        <div className="grid gap-6 xl:grid-cols-2">
          {sideBySideCandidates.slice(0, 2).map((a) => (
            <div key={`${a.ticker}-${a.filing_date}-${a.form_type}`}>
              <AnalysisDetail analysis={a} compact />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
