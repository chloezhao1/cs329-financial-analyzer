import { useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo } from "react";

import { fetchBaseline } from "@/api/signals";
import { DataTable } from "@/components/common/DataTable";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { Loader } from "@/components/common/Loader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumber } from "@/lib/utils";
import type { BaselineSectorStats, BaselineStats } from "@/types/api";

interface SectorRow {
  sector: string;
  n: number;
  reliable: boolean;
  growth_mean: number;
  growth_stdev: number;
  risk_mean: number;
  risk_stdev: number;
  tickers: string[];
}

function flatten(stats: BaselineStats): SectorRow[] {
  const rows: SectorRow[] = [];
  const entries: [string, BaselineSectorStats][] = [
    ...Object.entries(stats.sectors),
  ];
  entries.push(["ALL (corpus)", stats._corpus_all]);
  for (const [sector, s] of entries) {
    rows.push({
      sector,
      n: s.n,
      reliable: s.reliable,
      growth_mean: s.growth.mean,
      growth_stdev: s.growth.stdev,
      risk_mean: s.risk.mean,
      risk_stdev: s.risk.stdev,
      tickers: s.tickers,
    });
  }
  return rows;
}

export function SectorsPage() {
  const query = useQuery<BaselineStats | null>({
    queryKey: ["baseline"],
    queryFn: fetchBaseline,
  });

  const columns = useMemo<ColumnDef<SectorRow>[]>(
    () => [
      {
        accessorKey: "sector",
        header: "Sector",
        cell: (ctx) => <span className="font-semibold">{ctx.getValue<string>()}</span>,
      },
      {
        accessorKey: "n",
        header: "n",
        cell: (ctx) => <span className="font-mono">{ctx.getValue<number>()}</span>,
      },
      {
        accessorKey: "reliable",
        header: "Reliable",
        cell: (ctx) => (
          <Badge variant={ctx.getValue<boolean>() ? "default" : "muted"}>
            {ctx.getValue<boolean>() ? "yes" : "small sample"}
          </Badge>
        ),
      },
      {
        accessorKey: "growth_mean",
        header: "Growth μ",
        cell: (ctx) => (
          <span className="font-mono text-[hsl(var(--growth))]">
            {formatNumber(ctx.getValue<number>())}
          </span>
        ),
      },
      {
        accessorKey: "growth_stdev",
        header: "Growth σ",
        cell: (ctx) => (
          <span className="font-mono">{formatNumber(ctx.getValue<number>())}</span>
        ),
      },
      {
        accessorKey: "risk_mean",
        header: "Risk μ",
        cell: (ctx) => (
          <span className="font-mono text-[hsl(var(--risk))]">
            {formatNumber(ctx.getValue<number>())}
          </span>
        ),
      },
      {
        accessorKey: "risk_stdev",
        header: "Risk σ",
        cell: (ctx) => (
          <span className="font-mono">{formatNumber(ctx.getValue<number>())}</span>
        ),
      },
      {
        accessorKey: "tickers",
        header: "Tickers",
        cell: (ctx) => {
          const tickers = ctx.getValue<string[]>();
          return (
            <span className="truncate text-xs text-muted-foreground">
              {tickers.slice(0, 8).join(", ")}
              {tickers.length > 8 ? ` +${tickers.length - 8}` : ""}
            </span>
          );
        },
      },
    ],
    [],
  );

  if (query.isLoading) return <Loader label="Loading baseline statistics…" />;
  if (query.isError)
    return (
      <ErrorState
        title="Could not load baseline"
        error={query.error}
        onRetry={() => query.refetch()}
      />
    );

  if (!query.data) {
    return (
      <EmptyState
        title="No baseline yet"
        description="Run `python build_baseline.py` in the project root to generate baseline_stats.json."
      />
    );
  }

  const rows = flatten(query.data);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Sector baselines</h2>
        <p className="text-sm text-muted-foreground">
          Reference means and standard deviations used to compute sector-relative
          z-scores. Engine version:{" "}
          <span className="font-mono">{query.data.engine_version}</span>.
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Total records</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold tabular-nums">
              {query.data.n_total_records}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Sectors tracked</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold tabular-nums">
              {Object.keys(query.data.sectors).length}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Minimum sector size</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold tabular-nums">
              {query.data.min_sector_size}
            </p>
          </CardContent>
        </Card>
      </div>

      <DataTable columns={columns} data={rows} />
    </div>
  );
}
