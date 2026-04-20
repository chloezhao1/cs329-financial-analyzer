import { useMutation, useQuery } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";

import { fetchSectors, listFilings, resolveCik } from "@/api/sec";
import { DataTable } from "@/components/common/DataTable";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { Loader } from "@/components/common/Loader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { SecFiling } from "@/types/api";

export function SECPage() {
  const [ticker, setTicker] = useState("AAPL");

  const sectorsQuery = useQuery({
    queryKey: ["sectors"],
    queryFn: fetchSectors,
    staleTime: 5 * 60_000,
  });

  const filingsMutation = useMutation({
    mutationFn: () =>
      listFilings({ ticker, forms: ["10-K", "10-Q"], max_per_type: 4 }),
  });

  const cikMutation = useMutation({
    mutationFn: () => resolveCik(ticker),
  });

  const columns = useMemo<ColumnDef<SecFiling>[]>(
    () => [
      { accessorKey: "filing_date", header: "Filed" },
      {
        accessorKey: "form_type",
        header: "Form",
        cell: (ctx) => <Badge variant="outline">{ctx.getValue<string>()}</Badge>,
      },
      { accessorKey: "company_name", header: "Company" },
      {
        accessorKey: "accession_number",
        header: "Accession",
        cell: (ctx) => (
          <span className="font-mono text-xs">{ctx.getValue<string>()}</span>
        ),
      },
      {
        accessorKey: "primary_document",
        header: "Document",
        cell: (ctx) => (
          <span className="truncate font-mono text-xs text-muted-foreground">
            {ctx.getValue<string>()}
          </span>
        ),
      },
    ],
    [],
  );

  const submit = (): void => {
    if (!ticker.trim()) return;
    cikMutation.mutate();
    filingsMutation.mutate();
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">SEC filings lookup</h2>
        <p className="text-sm text-muted-foreground">
          Resolve CIKs and preview recent filing metadata before committing to a
          full pipeline run.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Look up a ticker</CardTitle>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <div className="w-full max-w-[220px] space-y-2">
              <Label htmlFor="sec-ticker">Ticker</Label>
              <Input
                id="sec-ticker"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="AAPL"
              />
            </div>
            <Button type="submit" className="gap-2">
              <Search className="h-4 w-4" /> Lookup
            </Button>
            {cikMutation.data ? (
              <Badge variant="default" className="ml-1">
                CIK {cikMutation.data.cik}
              </Badge>
            ) : null}
          </form>
        </CardContent>
      </Card>

      {filingsMutation.isPending ? (
        <Loader label="Fetching filings from SEC EDGAR…" />
      ) : filingsMutation.isError ? (
        <ErrorState
          title="SEC lookup failed"
          error={filingsMutation.error}
          onRetry={submit}
        />
      ) : filingsMutation.data ? (
        filingsMutation.data.length === 0 ? (
          <EmptyState title="No filings matched" description="Try another ticker." />
        ) : (
          <DataTable columns={columns} data={filingsMutation.data} />
        )
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Sector coverage</CardTitle>
        </CardHeader>
        <CardContent>
          {sectorsQuery.isLoading ? (
            <Loader label="Loading sector map…" />
          ) : sectorsQuery.isError ? (
            <ErrorState title="Could not load sector map" error={sectorsQuery.error} />
          ) : (
            <div className="flex flex-wrap gap-2">
              {Object.entries(sectorsQuery.data?.coverage ?? {}).map(([sector, n]) => (
                <Badge key={sector} variant="muted" className="gap-1">
                  <span>{sector}</span>
                  <span className="text-muted-foreground">· {n}</span>
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
