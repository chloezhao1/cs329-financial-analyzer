import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { PlayCircle } from "lucide-react";
import { useMemo, useState } from "react";

import { fetchLatestEvaluation, runEvaluation } from "@/api/evaluation";
import { ConfusionMatrix } from "@/components/charts/ConfusionMatrix";
import { DataTable } from "@/components/common/DataTable";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { Loader } from "@/components/common/Loader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError } from "@/api/http";
import { formatNumber, truncate } from "@/lib/utils";
import type {
  EvaluationErrorSample,
  EvaluationResults,
  EvaluationThresholdRow,
} from "@/types/api";

export function EvaluationPage() {
  const queryClient = useQueryClient();
  const [threshold, setThreshold] = useState(0.1);

  const query = useQuery<EvaluationResults>({
    queryKey: ["evaluation-latest"],
    queryFn: fetchLatestEvaluation,
    retry: 0,
  });

  const mutation = useMutation<EvaluationResults, Error, number>({
    mutationFn: (t: number) => runEvaluation(t),
    onSuccess: (data) => {
      queryClient.setQueryData(["evaluation-latest"], data);
    },
  });

  const thresholdColumns = useMemo<ColumnDef<EvaluationThresholdRow>[]>(
    () => [
      { accessorKey: "threshold", header: "Threshold" },
      {
        accessorKey: "accuracy",
        header: "Accuracy",
        cell: (ctx) => (
          <span className="font-mono">{formatNumber(ctx.getValue<number>(), 3)}</span>
        ),
      },
      {
        accessorKey: "macro_f1",
        header: "Macro F1",
        cell: (ctx) => (
          <span className="font-mono font-semibold">
            {formatNumber(ctx.getValue<number>(), 3)}
          </span>
        ),
      },
      {
        accessorKey: "macro_precision",
        header: "Macro P",
        cell: (ctx) => (
          <span className="font-mono">{formatNumber(ctx.getValue<number>(), 3)}</span>
        ),
      },
      {
        accessorKey: "macro_recall",
        header: "Macro R",
        cell: (ctx) => (
          <span className="font-mono">{formatNumber(ctx.getValue<number>(), 3)}</span>
        ),
      },
    ],
    [],
  );

  const errorColumns = useMemo<ColumnDef<EvaluationErrorSample>[]>(
    () => [
      {
        accessorKey: "true",
        header: "True",
        cell: (ctx) => <Badge variant="muted">{ctx.getValue<string>()}</Badge>,
      },
      {
        accessorKey: "pred",
        header: "Pred",
        cell: (ctx) => <Badge variant="destructive">{ctx.getValue<string>()}</Badge>,
      },
      {
        accessorKey: "net",
        header: "Net",
        cell: (ctx) => (
          <span className="font-mono">{formatNumber(ctx.getValue<number>(), 2)}</span>
        ),
      },
      {
        accessorKey: "text",
        header: "Sentence",
        cell: (ctx) => (
          <span className="block max-w-[560px] text-sm">
            {truncate(ctx.getValue<string>(), 200)}
          </span>
        ),
      },
    ],
    [],
  );

  const Header = (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">PhraseBank evaluation</h2>
        <p className="text-sm text-muted-foreground">
          Benchmark of the signal engine against the Financial PhraseBank labels.
        </p>
      </div>
      <div className="flex items-end gap-2">
        <div className="w-[140px] space-y-2">
          <Label htmlFor="threshold">Threshold</Label>
          <Input
            id="threshold"
            type="number"
            step={0.05}
            min={0}
            max={2}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </div>
        <Button
          onClick={() => mutation.mutate(threshold)}
          disabled={mutation.isPending}
          className="gap-2"
        >
          <PlayCircle className="h-4 w-4" />
          {mutation.isPending ? "Running…" : "Run evaluation"}
        </Button>
      </div>
    </div>
  );

  if (query.isLoading) return <Loader label="Loading latest evaluation…" />;

  if (query.isError) {
    const err = query.error as unknown;
    const is404 = err instanceof ApiError && err.status === 404;
    if (is404) {
      return (
        <div className="space-y-6">
          {Header}
          <EmptyState
            title="No evaluation cached"
            description="Click ‘Run evaluation’ to generate data/eval_results.json. The first run downloads spaCy + the Financial PhraseBank; expect a couple of minutes."
          />
          {mutation.isPending ? <Loader label="Running PhraseBank evaluation…" /> : null}
          {mutation.isError ? (
            <ErrorState
              title="Evaluation failed"
              error={mutation.error}
              onRetry={() => mutation.mutate(threshold)}
            />
          ) : null}
        </div>
      );
    }
    return (
      <ErrorState
        title="Could not load evaluation"
        error={query.error}
        onRetry={() => query.refetch()}
      />
    );
  }

  const data = query.data!;
  const labels = data.confusion_matrix.labels;

  return (
    <div className="space-y-6">
      {Header}

      <div className="grid gap-3 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold tabular-nums">
              {formatNumber(data.accuracy, 3)}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Threshold used: {data.threshold_used}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold tabular-nums">
              {formatNumber(data.coverage.coverage_rate * 100, 1)}%
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {data.coverage.sentences_with_signal_hit} / {data.coverage.total_sentences}
              sentences with signal hits
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Neutral inflation</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold tabular-nums">
              {formatNumber(data.coverage.neutral_inflation, 2)}×
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              predicted / true neutrals
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Confusion matrix</CardTitle>
        </CardHeader>
        <CardContent>
          <ConfusionMatrix labels={labels} matrix={data.confusion_matrix.matrix} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Threshold sweep</CardTitle>
        </CardHeader>
        <CardContent>
          <DataTable columns={thresholdColumns} data={data.threshold_sweep} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Error samples</CardTitle>
        </CardHeader>
        <CardContent>
          <DataTable
            columns={errorColumns}
            data={data.error_sample}
            emptyLabel="No misclassified samples in the cached output."
          />
        </CardContent>
      </Card>
    </div>
  );
}
