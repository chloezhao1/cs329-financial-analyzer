import { AnalysisDetail } from "@/components/common/AnalysisDetail";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { HybridRescorePanel } from "@/components/common/HybridRescorePanel";
import { Loader } from "@/components/common/Loader";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAnalyses } from "@/hooks/useAnalyses";
import { formatAnalysisLabel, useFiltersStore } from "@/store/filters";

export function DashboardPage() {
  const { analyses, labels, byLabel, isLoading, isError, error, refetch } = useAnalyses();
  const selected = useFiltersStore((s) => s.selectedLabels);
  const setSelected = useFiltersStore((s) => s.setSelectedLabels);

  if (isLoading) return <Loader label="Loading analyses from the pipeline…" />;
  if (isError)
    return (
      <ErrorState
        title="Could not load analyses"
        error={error}
        onRetry={() => refetch()}
      />
    );

  if (analyses.length === 0) {
    return (
      <EmptyState
        title="No processed documents yet"
        description="Run the pipeline from the ‘Fetch & Analyze’ page to populate the dashboard, or make sure data/processed/ contains at least one .processed.json."
      />
    );
  }

  const primaryLabel =
    (selected[0] && labels.includes(selected[0]) ? selected[0] : labels[0]) ?? labels[0];
  const primary = primaryLabel ? byLabel.get(primaryLabel) : undefined;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-sm text-muted-foreground">
            Pick a document to inspect its signal breakdown.
          </p>
        </div>
        <div className="min-w-[280px]">
          <Select
            value={primaryLabel}
            onValueChange={(value) => {
              const remaining = selected.filter((l) => l !== value);
              setSelected([value, ...remaining]);
            }}
          >
            <SelectTrigger>
              <SelectValue placeholder="Pick a document" />
            </SelectTrigger>
            <SelectContent>
              {analyses.map((a) => {
                const l = formatAnalysisLabel(a);
                return (
                  <SelectItem key={l} value={l}>
                    {l}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>
      </div>

      {primary ? (
        <>
          <AnalysisDetail analysis={primary} />
          <HybridRescorePanel label={primaryLabel} originalAnalysis={primary} />
        </>
      ) : (
        <EmptyState
          title="Nothing selected"
          description="Pick a document from the selector above."
        />
      )}
    </div>
  );
}
