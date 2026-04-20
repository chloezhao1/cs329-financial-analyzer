import { useQuery } from "@tanstack/react-query";
import { Database } from "lucide-react";

import { fetchDataSource } from "@/api/signals";
import { ApiStatus } from "@/components/common/ApiStatus";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import { Badge } from "@/components/ui/badge";

const DATA_SOURCE_LABEL: Record<string, string> = {
  "data/processed": "Full pipeline: preprocessed filings",
  pipeline_output: "Scraped (not yet preprocessed)",
  demo_data: "Bundled demo dataset",
};

export function TopBar() {
  const ds = useQuery({
    queryKey: ["data-source"],
    queryFn: fetchDataSource,
    staleTime: 60_000,
  });

  const label = ds.data ? DATA_SOURCE_LABEL[ds.data.data_source] ?? ds.data.data_source : "–";

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-border/70 bg-background/70 px-4 py-3 backdrop-blur md:px-6">
      <div className="flex min-w-0 items-center gap-3">
        <Badge variant="outline" className="gap-1.5 font-normal">
          <Database className="h-3 w-3" />
          <span className="truncate">{label}</span>
        </Badge>
      </div>
      <div className="flex items-center gap-2">
        <ApiStatus />
        <ThemeToggle />
      </div>
    </header>
  );
}
