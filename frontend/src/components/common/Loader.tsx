import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";

interface LoaderProps {
  label?: string;
  className?: string;
}

export function Loader({ label = "Loading…", className }: LoaderProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-center gap-3 rounded-xl border border-border/60 bg-card/40 px-6 py-10 text-sm text-muted-foreground",
        className,
      )}
      role="status"
      aria-live="polite"
    >
      <Loader2 className="h-4 w-4 animate-spin text-primary" />
      <span>{label}</span>
    </div>
  );
}
