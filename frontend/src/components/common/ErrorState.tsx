import { AlertTriangle, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ErrorStateProps {
  title?: string;
  description?: string;
  error?: unknown;
  onRetry?: () => void;
  className?: string;
}

function describeError(err: unknown): string {
  if (!err) return "";
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

export function ErrorState({
  title = "Something went wrong",
  description,
  error,
  onRetry,
  className,
}: ErrorStateProps) {
  const detail = description ?? describeError(error);
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-4 rounded-xl border border-destructive/40 bg-destructive/10 p-5 text-destructive-foreground",
        className,
      )}
    >
      <div className="mt-1 rounded-lg bg-destructive/20 p-2">
        <AlertTriangle className="h-4 w-4 text-destructive" />
      </div>
      <div className="flex-1 space-y-2">
        <div>
          <p className="text-sm font-semibold text-foreground">{title}</p>
          {detail ? (
            <p className="mt-1 text-sm text-muted-foreground">{detail}</p>
          ) : null}
        </div>
        {onRetry ? (
          <Button size="sm" variant="outline" onClick={onRetry} className="gap-2">
            <RefreshCw className="h-3.5 w-3.5" /> Try again
          </Button>
        ) : null}
      </div>
    </div>
  );
}
