import { cn } from "@/lib/utils";

interface ConfusionMatrixProps {
  labels: string[];
  matrix: number[][];
}

export function ConfusionMatrix({ labels, matrix }: ConfusionMatrixProps) {
  const max = Math.max(1, ...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <table className="min-w-[360px] border-separate border-spacing-1 text-xs">
        <thead>
          <tr>
            <th className="px-2 py-1 text-right font-medium text-muted-foreground">
              true ↓ / pred →
            </th>
            {labels.map((l) => (
              <th key={l} className="px-2 py-1 font-semibold text-foreground">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="px-2 py-1 text-right font-semibold text-foreground">
                {labels[i]}
              </th>
              {row.map((v, j) => {
                const intensity = v / max;
                return (
                  <td
                    key={j}
                    className={cn(
                      "rounded-md px-3 py-2 text-center font-mono font-semibold",
                      i === j ? "text-white" : "text-foreground",
                    )}
                    style={{
                      background:
                        i === j
                          ? `hsl(var(--growth) / ${0.25 + intensity * 0.75})`
                          : `hsl(var(--destructive) / ${intensity * 0.65})`,
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
