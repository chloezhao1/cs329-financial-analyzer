import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ComparisonRow } from "@/types/api";

type Mode = "raw" | "zscore";

interface SignalBarChartProps {
  rows: ComparisonRow[];
  mode?: Mode;
}

const COLORS = {
  growth: "hsl(var(--growth))",
  risk: "hsl(var(--risk))",
  cost: "hsl(var(--cost))",
  net: "hsl(var(--net))",
};

export function SignalBarChart({ rows, mode = "raw" }: SignalBarChartProps) {
  const data = rows.map((r) => ({
    label: r.label,
    Growth: mode === "raw" ? r.growth : r.z_growth ?? 0,
    Risk: mode === "raw" ? r.risk : r.z_risk ?? 0,
    "Net Signal": mode === "raw" ? r.net_operating_signal : r.z_net ?? 0,
    "Cost Pressure": mode === "raw" ? r.cost_pressure : 0,
  }));

  return (
    <div className="h-[360px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="label"
            stroke="hsl(var(--muted-foreground))"
            fontSize={11}
            angle={-20}
            textAnchor="end"
            interval={0}
            height={70}
          />
          <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              color: "hsl(var(--popover-foreground))",
              fontSize: 12,
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            iconType="circle"
          />
          <ReferenceLine y={0} stroke="hsl(var(--border))" />
          <Bar dataKey="Growth" fill={COLORS.growth} radius={[4, 4, 0, 0]} />
          <Bar dataKey="Risk" fill={COLORS.risk} radius={[4, 4, 0, 0]} />
          <Bar dataKey="Net Signal" fill={COLORS.net} radius={[4, 4, 0, 0]} />
          {mode === "raw" ? (
            <Bar dataKey="Cost Pressure" fill={COLORS.cost} radius={[4, 4, 0, 0]} />
          ) : null}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
