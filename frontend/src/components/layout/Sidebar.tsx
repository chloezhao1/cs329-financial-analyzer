import {
  BarChart3,
  Building2,
  Download,
  GitCompare,
  LayoutDashboard,
  Target,
} from "lucide-react";
import { NavLink } from "react-router-dom";

import { cn } from "@/lib/utils";

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    to: "/",
    label: "Dashboard",
    icon: LayoutDashboard,
    description: "Per-document signal breakdown",
  },
  {
    to: "/compare",
    label: "Compare",
    icon: GitCompare,
    description: "Side-by-side analysis",
  },
  {
    to: "/fetch",
    label: "Fetch & Analyze",
    icon: Download,
    description: "Run the scraping pipeline",
  },
  {
    to: "/sectors",
    label: "Sector Baselines",
    icon: BarChart3,
    description: "Z-score reference corpus",
  },
  {
    to: "/sec",
    label: "SEC Filings",
    icon: Building2,
    description: "Look up filings by ticker",
  },
  {
    to: "/evaluation",
    label: "Evaluation",
    icon: Target,
    description: "Financial PhraseBank metrics",
  },
];

export function Sidebar() {
  return (
    <aside className="hidden h-screen w-64 shrink-0 flex-col border-r border-border bg-card/40 backdrop-blur lg:flex">
      <div className="flex items-center gap-2 border-b border-border/60 px-5 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/20 text-primary">
          <BarChart3 className="h-4 w-4" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">Financial Report Analyzer</p>
          <p className="truncate text-[0.7rem] text-muted-foreground">CS329 Signal Engine</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4 scrollbar-thin">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              cn(
                "group flex items-start gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                isActive
                  ? "bg-primary/15 text-foreground"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon
                  className={cn(
                    "mt-0.5 h-4 w-4 shrink-0",
                    isActive ? "text-primary" : "text-muted-foreground",
                  )}
                />
                <div className="min-w-0">
                  <p className="font-medium text-foreground">{item.label}</p>
                  <p className="truncate text-[0.7rem] text-muted-foreground">
                    {item.description}
                  </p>
                </div>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border/60 px-5 py-4 text-[0.7rem] text-muted-foreground">
        Loughran–McDonald lexicon · v2 engine
      </div>
    </aside>
  );
}
