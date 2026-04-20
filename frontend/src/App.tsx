import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { ComparePage } from "@/pages/ComparePage";
import { DashboardPage } from "@/pages/DashboardPage";
import { EvaluationPage } from "@/pages/EvaluationPage";
import { FetchPage } from "@/pages/FetchPage";
import { SECPage } from "@/pages/SECPage";
import { SectorsPage } from "@/pages/SectorsPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/fetch" element={<FetchPage />} />
        <Route path="/sectors" element={<SectorsPage />} />
        <Route path="/sec" element={<SECPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
