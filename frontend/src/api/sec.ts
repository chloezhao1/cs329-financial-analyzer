import type { SecFiling, SectorOverview } from "@/types/api";

import { apiRequest } from "./http";

export async function resolveCik(ticker: string): Promise<{ ticker: string; cik: string }> {
  return apiRequest(`/api/sec/cik/${encodeURIComponent(ticker.toUpperCase())}`);
}

export async function listFilings(params: {
  ticker: string;
  forms?: string[];
  max_per_type?: number;
  start_date?: string;
  end_date?: string;
}): Promise<SecFiling[]> {
  return apiRequest<SecFiling[]>("/api/sec/filings", {
    query: {
      ticker: params.ticker.toUpperCase(),
      forms: params.forms,
      max_per_type: params.max_per_type,
      start_date: params.start_date,
      end_date: params.end_date,
    },
  });
}

export async function fetchSectors(): Promise<SectorOverview> {
  return apiRequest<SectorOverview>("/api/sec/sectors");
}

export async function sectorForTicker(ticker: string): Promise<{ ticker: string; sector: string }> {
  return apiRequest(`/api/sec/sectors/ticker/${encodeURIComponent(ticker.toUpperCase())}`);
}

export async function tickersInSector(sector: string): Promise<{ sector: string; tickers: string[] }> {
  return apiRequest(`/api/sec/sectors/${encodeURIComponent(sector)}/tickers`);
}
