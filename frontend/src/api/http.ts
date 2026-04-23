/**
 * - In Vite **dev** (`npm run dev`), if `VITE_API_URL` is unset or empty, use
 *   same-origin requests so `vite.config.ts` can proxy `/api` to the FastAPI
 *   backend and avoid CORS.
 * - In **production** builds, default to this machine's API (override via env).
 */
const envUrl = import.meta.env.VITE_API_URL;
const trimmed = envUrl != null && String(envUrl).trim() !== "" ? String(envUrl).trim() : null;
const RAW_BASE = (trimmed ??
  (import.meta.env.DEV ? "" : "http://localhost:8000")) as string;

export const API_BASE = RAW_BASE.replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.status = status;
    this.body = body;
    this.name = "ApiError";
  }
}

interface RequestOptions {
  method?: string;
  query?: Record<string, string | number | boolean | string[] | undefined | null>;
  body?: unknown;
  signal?: AbortSignal;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const base = `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
  if (!query) return base;

  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue;
    if (Array.isArray(value)) {
      for (const v of value) params.append(key, String(v));
    } else {
      params.append(key, String(value));
    }
  }
  const qs = params.toString();
  return qs ? `${base}?${qs}` : base;
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = buildUrl(path, options.query);
  const init: RequestInit = {
    method: options.method ?? "GET",
    headers: { Accept: "application/json" },
    signal: options.signal,
  };
  if (options.body !== undefined) {
    init.body = JSON.stringify(options.body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }

  let resp: Response;
  try {
    resp = await fetch(url, init);
  } catch (err) {
    throw new ApiError(0, "Network error: the API is not reachable.", err);
  }

  const contentType = resp.headers.get("content-type") ?? "";
  const parsed: unknown = contentType.includes("application/json")
    ? await resp.json().catch(() => null)
    : await resp.text().catch(() => null);

  if (!resp.ok) {
    const message =
      (parsed as { detail?: { message?: string } | string } | null)?.detail &&
      typeof (parsed as { detail: unknown }).detail === "string"
        ? (parsed as { detail: string }).detail
        : `Request failed (${resp.status})`;
    throw new ApiError(resp.status, message, parsed);
  }

  return parsed as T;
}
