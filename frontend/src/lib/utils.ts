import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return "–";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatSignedNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return "–";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${formatNumber(value, digits)}`;
}

export function truncate(text: string, max = 140): string {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}
