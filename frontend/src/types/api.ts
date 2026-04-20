/**
 * Shared TypeScript interfaces mirroring the FastAPI response shapes.
 * Keep these in sync with backend/routers/*.py.
 */

export type FormType = "10-K" | "10-Q" | "EARNINGS_CALL" | "8-K" | string;

export type DataSource = "data/processed" | "pipeline_output" | "demo_data";

export interface MethodInfo {
  type: string;
  engine_version: string;
  lm_words_loaded: number;
  lm_growth_words: number;
  lm_risk_words: number;
  lm_uncertainty_words: number;
  phrase_counts: {
    growth: number;
    risk: number;
    cost: number;
  };
  aggregation: string;
}

export interface Scores {
  growth: number;
  risk: number;
  cost_pressure: number;
  net_operating_signal: number;
}

export interface Coverage {
  scored_sentences: number;
  scored_with_hits: number;
  sentences_by_section: Record<string, number>;
}

export interface ZScores {
  reference: string;
  reference_label: string;
  reference_n: number;
  reference_reliable: boolean;
  is_sector_specific: boolean;
  growth: number;
  risk: number;
  net_operating_signal: number;
}

export interface PhraseHit {
  term: string;
  source: "lm_word" | "phrase" | string;
  count: number;
}

export interface SentenceScore {
  sent_id: number;
  section: string;
  text: string;
  growth: number;
  risk: number;
  cost_pressure: number;
  net_score: number;
  has_negation: boolean;
  has_hedge: boolean;
  lm_growth_hits: string[];
  lm_risk_hits: string[];
  lm_uncertainty_hits: string[];
  phrase_growth_hits: string[];
  phrase_risk_hits: string[];
  phrase_cost_hits: string[];
}

export interface Analysis {
  ticker: string;
  company_name: string;
  form_type: FormType;
  filing_date: string;
  source: string;
  method?: MethodInfo;
  scores: Scores;
  coverage: Coverage;
  top_sentences: SentenceScore[];
  top_growth_phrases: PhraseHit[];
  top_risk_phrases: PhraseHit[];
  top_cost_phrases: PhraseHit[];
  zscores?: ZScores;
}

export interface ComparisonRow {
  label: string;
  ticker: string;
  filing_date: string;
  form_type: FormType;
  growth: number;
  risk: number;
  cost_pressure: number;
  net_operating_signal: number;
  z_growth: number | null;
  z_risk: number | null;
  z_net: number | null;
  z_reference: string | null;
  z_reference_label: string | null;
  z_is_sector_specific: boolean | null;
  z_reliable: boolean | null;
  scored_sentences: number;
}

export interface RunPipelineRequest {
  tickers: string[];
  form_types: FormType[];
  max_per_type: number;
  skip_sec?: boolean;
  skip_transcripts?: boolean;
  start_date?: string;
  end_date?: string;
  kaggle_pkl?: string | null;
}

export interface RunPipelineResponse {
  n_records: number;
  n_analyses: number;
  tickers: string[];
  form_types: string[];
}

export interface SectorOverview {
  sectors: string[];
  map: Record<string, string>;
  coverage: Record<string, number>;
}

export interface SecFiling {
  ticker: string;
  company_name: string;
  cik: string;
  form_type: string;
  filing_date: string;
  accession_number: string;
  primary_document: string;
}

export interface BaselineSectorStats {
  n: number;
  tickers: string[];
  reliable: boolean;
  growth: { mean: number; stdev: number };
  risk: { mean: number; stdev: number };
}

export interface BaselineStats {
  engine_version: string;
  n_total_records: number;
  indicators: string[];
  min_sector_size: number;
  sectors: Record<string, BaselineSectorStats>;
  _corpus_all: BaselineSectorStats;
}

export interface EvaluationThresholdRow {
  threshold: number;
  accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
}

export interface EvaluationCoverage {
  total_sentences: number;
  sentences_with_signal_hit: number;
  coverage_rate: number;
  predicted_neutral: number;
  true_neutral: number;
  neutral_inflation: number;
}

export interface EvaluationErrorSample {
  true: string;
  pred: string;
  net: number;
  growth: number;
  risk: number;
  cost: number;
  text: string;
}

export interface EvaluationResults {
  eval_version: string;
  dataset: string;
  n_sentences: number;
  threshold_used: number;
  lm_csv: string;
  accuracy: number;
  classification_report: Record<string, Record<string, number> | number>;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  coverage: EvaluationCoverage;
  threshold_sweep: EvaluationThresholdRow[];
  error_sample: EvaluationErrorSample[];
}

export type HybridMethod = "lexicon" | "llm" | "llm_pending";
export type HybridLabel = "positive" | "negative" | "neutral";

export interface HybridSentence {
  text: string;
  method: HybridMethod;
  label: HybridLabel;
  net_score: number;
  growth: number;
  risk: number;
  cost_pressure: number;
  llm_reason: string | null;
}

export interface HybridRescoreResponse {
  label: string;
  ticker: string;
  form_type: string;
  filing_date: string;
  total_sentences: number;
  scanned_sentences: number;
  lexicon_hits: number;
  llm_fallback: number;
  llm_positive: number;
  llm_negative: number;
  llm_neutral: number;
  lexicon_coverage_rate: number;
  hybrid_coverage_rate: number;
  sentences: HybridSentence[];
  // Aggregated hybrid document scores
  hybrid_growth_score: number;
  hybrid_risk_score: number;
  hybrid_cost_score: number;
  hybrid_net_score: number;
  hybrid_positive_count: number;
  hybrid_negative_count: number;
  hybrid_neutral_count: number;
}
