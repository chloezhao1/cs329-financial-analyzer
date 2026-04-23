"""
evaluate_all.py
CS329 Financial Report Analyzer -- Full Engine Comparison

Evaluates all five engines side-by-side on the same held-out test split:
    V1  Baseline lexicon         (financial_signal_engine.py)
    V2  Improved lexicon         (financial_signal_engine_v2.py)
    Hybrid  V2 + LLM fallback    (financial_signal_engine_LLMv1.py)
    V3  V2 + FinBERT embedding   (financial_signal_engine_v3.py)
    PureLLM  100% LLM            (financial_signal_engine_LLMpure.py)

Uses the same 70/15/15 stratified split as evaluate_v3.py (seed=42) so
numbers are directly comparable across scripts.

V3 reuses the trained model at data/models/finbert_logreg.pkl if it exists,
or trains fresh if --retrain is passed or no model is found.

Metrics per engine:
    Accuracy, Macro F1, per-class F1 (positive / negative / neutral)
    Coverage  -- fraction of test sentences that received a non-neutral,
                 non-zero signal hit from the primary (non-fallback) tier
    Confusion matrix
    Method breakdown  -- how many sentences were scored by each internal tier

Usage:
    python3 evaluate_all.py                  # all engines
    python3 evaluate_all.py --skip-llm       # lexicon + V3 only (no API calls)
    python3 evaluate_all.py --retrain        # force V3 retrain
    python3 evaluate_all.py --skip-hybrid    # skip Hybrid only
    python3 evaluate_all.py --skip-pure-llm  # skip PureLLM only

Output:
    data/eval_results_all.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from financial_signal_engine import LMDictionary, SignalEngine
from financial_signal_engine_v2 import SignalEngineV2
from financial_signal_engine_v3 import HybridSignalEngineV3, DEFAULT_MODEL_PATH
from text_preprocessor import FinancialNLPProcessor

logger = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────────────
LM_CSV       = Path("data/lexicons/loughran_mcdonald.csv")
OUT_PATH     = Path("data/eval_results_all.json")
LABEL_MAP    = {0: "negative", 1: "neutral", 2: "positive"}
ALL_LABELS   = ["positive", "negative", "neutral"]
TRAIN_SIZE   = 0.70
VAL_SIZE     = 0.15
TEST_SIZE    = 0.15
RANDOM_STATE = 42
NET_THRESHOLD = 0.1


# ── shared helpers ───────────────────────────────────────────────────────────

def net_to_label(score: float, threshold: float = NET_THRESHOLD) -> str:
    if score >  threshold: return "positive"
    if score < -threshold: return "negative"
    return "neutral"


def _safe_record(r):
    return r if r is not None else {
        "text": "", "tokens": [], "section": "mdna",
        "has_negation": False, "has_hedge": False, "sent_id": -1,
    }


def compute_metrics(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, labels=ALL_LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS).tolist()

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(report_str)

    return {
        "accuracy":     round(acc, 3),
        "macro_f1":     round(report["macro avg"]["f1-score"], 3),
        "macro_precision": round(report["macro avg"]["precision"], 3),
        "macro_recall": round(report["macro avg"]["recall"], 3),
        "positive_f1":  round(report["positive"]["f1-score"], 3),
        "negative_f1":  round(report["negative"]["f1-score"], 3),
        "neutral_f1":   round(report["neutral"]["f1-score"], 3),
        "confusion_matrix": {"labels": ALL_LABELS, "matrix": cm},
        "classification_report": report,
    }


def compute_coverage(results: list[dict]) -> dict:
    """
    Coverage = fraction of sentences that received a real signal (not a
    default neutral fallback).

    - "lexicon"   → real lexicon hit             → covered
    - "embedding" → real embedding prediction    → covered
    - "llm"       → LLM fallback in hybrid       → covered
    - "llm_pure"  → PureLLM (always covered)     → covered
    - "lexicon_miss" → no lexicon hit, no fallback → NOT covered

    For V1/V2 this reflects the fraction where the LM dictionary or
    phrases actually matched.  For Hybrid/V3/PureLLM it should be ~1.0
    since every sentence gets a real score from some tier.
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "coverage_rate": 0.0, "method_breakdown": {}}

    method_counts: dict[str, int] = {}
    for r in results:
        m = r.get("method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    NOT_COVERED = {"lexicon_miss", "llm_pending", "llm_pure_error", "unknown"}
    covered = sum(v for k, v in method_counts.items() if k not in NOT_COVERED)
    coverage_rate = round(covered / total, 3) if total else 0.0

    # also break out lexicon-only hits for engines that have mixed methods
    lexicon_hits = method_counts.get("lexicon", 0)

    return {
        "total": total,
        "covered": covered,
        "coverage_rate": coverage_rate,
        "lexicon_hits": lexicon_hits,
        "lexicon_hit_rate": round(lexicon_hits / total, 3) if total else 0.0,
        "method_breakdown": method_counts,
    }


def preprocess_split(
    texts: list[str], labels: list[str], nlp: FinancialNLPProcessor
) -> tuple[list, list[str]]:
    records, out_labels = [], []
    for text, label in zip(texts, labels):
        result, _ = nlp.process_section(text, section_name="mdna")
        records.append(result[0] if result else None)
        out_labels.append(label)
    return records, out_labels


# ── per-engine scorers ───────────────────────────────────────────────────────

def run_v1(lm: LMDictionary, test_records: list) -> list[dict]:
    engine = SignalEngine(lm)
    results = []
    for r in test_records:
        if r is None:
            results.append({"label": "neutral", "net_score": 0.0, "method": "lexicon_miss",
                             "growth": 0.0, "risk": 0.0})
            continue
        scored = engine.score_sentence(r)
        net = scored.net_score if scored else 0.0
        has_hit = scored is not None and (scored.growth != 0.0 or scored.risk != 0.0)
        results.append({
            "label": net_to_label(net),
            "net_score": net,
            "method": "lexicon" if has_hit else "lexicon_miss",
            "growth": scored.growth if scored else 0.0,
            "risk": scored.risk if scored else 0.0,
        })
    return results


def run_v2(lm: LMDictionary, test_records: list) -> list[dict]:
    engine = SignalEngineV2(lm)
    results = []
    for r in test_records:
        if r is None:
            results.append({"label": "neutral", "net_score": 0.0, "method": "lexicon",
                             "growth": 0.0, "risk": 0.0})
            continue
        scored = engine.score_sentence(r)
        net = scored.net_score if scored else 0.0
        results.append({
            "label": net_to_label(net),
            "net_score": net,
            "method": "lexicon" if (scored and (
                scored.lm_growth_hits or scored.lm_risk_hits
                or scored.phrase_growth_hits or scored.phrase_risk_hits
                or scored.phrase_cost_hits or scored.lm_uncertainty_hits
            )) else "lexicon_miss",
            "growth": scored.growth if scored else 0.0,
            "risk": scored.risk if scored else 0.0,
        })
    return results


def run_hybrid(lm: LMDictionary, safe_test: list) -> tuple[list[dict], dict]:
    from financial_signal_engine_LLMv1 import HybridSignalEngine
    engine = HybridSignalEngine(lm)
    results = engine.score_batch(safe_test)
    engine.log_stats()
    return results, engine.stats


def run_v3(
    lm: LMDictionary,
    safe_test: list,
    train_texts: list[str],
    train_labels: list[str],
    retrain: bool,
) -> tuple[list[dict], dict]:
    v3_engine = HybridSignalEngineV3(lm, use_llm=False)

    if v3_engine.embedding_clf.classifier is None or retrain:
        if retrain and DEFAULT_MODEL_PATH.exists():
            DEFAULT_MODEL_PATH.unlink()
        logger.info(
            "Training V3 classifier on %d sentences...", len(train_texts)
        )
        v3_engine.embedding_clf.train(train_texts, train_labels, C=1.0)
    else:
        logger.info("Reusing existing V3 classifier at %s", DEFAULT_MODEL_PATH)

    results = v3_engine.score_batch(safe_test)
    v3_engine.log_stats()
    return results, v3_engine.stats


def run_pure_llm(test_records: list) -> tuple[list[dict], dict]:
    from financial_signal_engine_LLMpure import PureLLMSignalEngine
    engine = PureLLMSignalEngine()
    texts = [r.get("text", "") if r else "" for r in test_records]
    sentence_dicts = [{"text": t} for t in texts]
    results = engine.score_batch(sentence_dicts)
    engine.log_stats()
    return results, engine.stats


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare all signal engines on Financial PhraseBank."
    )
    ap.add_argument("--lm-csv",        type=Path, default=LM_CSV)
    ap.add_argument("--out",           type=Path, default=OUT_PATH)
    ap.add_argument("--retrain",       action="store_true",
                    help="Force retrain V3 classifier from scratch.")
    ap.add_argument("--skip-llm",      action="store_true",
                    help="Skip both Hybrid and PureLLM (no API calls).")
    ap.add_argument("--skip-hybrid",   action="store_true",
                    help="Skip Hybrid (V2 + LLM fallback) engine.")
    ap.add_argument("--skip-pure-llm", action="store_true",
                    help="Skip PureLLM engine.")
    ap.add_argument("--skip-v3",       action="store_true",
                    help="Skip V3 (FinBERT) engine.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── load shared resources ────────────────────────────────────────────────
    logger.info("Loading Loughran-McDonald dictionary from %s", args.lm_csv)
    lm = LMDictionary.from_csv(args.lm_csv)

    logger.info("Initializing spaCy NLP processor")
    nlp = FinancialNLPProcessor(model="en_core_web_sm")

    logger.info("Loading Financial PhraseBank (sentences_allagree)")
    dataset    = load_dataset(
        "takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True
    )
    all_items  = list(dataset["train"])
    all_texts  = [item["sentence"]         for item in all_items]
    all_labels = [LABEL_MAP[item["label"]] for item in all_items]
    logger.info("Loaded %d sentences", len(all_items))

    # ── 70 / 15 / 15 split (identical to evaluate_v3.py) ────────────────────
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE, stratify=all_labels,
    )
    _, test_texts, _, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5,
        random_state=RANDOM_STATE, stratify=temp_labels,
    )

    logger.info(
        "Split: %d train | %d test (15%%) | seed=%d",
        len(train_texts), len(test_texts), RANDOM_STATE,
    )
    dist = Counter(test_labels)
    logger.info(
        "Test distribution: positive=%d | negative=%d | neutral=%d",
        dist["positive"], dist["negative"], dist["neutral"],
    )

    # ── preprocess test set once (shared by all engines) ────────────────────
    logger.info("Preprocessing %d test sentences...", len(test_texts))
    test_records, test_labels = preprocess_split(test_texts, test_labels, nlp)
    safe_test = [_safe_record(r) for r in test_records]

    # ── evaluate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Engine Comparison -- Financial PhraseBank (sentences_allagree)")
    print(f"Split: 70% train / 15% test  |  N test = {len(test_labels)}  |  seed = {RANDOM_STATE}")
    print("=" * 70)

    results_out: dict[str, dict] = {}

    # V1
    logger.info("Evaluating V1 Baseline...")
    v1_results = run_v1(lm, test_records)
    y_pred_v1  = [r["label"] for r in v1_results]
    results_out["V1 Baseline"] = compute_metrics("V1 Baseline", test_labels, y_pred_v1)
    results_out["V1 Baseline"]["coverage"] = compute_coverage(v1_results)

    # V2
    logger.info("Evaluating V2 Improved Lexicon...")
    v2_results = run_v2(lm, test_records)
    y_pred_v2  = [r["label"] for r in v2_results]
    results_out["V2 Improved"] = compute_metrics("V2 Improved", test_labels, y_pred_v2)
    results_out["V2 Improved"]["coverage"] = compute_coverage(v2_results)

    # Hybrid
    run_hybrid_flag = not args.skip_llm and not args.skip_hybrid
    if run_hybrid_flag:
        logger.info("Evaluating Hybrid (V2 + LLM fallback)...")
        hybrid_results, hybrid_stats = run_hybrid(lm, safe_test)
        y_pred_hybrid = [r["label"] for r in hybrid_results]
        results_out["Hybrid (V2+LLM)"] = compute_metrics(
            "Hybrid (V2+LLM)", test_labels, y_pred_hybrid
        )
        results_out["Hybrid (V2+LLM)"]["coverage"] = compute_coverage(hybrid_results)
        results_out["Hybrid (V2+LLM)"]["engine_stats"] = hybrid_stats
    else:
        logger.info("Skipping Hybrid engine.")

    # V3
    if not args.skip_v3:
        logger.info("Evaluating V3 (V2 + FinBERT embedding)...")
        v3_results, v3_stats = run_v3(
            lm, safe_test, train_texts, train_labels, args.retrain
        )
        y_pred_v3 = [r["label"] for r in v3_results]
        results_out["V3 FinBERT"] = compute_metrics(
            "V3 FinBERT", test_labels, y_pred_v3
        )
        results_out["V3 FinBERT"]["coverage"] = compute_coverage(v3_results)
        results_out["V3 FinBERT"]["engine_stats"] = v3_stats
    else:
        logger.info("Skipping V3 engine.")

    # PureLLM
    run_pure_flag = not args.skip_llm and not args.skip_pure_llm
    if run_pure_flag:
        logger.info("Evaluating PureLLM (100%% LLM)...")
        pure_results, pure_stats = run_pure_llm(test_records)
        y_pred_pure = [r["label"] for r in pure_results]
        results_out["PureLLM"] = compute_metrics(
            "PureLLM", test_labels, y_pred_pure
        )
        results_out["PureLLM"]["coverage"] = compute_coverage(pure_results)
        results_out["PureLLM"]["engine_stats"] = pure_stats
    else:
        logger.info("Skipping PureLLM engine.")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print(f"{'Engine':<28} {'Accuracy':>9} {'Macro F1':>9} "
          f"{'Pos F1':>8} {'Neg F1':>8} {'Neu F1':>8} {'Coverage':>9}")
    print("-" * 90)
    for name, res in results_out.items():
        cov = res.get("coverage", {}).get("coverage_rate", "n/a")
        cov_str = f"{cov:.3f}" if isinstance(cov, float) else str(cov)
        print(
            f"  {name:<26} {res['accuracy']:>9.3f} {res['macro_f1']:>9.3f} "
            f"{res['positive_f1']:>8.3f} {res['negative_f1']:>8.3f} "
            f"{res['neutral_f1']:>8.3f} {cov_str:>9}"
        )
    print("=" * 90)

    # ── coverage breakdown ───────────────────────────────────────────────────
    print("\nCoverage / method breakdown per engine:")
    for name, res in results_out.items():
        cov = res.get("coverage", {})
        breakdown = cov.get("method_breakdown", {})
        rate = cov.get("coverage_rate", "n/a")
        total = cov.get("total", 0)
        print(f"\n  {name}  (coverage={rate})")
        for method, count in sorted(breakdown.items()):
            print(f"    {method:<22} {count:>5}  ({100*count/total:.1f}%)")

    # ── confusion matrices ───────────────────────────────────────────────────
    print("\nConfusion matrices  (rows = true, cols = pred: pos / neg / neu):")
    for name, res in results_out.items():
        cm = res["confusion_matrix"]["matrix"]
        print(f"\n  {name}")
        print(f"  {'':15s} {'pos':>6} {'neg':>6} {'neu':>6}")
        for label, row in zip(ALL_LABELS, cm):
            print(f"  {label:15s} {row[0]:>6} {row[1]:>6} {row[2]:>6}")

    # ── save ─────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": "financial_phrasebank/sentences_allagree",
        "split": {
            "total": len(all_items),
            "train": len(train_texts),
            "test": len(test_texts),
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "n_test": len(test_labels),
        "engines_run": list(results_out.keys()),
        "results": results_out,
    }, indent=2))
    logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
