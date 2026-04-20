"""

evaluate.py
 
evaluate.py evaluates the lexicon based signal engine against the Financial PhraseBank
benchmark dataset. PhraseBank contains ~4,800 financial sentences annotated by domain 
experts as positive, negative, or neutral. Our engine produces a net_operating_signal 
per sentence. We threshold that score into one of the three PhraseBank labels and then 
compare against ground truth using standard classification metrics.
 
Evaluation sections:
    1. Basic classification metrics: accuracy, precision, recall, F1
    2. Confusion matrix: Where the engine makes mistakes
    3. Threshold sensitivity: How does performance change across thresholds
    4. Coverage analysis: How many sentences actually got non-zero scores
    5. Results saved to JSON: For final report/to compare any updates in engine

"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,)
from financial_signal_engine import LMDictionary
from financial_signal_engine_v2 import SignalEngineV2
from text_preprocessor import FinancialNLPProcessor

logger = logging.getLogger(__name__)
#for any future changes
EVAL_VERSION = "0.1.0"
DEFAULT_LM_CSV = Path("data/lexicons/loughran_mcdonald.csv")
DEFAULT_OUT = Path("data/eval_results.json")
DEFAULT_THRESHOLD = 0.1

#PhraseBank uses integer labels so here we map them to strings to match our output
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
ALL_LABELS = ["positive", "negative", "neutral"]

def score_to_label(net_score: float, threshold: float) -> str:
    if net_score > threshold:
        return "positive"
    if net_score < -threshold:
        return "negative"
    return "neutral"


#coverage is the fraction of sentences where the engine produced at least one
#non zero signal hit, sentences with no hits always get classified as neutral, 
#low coverage is a useful diagnostic 
def compute_coverage(scored_pairs: list[dict]) -> dict:
    total = len(scored_pairs)
    has_any_hit = sum(
        1 for p in scored_pairs
        if p["growth"] != 0 or p["risk"] != 0 or p["cost"] != 0
    )
    forced_neutral = sum(1 for p in scored_pairs if p["pred"] == "neutral")
    true_neutral = sum(1 for p in scored_pairs if p["true"] == "neutral")
    return {
        "total_sentences": total,
        "sentences_with_signal_hit": has_any_hit,
        "coverage_rate": round(has_any_hit / total, 3) if total else 0.0,
        "predicted_neutral": forced_neutral,
        "true_neutral": true_neutral,
        
        #if predicted_neutral >> true_neutral then the threshold may be too high

        "neutral_inflation": forced_neutral - true_neutral,
    }




#we try a range of thresholds and record the accuracy & the macro F1 for each.
#this is useful for picking a threshold and showing the tradeoff
def threshold_sweep(
    raw_scores: list[float],
    y_true: list[str],
    thresholds: list[float] | None = None,
) -> list[dict]:
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    results = []
    for t in thresholds:
        y_pred = [score_to_label(s, t) for s in raw_scores]
        acc = accuracy_score(y_true, y_pred)
        # macro F1 gives equal weight to each class regardless of support
        report = classification_report(
            y_true, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0
        )


        results.append({
            "threshold": t,
            "accuracy": round(acc, 3),
            "macro_f1": round(report["macro avg"]["f1-score"], 3),
            "macro_precision": round(report["macro avg"]["precision"], 3),
            "macro_recall": round(report["macro avg"]["recall"], 3),
        })

    return results




#main eval 
def run_evaluation(
    lm_csv: Path,
    threshold: float,
    out_path: Path,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
 
    #loading components for the engine, print statements for checks 
    logger.info("Check: Loading Loughran-McDonald dictionary from %s", lm_csv)
    lm = LMDictionary.from_csv(lm_csv)
    engine = SignalEngineV2(lm)
    logger.info("Check: Initializing spaCy NLP processor")
    nlp = FinancialNLPProcessor(model="en_core_web_sm")
 
    #loading the PhraseBank
    #"sentences_allagree" is only sentences where ALL annotators agreed
    #this is the strictest split, standard one used in the literature.
    logger.info("check: Loading Financial PhraseBank")
    dataset = load_dataset(
        "takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True
    )
    items = list(dataset["train"])
    logger.info("Loaded %d sentences", len(items))
 



    #score every sentence
    y_true: list[str] = []
    y_pred: list[str] = []
    raw_net_scores: list[float] = []
    scored_pairs: list[dict] = []
 



    for item in items:
        text = item["sentence"]
        true_label = LABEL_MAP[item["label"]]
 
        #here run the real preprocessor so the lemmatization and negation detection
        #match exactly what the engine sees during normal pipeline runs

        sentence_records, _ = nlp.process_section(text, section_name="mdna")
        if not sentence_records:
            #sentence too short, treat as neutral
            net = 0.0
            growth = risk = cost = 0.0
        else:
            result = engine.score_sentence(sentence_records[0])
            if result is None:
                net = 0.0
                growth = risk = cost = 0.0
            else:
                net = result.net_score
                growth = result.growth
                risk = result.risk
                cost = result.cost_pressure
 
        pred_label = score_to_label(net, threshold)
 
        y_true.append(true_label)
        y_pred.append(pred_label)
        raw_net_scores.append(net)
        scored_pairs.append({
            "true": true_label,
            "pred": pred_label,
            "net": round(net, 3),
            "growth": round(growth, 3),
            "risk": round(risk, 3),
            "cost": round(cost, 3),
            "text": text,
        })
 



    #metrics
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0
    )


    report_str = classification_report(
        y_true, y_pred, labels=ALL_LABELS, zero_division=0
    )


    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS).tolist()
    coverage = compute_coverage(scored_pairs)
    sweep = threshold_sweep(raw_net_scores, y_true)
 
    #print, formatted this for ease but can change if needed
    print("\n" + "=" * 60)
    print("Financial PhraseBank Evaluation Results")
    print(f"Engine: loughran_mcdonald_lexicon  |  Threshold: {threshold}")
    print(f"Dataset split: sentences_allagree  |  N = {len(y_true)}")
    print("=" * 60)
 


    print(f"\nOverall Accuracy: {acc:.3f}\n")
    print(report_str)
 

    print("Confusion Matrix (rows=true, cols=pred)")
    print(f"{'':15s} {'positive':>10s} {'negative':>10s} {'neutral':>10s}")
    for label, row in zip(ALL_LABELS, cm):
        print(f"{label:15s} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d}")
 

    print("\nCoverage Analysis:")
    for k, v in coverage.items():
        print(f"  {k}: {v}")
 

    print("\nThreshold Sensitivity (accuracy / macro-F1):")
    print(f"  {'threshold':>10s} {'accuracy':>10s} {'macro_f1':>10s}")
    for row in sweep:
        marker = " <-- default" if row["threshold"] == threshold else ""
        print(
            f"  {row['threshold']:>10.2f} {row['accuracy']:>10.3f}"
            f" {row['macro_f1']:>10.3f}{marker}"
        )
 

    #results ot JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "eval_version": EVAL_VERSION,
        "dataset": "financial_phrasebank/sentences_allagree",
        "n_sentences": len(y_true),
        "threshold_used": threshold,
        "lm_csv": str(lm_csv),
        "accuracy": round(acc, 3),
        "classification_report": report_dict,
        "confusion_matrix": {
            "labels": ALL_LABELS,
            "matrix": cm,
        },
        "coverage": coverage,
        "threshold_sweep": sweep,
        # Include a sample of wrong predictions for error analysis
        "error_sample": [
            p for p in scored_pairs if p["true"] != p["pred"]
        ][:30],
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", out_path)
 




def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate signal engine against Financial PhraseBank."
    )
    ap.add_argument(
        "--lm-csv", type=Path, default=DEFAULT_LM_CSV,
        help="Path to Loughran-McDonald Master Dictionary CSV",
    )
    ap.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Net signal threshold for positive/negative classification (default: 0.1)",
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Where to save the JSON results file",
    )
    args = ap.parse_args()
    run_evaluation(args.lm_csv, args.threshold, args.out)
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())



