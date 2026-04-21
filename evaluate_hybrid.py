"""
evaluate_hybrid.py

CS329 Financial Report Analyzer, Hybrid Evaluation
Owner: Riyaa

Compares two approaches side by side on Financial PhraseBank:
- Lexicon only (v2 engine)
- Hybrid (lexicon where it has hits, LLM fallback for zero-hit sentences)
Results are saved to data/eval_results_hybrid.json.

Usage:
    python3 evaluate_hybrid.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

from financial_signal_engine import LMDictionary
from financial_signal_engine_v2 import SignalEngineV2
from financial_signal_engine_LLMv1 import HybridSignalEngine
from text_preprocessor import FinancialNLPProcessor

logger = logging.getLogger(__name__)

LM_CSV = Path("data/lexicons/loughran_mcdonald.csv")
OUT_PATH = Path("data/eval_results_hybrid.json")
THRESHOLD = 0.1
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
ALL_LABELS = ["positive", "negative", "neutral"]




def score_to_label(net_score: float, threshold: float = THRESHOLD) -> str:
    if net_score > threshold:
        return "positive"
    if net_score < -threshold:
        return "negative"
    return "neutral"




def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading LM dictionary")
    lm = LMDictionary.from_csv(LM_CSV)

    logger.info("Loading spaCy processor")
    nlp = FinancialNLPProcessor(model="en_core_web_sm")

    logger.info("Loading Financial PhraseBank")
    dataset = load_dataset(
        "takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True
    )
    items = list(dataset["train"])
    logger.info("Loaded %d sentences", len(items))

    #preprocess all sentences once, reused by both methods
    logger.info("Preprocessing all sentences")
    preprocessed = []
    y_true = []
    for item in items:
        text = item["sentence"]
        true_label = LABEL_MAP[item["label"]]
        records, _ = nlp.process_section(text, section_name="mdna")
        preprocessed.append(records[0] if records else None)
        y_true.append(true_label)


    #safe fallback record for any sentence that was too short to preprocess
    def _safe(record):
        return record if record is not None else {
            "text": "", "tokens": [], "section": "mdna",
            "has_negation": False, "has_hedge": False, "sent_id": -1,
        }

    
    #Method 1: Lexicon only, v2
    logger.info("Running lexicon-only evaluation (v2)...")
    lexicon_engine = SignalEngineV2(lm)
    y_pred_lexicon = []
    for record in preprocessed:
        if record is None:
            y_pred_lexicon.append("neutral")
            continue
        result = lexicon_engine.score_sentence(record)
        if result is None:
            y_pred_lexicon.append("neutral")
        else:
            y_pred_lexicon.append(score_to_label(result.net_score))




    #Method 2: Hybrid (lexicon + LLM fallback for zero-hit sentences)
    logger.info("Running hybrid evaluation (lexicon + LLM fallback)...")
    hybrid_engine = HybridSignalEngine(lm)
    valid_records = [_safe(r) for r in preprocessed]
    hybrid_results = hybrid_engine.score_batch(valid_records)
    y_pred_hybrid = [r["label"] for r in hybrid_results]
    hybrid_engine.log_stats()




    # Print comparison
    methods = {
        "Lexicon only (v2)": y_pred_lexicon,
        "Hybrid": y_pred_hybrid,
    }

    results_out = {}
    print("\n" + "=" * 60)
    print("Hybrid Evaluation -- Method Comparison")
    print(f"Dataset: Financial PhraseBank (sentences_allagree) N={len(y_true)}")
    print("=" * 60)



    for name, y_pred in methods.items():
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y_true, y_pred, labels=ALL_LABELS, zero_division=0
        )
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.3f}")
        print(report_str)
        results_out[name] = {
            "accuracy": round(acc, 3),
            "macro_f1": round(report["macro avg"]["f1-score"], 3),
            "classification_report": report,
        }



    #summary table
    print("\nSummary:")
    print(f"  {'Method':<25} {'Accuracy':>10} {'Macro F1':>10}")
    for name, res in results_out.items():
        print(f"  {name:<25} {res['accuracy']:>10.3f} {res['macro_f1']:>10.3f}")



    #save results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_sentences": len(y_true),
        "results": results_out,
        "hybrid_stats": hybrid_engine.stats,
    }, indent=2))
    logger.info("Saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()