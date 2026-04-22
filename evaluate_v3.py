"""
evaluate_v3.py

CS329 Financial Report Analyzer -- V3 Evaluation
Evaluates all four engine versions on Financial PhraseBank using a proper
70/15/15 train/validation/test split following standard research protocol.

Experimental Protocol:
    This evaluation follows the three-way split methodology standard in
    NLP research to ensure no information from the test set influences
    any modeling decision:
    70% Train split:
        Used exclusively to train the V3 embedding classifier (Logistic
        Regression on FinBERT embeddings). No evaluation happens here.
    15% Validation split:
        Used exclusively for hyperparameter selection. We tune two
        hyperparameters on this split:
            (1) Regularization strength C for the Logistic Regression
                Candidates: [0.01, 0.1, 1.0, 10.0]
            (2) Confidence threshold for LLM escalation
                Candidates: [0.50, 0.55, 0.60, 0.65, 0.70]
        The best values are selected by Macro F1 on the validation set.
        The test set is never seen during this process.
    15% Test split:
        Used exactly once at the end to report final results for all
        four engines. No decisions are made after seeing these numbers.
        This is the number reported in the paper.
    Selecting hyperparameters by looking at test set performance is a
    form of data leakage -- the model indirectly fits to the test set.
    Using a separate validation set ensures the test results are a
    genuine measure of generalization to unseen data. This is important
    for our methodology section so we will explain it.

    This is the protocol used in published NLP research. The previous
    80/20 split was simpler but did not allow for principled
    hyperparameter selection without risking test set contamination.

Engines evaluated:
    V1 Baseline, original lexicon (financial_signal_engine.py)
    V2 Improved, expanded phrases + negation fix (financial_signal_engine_v2.py)
    Hybrid, V2 + LLM fallback (financial_signal_engine_LLMv1.py)
    V3 Embedding, V2 + FinBERT VSM (financial_signal_engine_v3.py)

Usage:
    python3 evaluate_v3.py
    python3 evaluate_v3.py

    USE THIS TO RUN:
    python3 evaluate_v3.py --retrain  

Output:
    data/eval_results_v3.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from financial_signal_engine import LMDictionary, SignalEngine
from financial_signal_engine_v2 import SignalEngineV2
from financial_signal_engine_LLMv1 import HybridSignalEngine
from financial_signal_engine_v3 import HybridSignalEngineV3, DEFAULT_MODEL_PATH
from text_preprocessor import FinancialNLPProcessor

logger = logging.getLogger(__name__)

#Config
LM_CSV   = Path("data/lexicons/loughran_mcdonald.csv")
OUT_PATH = Path("data/eval_results_v3.json")

LABEL_MAP  = {0: "negative", 1: "neutral", 2: "positive"}
ALL_LABELS = ["positive", "negative", "neutral"]

# Three-way split ratios
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_STATE = 42
NET_THRESHOLD = 0.1

#Hyperparameter candidates
C_CANDIDATES= [0.01, 0.1, 1.0, 10.0]
THRESHOLD_CANDIDATES  = [0.50, 0.55, 0.60, 0.65, 0.70]


#Helpers

def net_to_label(net_score: float, threshold: float = NET_THRESHOLD) -> str:
    if net_score >  threshold:
        return "positive"
    if net_score < -threshold:
        return "negative"
    return "neutral"


def _safe_record(record):
    return record if record is not None else {
        "text": "", "tokens": [], "section": "mdna",
        "has_negation": False, "has_hedge": False, "sent_id": -1,
    }


def evaluate_engine(name: str, y_true: list, y_pred: list) -> dict:
    """Compute and print metrics for one engine."""
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, labels=ALL_LABELS, zero_division=0
    )
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(report_str)
    return {
        "accuracy":round(acc, 3),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "positive_f1":round(report["positive"]["f1-score"], 3),
        "negative_f1":round(report["negative"]["f1-score"], 3),
        "neutral_f1": round(report["neutral"]["f1-score"], 3),
        "classification_report": report,
    }


def preprocess_sentences(
    texts: list[str],
    labels: list[str],
    nlp: "FinancialNLPProcessor",
) -> tuple[list, list]:
    """Preprocess a list of sentences, returning (records, labels) aligned."""
    records = []
    valid_labels = []
    for text, label in zip(texts, labels):
        result, _ = nlp.process_section(text, section_name="mdna")
        records.append(result[0] if result else None)
        valid_labels.append(label)
    return records, valid_labels



#Hyperparameter tuning on validation set

def tune_C_on_validation(
    train_texts: list[str],
    train_labels: list[str],
    val_records: list,
    val_labels: list[str],
    lm: "LMDictionary",
    model_path: Path,
) -> float:
    """
    Select best regularization strength C by Macro F1 on validation set.
    This is the correct way to choose hyperparameters, we never look
    at the test set during this process. The validation set is used
    exclusively for this selection, then discarded.
    Returns the best C value.
    """
    from financial_signal_engine_v3 import EmbeddingClassifier, LABEL_TO_NET

    logger.info(
        "Tuning C on validation set. Candidates: %s", C_CANDIDATES
    )

    #load FinBERT once and reuse across C candidates to save time
    emb_clf = EmbeddingClassifier(model_path=Path("data/models/_tune_temp.pkl"))

    best_c      = 1.0
    best_macro  = -1.0
    tuning_log  = []

    for c in C_CANDIDATES:
        logger.info("  Trying C=%.3f...", c)

        #train with this C value
        emb_clf.train(train_texts, train_labels, C=c)

        #build a temporary V3 engine with this classifier
        engine = HybridSignalEngineV3(lm, model_path=Path("data/models/_tune_temp.pkl"), use_llm=False)

        #score validation set
        safe_records = [_safe_record(r) for r in val_records]
        results = engine.score_batch(safe_records)
        y_pred= [r["label"] for r in results]

        macro_f1 = accuracy_score(val_labels, y_pred)
        report = classification_report(
            val_labels, y_pred, labels=ALL_LABELS,
            output_dict=True, zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]

        logger.info("    C=%.3f → val Macro F1=%.3f", c, macro_f1)
        tuning_log.append({"C": c, "val_macro_f1": round(macro_f1, 4)})

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_c = c

    logger.info(
        "Best C=%.3f (val Macro F1=%.3f)", best_c, best_macro
    )

    # clean up temp model
    temp_path = Path("data/models/_tune_temp.pkl")
    if temp_path.exists():
        temp_path.unlink()

    return best_c, tuning_log


def tune_threshold_on_validation(
    val_records: list,
    val_labels: list[str],
    lm: "LMDictionary",
    model_path: Path,
) -> float:
    """
    Select best confidence threshold by Macro F1 on validation set.
    The threshold controls when a low-confidence embedding prediction
    escalates to the LLM. We select it on the validation set so the
    test set is never influenced by this choice.
    Returns the best threshold value.
    """
    logger.info(
        "Tuning confidence threshold on validation set. Candidates: %s",
        THRESHOLD_CANDIDATES,
    )

    best_threshold = 0.60
    best_macro     = -1.0
    tuning_log     = []

    for threshold in THRESHOLD_CANDIDATES:
        from financial_signal_engine_v3 import EMBEDDING_CONFIDENCE_THRESHOLD
        import financial_signal_engine_v3 as v3_module

        #temporarily override the threshold constant
        original = v3_module.EMBEDDING_CONFIDENCE_THRESHOLD
        v3_module.EMBEDDING_CONFIDENCE_THRESHOLD = threshold

        engine = HybridSignalEngineV3(lm, model_path=model_path, use_llm=False)
        safe_records = [_safe_record(r) for r in val_records]
        results = engine.score_batch(safe_records)
        y_pred = [r["label"] for r in results]

        report = classification_report(
            val_labels, y_pred, labels=ALL_LABELS,
            output_dict=True, zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]

        logger.info(
            "  threshold=%.2f → val Macro F1=%.3f", threshold, macro_f1
        )
        tuning_log.append({
            "threshold": threshold,
            "val_macro_f1": round(macro_f1, 4)
        })

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_threshold = threshold

        #restore original
        v3_module.EMBEDDING_CONFIDENCE_THRESHOLD = original

    logger.info(
        "Best threshold=%.2f (val Macro F1=%.3f)", best_threshold, best_macro
    )
    return best_threshold, tuning_log


#Main

def main() -> None:
    ap = argparse.ArgumentParser(description="V3 evaluation with 70/15/15 split.")
    ap.add_argument("--skip-hybrid", action="store_true",
                    help="Skip Hybrid (LLM) evaluation.")
    ap.add_argument("--retrain", action="store_true",
                    help="Force retrain classifier from scratch.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    #Load resources

    logger.info("Loading LM dictionary...")
    lm = LMDictionary.from_csv(LM_CSV)

    logger.info("Loading spaCy NLP processor...")
    nlp = FinancialNLPProcessor(model="en_core_web_sm")

    #Load PhraseBank and split 70/15/15

    logger.info("Loading Financial PhraseBank (sentences_allagree)...")
    dataset   = load_dataset(
        "takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True
    )
    all_items  = list(dataset["train"])
    all_texts  = [item["sentence"]        for item in all_items]
    all_labels = [LABEL_MAP[item["label"]] for item in all_items]

    logger.info("Total PhraseBank sentences: %d", len(all_items))

    #first split: 70% train, 30% temp (will become val + test)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=all_labels,
    )

    #second split: split the 30% temp into 15% val and 15% test
    #test_size=0.5 of the 30% temp = 15% of total
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    logger.info(
        "Split: %d train (70%%) | %d val (15%%) | %d test (15%%)",
        len(train_texts), len(val_texts), len(test_texts),
    )

    #log class distributions as sanity check
    for split_name, split_labels in [
        ("Train", train_labels), ("Val", val_labels), ("Test", test_labels)
    ]:
        dist = Counter(split_labels)
        logger.info(
            "%s distribution: positive=%d | negative=%d | neutral=%d",
            split_name,
            dist.get("positive", 0),
            dist.get("negative", 0),
            dist.get("neutral",  0),
        )

    #Preprocess val and test sentences

    logger.info("Preprocessing validation sentences...")
    val_records, val_labels = preprocess_sentences(val_texts, val_labels, nlp)

    logger.info("Preprocessing test sentences...")
    test_records, test_labels = preprocess_sentences(test_texts, test_labels, nlp)

    #Tune C on validation set, train final classifier

    if args.retrain and DEFAULT_MODEL_PATH.exists():
        logger.info("--retrain flag set, removing existing model...")
        DEFAULT_MODEL_PATH.unlink()

    #update EmbeddingClassifier.train to accept C parameter
    logger.info("Initializing V3 engine...")
    v3_engine = HybridSignalEngineV3(lm, use_llm=False)

    c_tuning_log       = []
    threshold_tuning_log = []
    best_c             = 1.0
    best_threshold     = 0.60

    if v3_engine.embedding_clf.classifier is None or args.retrain:
        # tune C on validation set first
        best_c, c_tuning_log = tune_C_on_validation(
            train_texts, train_labels,
            val_records, val_labels,
            lm, DEFAULT_MODEL_PATH,
        )

        #train final classifier with best C on full train split
        logger.info(
            "Training final classifier with best C=%.3f on %d training sentences...",
            best_c, len(train_texts),
        )
        v3_engine.embedding_clf.train(train_texts, train_labels, C=best_c)

        #tune confidence threshold on validation set
        best_threshold, threshold_tuning_log = tune_threshold_on_validation(
            val_records, val_labels, lm, DEFAULT_MODEL_PATH
        )

        logger.info(
            "Final hyperparameters: C=%.3f, threshold=%.2f",
            best_c, best_threshold,
        )
    else:
        logger.info("Loaded existing trained classifier -- skipping tuning.")

    #apply best threshold
    import financial_signal_engine_v3 as v3_module
    v3_module.EMBEDDING_CONFIDENCE_THRESHOLD = best_threshold

    #Evaluate all engines on TEST SET (run exactly once)

    safe_test = [_safe_record(r) for r in test_records]

    print("\n" + "=" * 65)
    print("V3 Evaluation -- Engine Comparison on Held-Out Test Set")
    print("Dataset: Financial PhraseBank (sentences_allagree)")
    print(f"Split: 70% train / 15% val / 15% test "
          f"(stratified, seed={RANDOM_STATE})")
    print(f"N test: {len(test_labels)} sentences")
    print(f"Best C: {best_c}  |  Best threshold: {best_threshold}")
    print("=" * 65)

    results_out = {}

    #V1 Baseline
    logger.info("Evaluating V1 baseline...")
    v1_engine = SignalEngine(lm)
    y_pred_v1 = []
    for record in test_records:
        if record is None:
            y_pred_v1.append("neutral")
            continue
        result = v1_engine.score_sentence(record)
        y_pred_v1.append(net_to_label(result.net_score) if result else "neutral")
    results_out["V1 Baseline"] = evaluate_engine("V1 Baseline", test_labels, y_pred_v1)

    #V2 Improved
    logger.info("Evaluating V2 improved lexicon...")
    v2_engine = SignalEngineV2(lm)
    y_pred_v2 = []
    for record in test_records:
        if record is None:
            y_pred_v2.append("neutral")
            continue
        result = v2_engine.score_sentence(record)
        y_pred_v2.append(net_to_label(result.net_score) if result else "neutral")
    results_out["V2 Improved"] = evaluate_engine("V2 Improved", test_labels, y_pred_v2)

    #Hybrid (optional)
    if not args.skip_hybrid:
        logger.info("Evaluating Hybrid (V2 + LLM fallback)...")
        hybrid_engine  = HybridSignalEngine(lm)
        hybrid_results = hybrid_engine.score_batch(safe_test)
        y_pred_hybrid  = [r["label"] for r in hybrid_results]
        hybrid_engine.log_stats()
        results_out["Hybrid (V2 + LLM)"] = evaluate_engine(
            "Hybrid (V2 + LLM)", test_labels, y_pred_hybrid
        )
        results_out["Hybrid (V2 + LLM)"]["hybrid_stats"] = hybrid_engine.stats
    else:
        logger.info("Skipping Hybrid evaluation (--skip-hybrid flag set).")

    #V3 Embedding
    logger.info("Evaluating V3 (V2 + FinBERT embedding VSM)...")
    v3_results = v3_engine.score_batch(safe_test)
    y_pred_v3  = [r["label"] for r in v3_results]
    v3_engine.log_stats()
    results_out["V3 Embedding (ours)"] = evaluate_engine(
        "V3 Embedding (ours)", test_labels, y_pred_v3
    )
    results_out["V3 Embedding (ours)"]["v3_stats"] = v3_engine.stats

    #Summary table

    print("\n" + "=" * 65)
    print("Summary")
    print(f"{'Engine':<28} {'Accuracy':>9} {'Macro F1':>9} "
          f"{'Pos F1':>8} {'Neg F1':>8} {'Neu F1':>8}")
    print("-" * 65)
    for name, res in results_out.items():
        print(
            f"  {name:<26} {res['accuracy']:>9.3f} {res['macro_f1']:>9.3f} "
            f"{res['positive_f1']:>8.3f} {res['negative_f1']:>8.3f} "
            f"{res['neutral_f1']:>8.3f}"
        )
    print("=" * 65)

    #coverage breakdown
    method_counts = {}
    for r in v3_results:
        m = r.get("method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    total = len(v3_results)
    print("\nV3 coverage breakdown:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method:<20} {count:>5} ({100*count/total:.1f}%)")

    
    #Save results

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "split": {
            "total":len(all_items),
            "train":len(train_texts),
            "val": len(val_texts),
            "test": len(test_texts),
            "train_pct": 0.70,
            "val_pct":0.15,
            "test_pct": 0.15,
            "random_state": RANDOM_STATE,
            "stratified":True,
        },
        "hyperparameter_tuning": {
            "C_candidates":C_CANDIDATES,
            "best_C": best_c,
            "C_tuning_log": c_tuning_log,
            "threshold_candidates": THRESHOLD_CANDIDATES,
            "best_threshold":best_threshold,
            "threshold_tuning_log":threshold_tuning_log,
        },
        "results": results_out,
        "v3_coverage":method_counts,
    }, indent=2))
    logger.info("Results saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()