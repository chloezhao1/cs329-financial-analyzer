"""
financial_signal_engine_v3.py
CS329 Financial Report Analyzer, Signal Extraction Engine V3, FINAL VERSION

This version introduces three improvements over the Hybrid (LLMv1) engine,
each grounded in a specific NLP / computational linguistics concept:

Token Level Negation Scope Detection
    Problem with V2:
        Negation was a binary sentence level flag. If "not" appeared anywhere
        in a sentence, all signal was zeroed out. This caused false negatives.
        As an example: "the company, which was not previously struggling, reported 
        strong revenue growth" would have its growth signal incorrectly cancelled.
    Fix:
        We detect the position of negation words in the token list and only
        cancel signal words that are within NEGATION_SCOPE_WINDOW (5) tokens
        of a negation word. Signal words outside negation scope are preserved.
        This is an application of linguistic scope theory that negation
        has a bounded syntactic and semantic scope in natural language.

TF-IDF Weighted Mean Pooling
    Problem:
        Mean pooling weighted every token equally when constructing sentence
        embeddings. Function words like "the", "was", "of" contributed as
        much as content words like "revenue", "declined", "growth."
    Fix:
        We now fit a TF-IDF vectorizer on training sentences and use each 
        word's IDF score as its pooling weight. Words that are more specific 
        and informative (high IDF) contribute more to the sentence vector.
        Now we connect distributional semantics with vector space model 
        weighting schemes from the course.

Continuous Sentiment Score
    Problem:
        The embedding classifier snapped predictions to three discrete values
        (-1.0, 0.0, 1.0), discarding the model's uncertainty. A sentence
        with 51% positive probability got the same score as one with 95%.
    Fix:
        We use positive_probability to negative_probability as a continuous
        net score in [-1, 1]. This preserves the model's confidence in the
        output and produces more nuanced signal profiles. A sentence scored
        0.89 is meaningfully different from one scored 0.12, even if both
        are labelled "positive". This is also meaningful for users looking
        to understand details before investing. Direct improvement for 
        our goal of accessibility and transparancy.

Architecture:
    Tier 1: ScopeAwareLexiconTier (V2 + negation scope fix)
    Tier 2: EmbeddingClassifier (FinBERT + TF-IDF pooling + continuous score)
    Tier 3: LLM fallback, optional, kept here in case(off by default)

Installation (do this before you run) I added this for new engine:
    pip install torch transformers scikit-learn joblib numpy

Training:
    Run evaluate_v3.py which handles the 70/15/15 split and hyperparameter
    tuning. The trained classifier and TF-IDF vectorizer are saved together
    to data/models/finbert_logreg.pkl.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np

from financial_signal_engine import LMDictionary, SentenceScore
from financial_signal_engine_v2 import (
    SignalEngineV2,
    GROWTH_PHRASES_V2,
    MIN_TOKENS_FOR_SCORING_V2,
    LM_POSITIVE_BLOCKLIST_V2,
    _apply_negation_v2,
)
from financial_signal_engine import (
    RISK_PHRASES,
    COST_PRESSURE_PHRASES,
    PHRASE_WEIGHT,
    UNCERTAINTY_RISK_WEIGHT,
    _section_weight,
    _sentence_lemmas,
    _find_phrases,
)
from financial_signal_engine_LLMv1 import LLMClassifier, BATCH_SIZE

logger = logging.getLogger(__name__)

V3_VERSION = "0.4.0"


#CONFIG
FINBERT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_MODEL_PATH = Path("data/models/finbert_logreg.pkl")
EMBEDDING_CONFIDENCE_THRESHOLD = 0.60

LABELS = ["positive", "negative", "neutral"]
LABEL_TO_NET: dict[str, float] = {
    "positive":1.0,
    "negative": -1.0,
    "neutral": 0.0,
}




#Negation scope constants
#Maximum token distance between a negation word and a signal word
#for negation to apply. Chose this based on research on negation scope
#negation rarely scopes beyond a 5 token window.
NEGATION_SCOPE_WINDOW = 5

NEGATION_WORDS: set[str] = {
    "not", "no", "never", "neither", "nor", "without",
    "hardly", "barely", "scarcely", "n't", "cannot",
}





#token-level negation scope
def _find_negation_positions(tokens: list[dict]) -> list[int]:
    """
    Return token indices where negation words appear.
    Uses both the preprocessor's is_neg flag (if present) and a
    curated set of negation trigger words for robustness.
    Args:
        tokens: list of token dicts from the preprocessor
    Returns:
        list of integer indices where negation tokens appear
    """
    positions = []
    for i, tok in enumerate(tokens):
        is_neg = tok.get("is_neg", False)
        text = tok.get("text",  "").lower()
        lemma = tok.get("lemma", "").lower()
        if is_neg or text in NEGATION_WORDS or lemma in NEGATION_WORDS:
            positions.append(i)
    return positions


def _find_signal_positions(
    tokens: list[dict],
    signal_lemmas: set[str],
) -> list[int]:
    """
    Return token indices where lemmas match a set of signal words.
    Used to locate growth or risk signal words in the token sequence
    so we can check whether a negation word is nearby.
    Args:
        tokens:list of token dicts
        signal_lemmas: set of lemmas to search for (e.g. lm_growth hits)
    Returns:
        list of integer indices where signal tokens appear
    """
    positions = []
    for i, tok in enumerate(tokens):
        lemma = tok.get("lemma") or tok.get("text", "").lower()
        if lemma in signal_lemmas:
            positions.append(i)
    return positions



def _any_in_scope(
    neg_positions: list[int],
    signal_positions: list[int],
    window: int,
) -> bool:
    """
    Check if any signal word falls within `window` tokens of any negation word.
    Here is the core scope detection function. It implements a sliding
    window check: for each (negation, signal) pair, compute the token
    distance and return True if any pair is within the scope window.
    Args:
        neg_positions:token indices of negation words
        signal_positions:token indices of signal words
        window:maximum distance for negation to apply
    Returns:
        True if at least one signal word is in scope of a negation word
    """
    for neg_pos in neg_positions:
        for sig_pos in signal_positions:
            if abs(neg_pos - sig_pos) <= window:
                return True
    return False


#Scope aware lexicon tier
class ScopeAwareLexiconTier(SignalEngineV2):
    """
    Extends SignalEngineV2 with token level negation scope detection.
    In V2, negation was a binary sentence-level flag that zeroed out
    all signal when any negation word appeared anywhere in the sentence.
    This was linguistically incorrect because negation has bounded scope.
    The company, which was not previously struggling, reported strong
    revenue growth" has negation but the growth signal is outside scope.
    V2 would zero out growth, now ScopeAwareLexiconTier preserves it.
    Implementation:
        We locate negation words by token index, locate signal words by
        token index, and only apply negation cancellation when a signal
        word is within NEGATION_SCOPE_WINDOW tokens of a negation word.
    """
    def score_sentence(self, sentence: dict) -> SentenceScore | None:
        tokens = sentence.get("tokens", [])
        if len(tokens) < MIN_TOKENS_FOR_SCORING_V2:
            return None
        text= sentence.get("text", "")
        text_lower = text.lower()
        lemmas = _sentence_lemmas(sentence)
        section = sentence.get("section", "")
        has_neg = bool(sentence.get("has_negation"))
        has_hedge = bool(sentence.get("has_hedge"))
        lm_growth = sorted(lemmas & self.lm.growth)
        lm_risk = sorted(lemmas & self.lm.risk)
        lm_uncert = sorted(lemmas & self.lm.uncertainty)

        ph_growth = _find_phrases(text_lower, GROWTH_PHRASES_V2)
        ph_risk = _find_phrases(text_lower, RISK_PHRASES)
        ph_cost = _find_phrases(text_lower, COST_PRESSURE_PHRASES)

        growth = len(lm_growth) + len(ph_growth) * PHRASE_WEIGHT
        risk = (len(lm_risk)
                  + len(lm_uncert) * UNCERTAINTY_RISK_WEIGHT
                  + len(ph_risk) * PHRASE_WEIGHT)
        cost = len(ph_cost) * PHRASE_WEIGHT




        #scope-aware negation
        #Only cancel growth/risk signal if negation is within scope window
        if has_neg:
            neg_positions = _find_negation_positions(tokens)
            if neg_positions:
                # check growth signal scope
                growth_signal_lemmas = set(lm_growth)
                growth_positions     = _find_signal_positions(tokens, growth_signal_lemmas)
                growth_in_scope      = _any_in_scope(
                    neg_positions, growth_positions, NEGATION_SCOPE_WINDOW
                )
                risk_signal_lemmas = set(lm_risk)
                risk_positions     = _find_signal_positions(tokens, risk_signal_lemmas)
                risk_in_scope      = _any_in_scope(
                    neg_positions, risk_positions, NEGATION_SCOPE_WINDOW
                )

                #ONLY zero out signals that are actually in negation scope
                if growth_in_scope:
                    growth = 0.0
                if risk_in_scope:
                    risk = 0.0
                cost = 0.0  # cost pressure always cancelled by negation

            else:
                #has_negation flag set but no negation tokens found
                #fall back to V2 behavior for safety
                growth, risk, cost = _apply_negation_v2(
                    growth, risk, cost, has_neg,
                    lm_growth, lm_risk, ph_growth, ph_risk,
                )

        if has_hedge:
            growth *= 0.5
            risk *= 0.5
            cost *= 0.5

        w = _section_weight(section)
        growth *= w
        risk *= w
        cost *= w

        return SentenceScore(
            sent_id = sentence.get("sent_id", -1),
            section= section,
            text = text,
            growth = growth,
            risk = risk,
            cost_pressure = cost,
            net_score = growth - risk,
            has_negation = has_neg,
            has_hedge = has_hedge,
            lm_growth_hits = lm_growth,
            lm_risk_hits= lm_risk,
            lm_uncertainty_hits = lm_uncert,
            phrase_growth_hits = ph_growth,
            phrase_risk_hits= ph_risk, 
            phrase_cost_hits= ph_cost,
        )


#FinBERT embedding utilities
def _load_finbert():
    """Load FinBERT tokenizer and model as frozen feature extractor."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError(
            "torch and transformers are required for the V3 engine.\n"
            "Install with: pip install torch transformers"
        )
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()
    return tokenizer, model


def get_sentence_embeddings(
    sentences: list[str],
    tokenizer,
    model,
    batch_size: int = 32,
    tfidf_vectorizer=None,
) -> np.ndarray:
    """
    Encode sentences into FinBERT embedding vectors.
    TF-IDF weighted mean pooling:
        Standard mean pooling weights every token equally. This gives
        equal importance to "the", "was", "of" as to "revenue", "declined",
        "growth". TF-IDF weighted pooling uses each word's IDF score as its
        pooling weight, so more informative and specific words contribute
        more to the final sentence vector.
        IDF (Inverse Document Frequency) measures how specific a word is
        across the training corpus -- words that appear in fewer documents
        get higher IDF scores. This is a direct application of the VSM
        weighting scheme from the course applied to neural embeddings.
        If tfidf_vectorizer is None, falls back to standard mean pooling.
    Args:
        sentences:list of raw sentence strings
        tokenizer: HuggingFace tokenizer for FinBERT
        model: FinBERT AutoModel (frozen feature extractor)
        batch_size:sentences per forward pass
        tfidf_vectorizer: fitted TfidfVectorizer for IDF weights (optional)
    Returns:
        np.ndarray of shape (len(sentences), 768)
    """
    import torch

    #precompute IDF lookup if vectorizer provided
    idf_scores: dict[str, float] = {}
    if tfidf_vectorizer is not None:
        feature_names = tfidf_vectorizer.get_feature_names_out()
        idf_values    = tfidf_vectorizer.idf_
        idf_scores    = dict(zip(feature_names, idf_values))

    all_embeddings = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start: start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
        mask = encoded["attention_mask"].unsqueeze(-1).float()

        if idf_scores:
            batch_size_actual = len(batch)
            seq_len = encoded["input_ids"].shape[1]
            weight_tensor = torch.ones(batch_size_actual, seq_len, 1)

            for sent_idx in range(batch_size_actual):
                word_ids = encoded.word_ids(batch_index=sent_idx)
                words = batch[sent_idx].lower().split()

                for token_idx, word_id in enumerate(word_ids):
                    if word_id is not None and word_id < len(words):
                        word = words[word_id]
                        idf = idf_scores.get(word, 1.0)
                        weight_tensor[sent_idx, token_idx, 0] = idf

            summed = (last_hidden * mask * weight_tensor).sum(dim=1)
            counts = (mask * weight_tensor).sum(dim=1).clamp(min=1e-9)
        else:
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).cpu().numpy()
        all_embeddings.append(pooled)
    return np.vstack(all_embeddings)


#Embedding classifier

class EmbeddingClassifier:
    """
    Financial sentiment classifier operating in FinBERT embedding space.
    applied here:
        1. Scope-aware negation
        2. TF-IDF weighted pooling
        3. Continuous net score output
    Saves both the trained LogReg classifier and the TF-IDF vectorizer
    to disk together so both are available at inference time.
    """



#generated this segment to help debug, not sure what issue is
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        logger.info(
            "Loading FinBERT '%s' (downloads ~440 MB on first run, cached after)...",
            FINBERT_MODEL_NAME,
        )
        self.tokenizer, self.bert_model = _load_finbert()
        logger.info("FinBERT loaded.")
        self.classifier       = None
        self.tfidf_vectorizer = None
        self.model_path       = model_path

        if model_path.exists():
            saved = joblib.load(model_path)
            if isinstance(saved, dict):
                # new format: dict with classifier and tfidf
                self.classifier       = saved.get("classifier")
                self.tfidf_vectorizer = saved.get("tfidf")
                logger.info(
                    "Loaded trained classifier and TF-IDF vectorizer from %s",
                    model_path,
                )
            else:
                self.classifier = saved
                logger.info(
                    "Loaded trained classifier (legacy format) from %s",
                    model_path,
                )
        else:
            logger.warning(
                "No trained classifier found at %s. "
                "Run evaluate_v3.py first to train and save the classifier.",
                model_path,
            )

    def train(
        self,
        sentences: list[str],
        labels: list[str],
        C: float = 1.0,
    ) -> None:
        """
        Train TF-IDF vectorizer and log reg on FinBERT embeddings
            1. Fit TF-IDF vectorizer on training sentences, This learns IDF scores 
            for all words in the training corpus, which are used as pooling weights 
            during embedding extraction.
            2. Generate TF-IDF weighted FinBERT embeddings for all training
            sentences. Each sentence becomes a 768 dimen vector where
            informative words contribute more to the representation.
            3. Fit Logistic Regression on those vectors with regularization
            strength C
            4. Save both the TF-IDF vectorizer and the LogReg classifier
            together to disk for use at inference time.
        Args:
            sentences: training sentences (70% PhraseBank train split)
            labels: ground truth labels ("positive"/"negative"/"neutral")
            C:regularization strength, tuned on validation set
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

  
        logger.info(
            "Fitting TF-IDF vectorizer on %d training sentences...",
            len(sentences),
        )
        tfidf_vec = TfidfVectorizer(
            max_features=10000,
            sublinear_tf=True,
            lowercase=True,
        )
        tfidf_vec.fit(sentences)
        self.tfidf_vectorizer = tfidf_vec
        logger.info(
            "TF-IDF vocabulary size: %d terms", len(tfidf_vec.vocabulary_)
        )

        #generate TF-IDF weighted embeddings
        logger.info(
            "Generating TF-IDF weighted FinBERT embeddings for %d sentences...",
            len(sentences),
        )
        X = get_sentence_embeddings(
            sentences,
            self.tokenizer,
            self.bert_model,
            tfidf_vectorizer=tfidf_vec,
        )

        #fit Logistic Regression in R^768
        logger.info(
            "Fitting Logistic Regression in R^%d embedding space...", X.shape[1]
        )
        clf = LogisticRegression(
            max_iter=1000,
            C=C,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X, labels)
        self.classifier = clf
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"classifier": clf, "tfidf": tfidf_vec},
            self.model_path,
        )
        logger.info("Saved classifier + TF-IDF vectorizer to %s", self.model_path)

        train_acc = clf.score(X, labels)
        logger.info("Training accuracy (in-sample): %.3f", train_acc)

    def classify_batch(self, sentences: list[str]) -> list[dict]:
        """
        Classify sentences using the trained VSM classifier.
        Continuous sentiment score:
            Instead of snapping to discrete net scores (-1.0, 0.0, 1.0),
            we compute net_score = positive_probability - negative_probability.
            This gives a continuous value in [-1, 1] that preserves the
            model's confidence. A sentence with 95% positive probability
            gets a higher net_score than one with 51%, even though both
            are labelled "positive". This produces more nuanced signal
            profiles and better captures linguistic ambiguity.
        Returns list of dicts:
            {
                "sentence": str,
                "label": "positive" | "negative" | "neutral",
                "confidence":float, 
                "net_score":float, 
                "method":"embedding"
            }
        """
        if not sentences:
            return []

        if self.classifier is None:
            raise RuntimeError(
                "Classifier not trained. Run evaluate_v3.py first."
            )
        X = get_sentence_embeddings(
            sentences,
            self.tokenizer,
            self.bert_model,
            tfidf_vectorizer=self.tfidf_vectorizer,
        )
        proba = self.classifier.predict_proba(X)  # (n, 3)
        predicted_indices = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        class_labels= list(self.classifier.classes_)
        pos_idx = class_labels.index("positive") if "positive" in class_labels else -1
        neg_idx = class_labels.index("negative") if "negative" in class_labels else -1

        results = []
        for i, sentence in enumerate(sentences):
            label = class_labels[predicted_indices[i]]
            if label not in LABELS:
                label = "neutral"
            if pos_idx >= 0 and neg_idx >= 0:
                # positive_prob - negative_prob gives continuous signal in [-1, 1]
                net_score = float(proba[i, pos_idx] - proba[i, neg_idx])
            else:
                # fallback to discrete if class indices not found
                net_score = LABEL_TO_NET[label]

            results.append({
                "sentence": sentence,
                "label":label,
                "confidence":round(float(confidences[i]), 4),
                "net_score": round(net_score, 4),
                "method": "embedding",
            })
        return results


# V3 three-tier engine
class HybridSignalEngineV3:
    """
    Three-tier signal engine with all three NLP improvements applied.
    Tier 1: ScopeAwareLexiconTier:
        V2 lexicon with token-level negation scope detection.
        Handles ~33% of sentences with direct keyword hits.
    Tier 2: EmbeddingClassifier:
        FinBERT embeddings with TF-IDF weighted pooling,
        classified by a trained Logistic Regression.
        Outputs continuous net scores.
        Handles the remaining ~67% of zero-hit sentences.

    Tier 3: LLM fallback (optional, off by default):
        Claude Haiku for low-confidence embedding predictions.
        Set use_llm=True to enable. Off for all evaluation runs.

    Args:
        lm: loaded LMDictionary instance
        model_path: path to saved trained classifier (.pkl)
        use_llm: enable LLM fallback for low-confidence sentences
        api_key:Anthropic API key (only needed if use_llm=True)
    """

    def __init__(
        self,
        lm: LMDictionary,
        model_path: Path = DEFAULT_MODEL_PATH,
        use_llm: bool = False,
        api_key: str | None = None,
    ):
        # Improvement 1: use scope-aware lexicon tier
        self.lexicon_engine = ScopeAwareLexiconTier(lm)
        self.embedding_clf  = EmbeddingClassifier(model_path=model_path)
        self.use_llm = use_llm
        self.llm = LLMClassifier(api_key=api_key) if use_llm else None

        self.stats = {
            "total":0,
            "lexicon_hits": 0,
            "embedding_confident":0,
            "embedding_low_confidence": 0,
            "llm_fallback":0,
            "llm_positive":0,
            "llm_negative":0,
            "llm_neutral":0,
            "embedding_positive":0,
            "embedding_negative":0,
            "embedding_neutral":0,
        }

    def _lexicon_pass(self, sentence: dict) -> dict:
        result  = self.lexicon_engine.score_sentence(sentence)
        has_hit = (
            result is not None and (
                result.lm_growth_hits or result.lm_risk_hits
                or result.lm_uncertainty_hits or result.phrase_growth_hits
                or result.phrase_risk_hits or result.phrase_cost_hits
            )
        )
        self.stats["total"] += 1





        if has_hit:
            self.stats["lexicon_hits"] += 1
            return {
                "text":sentence.get("text", ""),
                "method": "lexicon",
                "label":self._net_to_label(result.net_score),
                "net_score":result.net_score,
                "growth": result.growth,
                "risk": result.risk,
                "cost_pressure":result.cost_pressure,
                "embedding_confidence":None,
                "llm_reason":None,
            }
        else:
            return {
                "text": sentence.get("text", ""),
                "method": "embedding_pending",
                "label": "neutral",
                "net_score": 0.0,
                "growth":0.0,
                "risk":0.0,
                "cost_pressure":0.0,
                "embedding_confidence": None,
                "llm_reason": None,
            }

    def score_batch(self, sentences: list[dict]) -> list[dict]:
        """
        Score a full batch through all three tiers.
        Pass 1: Scope-aware lexicon 
        Pass 2: TF-IDF weighted embedding VSM 
        Pass 3: LLM fallback
        """
        results = [self._lexicon_pass(s) for s in sentences]
        embedding_indices = [
            i for i, r in enumerate(results)
            if r["method"] == "embedding_pending"
        ]

        if not embedding_indices:
            return results

        # --- Pass 2: TF-IDF weighted embedding VSM ---
        logger.info(
            "Running embedding classifier on %d zero-hit sentences.",
            len(embedding_indices),
        )

        embedding_texts   = [results[i]["text"] for i in embedding_indices]
        embedding_results = self.embedding_clf.classify_batch(embedding_texts)
        llm_needed_indices = []
        for pos, emb_result in enumerate(embedding_results):
            original_idx = embedding_indices[pos]
            label = emb_result["label"]
            confidence = emb_result["confidence"]
            net_score = emb_result["net_score"]  # Improvement 3: continuous

            if confidence >= EMBEDDING_CONFIDENCE_THRESHOLD or not self.use_llm:
                self.stats["embedding_confident"] += 1
                self.stats[f"embedding_{label}"] += 1
                results[original_idx]["method"]= "embedding"
                results[original_idx]["label"]= label
                results[original_idx]["net_score"] = net_score
                results[original_idx]["embedding_confidence"] = confidence
            else:
                self.stats["embedding_low_confidence"] += 1
                results[original_idx]["method"] = "llm_pending"
                results[original_idx]["embedding_confidence"] = confidence
                llm_needed_indices.append(original_idx)

        if not llm_needed_indices or not self.use_llm:
            return results

        logger.info(
            "Escalating %d low-confidence sentences to LLM.",
            len(llm_needed_indices),
        )

        llm_texts = [results[i]["text"] for i in llm_needed_indices]
        llm_results = []

        for start in range(0, len(llm_texts), BATCH_SIZE):
            batch = llm_texts[start: start + BATCH_SIZE]
            llm_results.extend(self.llm.classify_batch(batch))
            if start + BATCH_SIZE < len(llm_texts):
                time.sleep(0.5)

        for idx, llm_result in zip(llm_needed_indices, llm_results):
            label = llm_result["label"]
            self.stats["llm_fallback"]  += 1
            self.stats[f"llm_{label}"] += 1
            results[idx]["method"]      = "llm"
            results[idx]["label"]       = label
            results[idx]["net_score"]   = LABEL_TO_NET[label]
            results[idx]["llm_reason"]  = llm_result["reason"]

        return results

    @staticmethod
    def _net_to_label(net_score: float, threshold: float = 0.1) -> str:
        if net_score >  threshold:
            return "positive"
        if net_score < -threshold:
            return "negative"
        return "neutral"



    def log_stats(self) -> None:
        t = self.stats["total"] or 1
        logger.info(
            "V3 stats: %d total | lexicon=%d (%.1f%%) | "
            "embedding=%d (%.1f%%) | llm=%d (%.1f%%)",
            t,
            self.stats["lexicon_hits"],
            100 * self.stats["lexicon_hits"] / t,
            self.stats["embedding_confident"],
            100 * self.stats["embedding_confident"] / t,
            self.stats["llm_fallback"],
            100 * self.stats["llm_fallback"] / t,
        )
        logger.info(
            "Embedding breakdown: positive=%d | negative=%d | neutral=%d",
            self.stats["embedding_positive"],
            self.stats["embedding_negative"],
            self.stats["embedding_neutral"],
        )
        if self.use_llm:
            logger.info(
                "LLM breakdown: positive=%d | negative=%d | neutral=%d",
                self.stats["llm_positive"],
                self.stats["llm_negative"],
                self.stats["llm_neutral"],
            )