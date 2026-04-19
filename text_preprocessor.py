"""
text_preprocessor.py
====================
 
CS329 Financial Report Analyzer -- Text Preprocessing & Feature Engineering
Owner: Pranav
 
Consumes the JSON records produced by `sec_edgar_collector.py` and
`transcript_scraper.py` and adds a `processed` field containing clean,
segmented, linguistically annotated text ready for downstream signal
extraction.
 
Pipeline stages:
    1. FinancialTextCleaner       -- HTML entities / unicode / boilerplate
    2. SECSectionSegmenter        -- 10-K/10-Q item-based sectioning
    3. TranscriptSegmenter        -- prepared remarks vs Q&A + speaker turns
    4. FinancialNLPProcessor      -- spaCy: tokenize, lemmatize, POS,
                                     noun-phrase chunking, NER,
                                     negation detection (negspaCy),
                                     hedge detection (custom matcher),
                                     entity masking
    5. PreprocessingPipeline      -- orchestrates the above end-to-end
 
Output schema (appended to each record under the key "processed"):
    {
      "sections": {
        "mdna": str, "risk_factors": str, "forward_guidance": str,
        "qa_section": [ {"speaker","role","text"}, ... ]   # transcripts only
      },
      "sentences": [
        {
          "sent_id": int,
          "section": str,
          "text": str,
          "text_masked": str,
          "tokens": [ {"text","lemma","pos","is_stop","is_neg"}, ... ],
          "noun_phrases": [str, ...],
          "has_negation": bool,
          "has_hedge": bool
        }, ...
      ],
      "stats": {...}
    }
"""
from __future__ import annotations
 
import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
 
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
 
try:
    from negspacy.negation import Negex  # noqa: F401
    _HAS_NEGSPACY = True
except ImportError:
    _HAS_NEGSPACY = False
 
logger = logging.getLogger(__name__)
 
PREPROCESSING_VERSION = "0.1.0"
 
 
# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------
 
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\.{4,}\s*\d+"),
    re.compile(r"\b(?:xbrl|ixbrl|us-gaap|dei)[:\-][\w\-]+", re.IGNORECASE),
    re.compile(
        r"united states\s+securities and exchange commission.*?"
        r"washington,?\s*d\.?c\.?\s*20549",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"[ \t]+"),   # collapse spaces/tabs (second-to-last)
    re.compile(r"\n{3,}"),   # collapse newlines (last)
]
 
 
class FinancialTextCleaner:
    """Pre-NLP cleanup. Lossy by design but preserves sentence boundaries."""
 
    def clean(self, text: str) -> str:
        if not text:
            return ""
        # Decode HTML entities (&#160;, &amp;, &#8220;) that survived
        # HTML-stripping in the scraper.
        text = html.unescape(text)
        # Normalize unicode (smart quotes, non-breaking spaces, etc.)
        text = unicodedata.normalize("NFKC", text)
        # Replace common ligatures / odd chars
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Apply boilerplate patterns: most substitute with space, last two
        # (whitespace, newlines) get special treatment.
        for i, pat in enumerate(_BOILERPLATE_PATTERNS):
            if i == len(_BOILERPLATE_PATTERNS) - 2:   # whitespace collapse
                text = pat.sub(" ", text)
            elif i == len(_BOILERPLATE_PATTERNS) - 1: # newline collapse
                text = pat.sub("\n\n", text)
            else:
                text = pat.sub(" ", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        return text.strip()
 
 
# ---------------------------------------------------------------------------
# 2. SEC section segmentation
# ---------------------------------------------------------------------------
 
# Title keywords that identify real section headers (vs cross-references).
# Each section we care about has signature phrases that appear only in its
# real header, never in prose references to it.
_SECTION_TITLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "mdna": re.compile(
        r"management'?s\s+discussion\s+and\s+analysis",
        re.IGNORECASE,
    ),
    "risk_factors": re.compile(r"risk\s+factors", re.IGNORECASE),
}
 
_ITEM_HEADER = re.compile(
    # Require a letter immediately after optional punctuation. Also require
    # either whitespace/punctuation separator before the title OR end-of-line,
    # so we reject embedded item-letter patterns like "Item 1A," where the
    # "A" is part of the item designator, not a title starting with "A,".
    # Key change: the punctuation group is no longer optional -- we require
    # SOME separator (period, dash, colon, or whitespace >= 1 char) between
    # the number and the title.
    r"\bitem\s+(?P<num>\d+[aAbB]?)"
    r"(?:\s*[.\-:]\s{0,6}|\s{2,})"          # require . - : OR multiple spaces
    r"(?=[A-Za-z])"
    r"(?P<title>[^\n]{0,120})",
    re.IGNORECASE | re.MULTILINE,
)

# Phrases that appear in cross-references to other sections, never in real
# section headers. When a candidate's trailing title starts with one of
# these, we reject the candidate.
_CROSS_REF_TITLE = re.compile(
    r"^\s*(?:of\s+(?:the|this|our)"
    r"|in\s+(?:the|this|our)"
    r"|under\s+(?:the|this|our)"
    r"|described\s+in|discussed\s+in|referenced\s+in|set\s+forth\s+in"
    r"|to\s+the|included\s+in|and\s+(?:our|this)|see\s+(?:also|part)"
    r"|for\s+(?:the|this|our|a)"
    # NEW: prose continuations. Real section headers start with a noun
    # phrase (e.g. "Risk Factors", "Management's Discussion..."). Anything
    # starting with "We ", "Our ", "This ", "These ", etc. is a sentence
    # that happens to mention "Item NX" in passing.
    r"|we\s+|our\s+|this\s+|these\s+|there\s+|"
    r"any\s+|such\s+|if\s+|when\s+|because\s+|although\s+)",
    re.IGNORECASE,
)
 
_FORWARD_LOOKING_CUES = [
    "expect", "anticipate", "project", "forecast", "outlook",
    "guidance", "plan to", "intend to", "believe", "estimate",
    "may", "will", "could", "would", "should", "going forward",
    "next quarter", "next year", "in the coming", "we target",
]
 
 
@dataclass
class SECSections:
    mdna: str = ""
    risk_factors: str = ""
    forward_guidance: str = ""
 
    def as_dict(self) -> dict[str, str]:
        return {
            "mdna": self.mdna,
            "risk_factors": self.risk_factors,
            "forward_guidance": self.forward_guidance,
        }
 
 
class SECSectionSegmenter:
    """Segment a cleaned 10-K or 10-Q into the sections we care about."""
 
    def segment(self, cleaned_text: str, form_type: str) -> SECSections:
        sections = SECSections()
        form_type = form_type.upper().strip()
        if form_type not in {"10-K", "10-Q"}:
            # 8-K and others: no standard Item structure. Best-effort fallback.
            sections.mdna = cleaned_text
            sections.forward_guidance = self._extract_forward_guidance(cleaned_text)
            return sections
 
        headers = list(_ITEM_HEADER.finditer(cleaned_text))
        if not headers:
            logger.warning(
                "No Item headers found in %s filing; using full body as MD&A.",
                form_type,
            )
            sections.mdna = cleaned_text
            sections.forward_guidance = self._extract_forward_guidance(cleaned_text)
            return sections
 
        # Build a list of (position, item_num, title_text) per candidate.
        candidates = [
            (m.start(), m.group("num").upper(), (m.group("title") or ""))
            for m in headers
            if not _CROSS_REF_TITLE.match(m.group("title") or "")
        ]
 
        # For each section we want, find ALL candidates whose trailing title
        # matches our section-title pattern, compute each body length, and
        # pick the longest. TOC entries and cross-references naturally lose
        # because their bodies are short (next candidate is close behind).
        for section_name, title_pat in _SECTION_TITLE_PATTERNS.items():
            best_body = ""
            best_start: int | None = None
            for i, (pos, _num, title) in enumerate(candidates):
                if not title_pat.search(title):
                    continue
                end_pos = (
                    candidates[i + 1][0]
                    if i + 1 < len(candidates)
                    else len(cleaned_text)
                )
                body = cleaned_text[pos:end_pos].strip()
                if len(body) > len(best_body):
                    best_body = body
                    best_start = pos
            if best_start is not None:
                setattr(sections, section_name, best_body)
            else:
                logger.debug(
                    "Could not locate section '%s' in %s filing.",
                    section_name, form_type,
                )
 
        sections.forward_guidance = self._extract_forward_guidance(
            sections.mdna or cleaned_text
        )
        return sections
 
    def _extract_forward_guidance(self, text: str) -> str:
        """Pull sentences containing forward-looking cue phrases."""
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        keep: list[str] = []
        for s in sentences:
            s_lower = s.lower()
            if any(cue in s_lower for cue in _FORWARD_LOOKING_CUES):
                keep.append(s.strip())
        return " ".join(keep)
 
 
# ---------------------------------------------------------------------------
# 3. Transcript segmentation
# ---------------------------------------------------------------------------
 
# Speaker lines in Motley Fool transcripts typically look like:
#     "Tim Cook -- Chief Executive Officer"
# Require the whole line to match so utterances with mid-sentence em-dashes
# (e.g. "strong demand -- although we remain cautious") don't false-match.
# The name must be name-shaped: each token capitalized, at most 4 tokens.
_SPEAKER_LINE = re.compile(
    r"^[ \t]*"
    r"(?P<n>(?:[A-Z][A-Za-z.\-']{0,20})(?:\s+[A-Z][A-Za-z.\-']{0,20}){0,3})"
    r"\s*(?:--|\u2014)\s*"
    r"(?P<title>[A-Za-z][A-Za-z,&/\- ]{2,80}?)"
    r"[ \t]*$",
    re.MULTILINE,
)
 
_QA_MARKER = re.compile(
    r"(questions?\s*(?:&|and)\s*answers?|q\s*&\s*a\s*session|"
    r"we\s+will\s+now\s+open\s+the\s+(?:call|line)\s+(?:up\s+)?for\s+questions)",
    re.IGNORECASE,
)
 
_ANALYST_TITLE_CUES = ("analyst", "research", "capital", "securities", "bank")
 
 
@dataclass
class TranscriptSections:
    prepared_remarks: str = ""
    qa_section: list[dict[str, str]] = field(default_factory=list)
 
    def as_dict(self) -> dict[str, Any]:
        return {
            "prepared_remarks": self.prepared_remarks,
            "qa_section": self.qa_section,
        }
 
 
class TranscriptSegmenter:
    """Split an earnings call transcript into prepared remarks and Q&A turns."""
 
    def segment(self, cleaned_text: str) -> TranscriptSections:
        result = TranscriptSections()
        if not cleaned_text:
            return result
 
        qa_match = _QA_MARKER.search(cleaned_text)
        if qa_match:
            result.prepared_remarks = cleaned_text[:qa_match.start()].strip()
            qa_body = cleaned_text[qa_match.end():].strip()
        else:
            result.prepared_remarks = cleaned_text
            qa_body = ""
 
        if qa_body:
            result.qa_section = self._parse_turns(qa_body)
        return result
 
    def _parse_turns(self, qa_body: str) -> list[dict[str, str]]:
        turns: list[dict[str, str]] = []
        matches = list(_SPEAKER_LINE.finditer(qa_body))
        for i, m in enumerate(matches):
            speaker = m.group("n").strip()
            title = m.group("title").strip()
            utter_start = m.end()
            utter_end = (
                matches[i + 1].start() if i + 1 < len(matches) else len(qa_body)
            )
            utterance = qa_body[utter_start:utter_end].strip()
            if not utterance:
                continue
            role = (
                "Q"
                if any(cue in title.lower() for cue in _ANALYST_TITLE_CUES)
                else "A"
            )
            turns.append({
                "speaker": speaker,
                "title": title,
                "role": role,
                "text": utterance,
            })
        return turns
 
 
# ---------------------------------------------------------------------------
# 4. spaCy NLP processor: lemmas, POS, negation, hedging, entity masking
# ---------------------------------------------------------------------------
 
# Loughran & McDonald modal/uncertainty words (abbreviated; full list is
# ~300 terms and should live in a resource file in production).
HEDGE_LEMMAS: set[str] = {
    "may", "might", "could", "would", "should", "possibly", "perhaps",
    "potentially", "likely", "unlikely", "appear", "seem", "suggest",
    "approximately", "about", "roughly", "expect", "anticipate", "believe",
    "estimate", "project", "forecast", "assume", "intend", "plan", "hope",
    "contingent", "conditional", "depend", "tentative", "preliminary",
}
 
# Stopwords MINUS negation particles (we keep these so they propagate to
# downstream signal scoring).
NEGATION_WORDS: set[str] = {
    "no", "not", "nor", "never", "neither", "none", "nothing", "nobody",
    "nowhere", "without", "cannot", "cant", "wont", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
}
 
ENTITIES_TO_MASK: dict[str, str] = {
    "MONEY":    "<MONEY>",
    "PERCENT":  "<PCT>",
    "DATE":     "<DATE>",
    "TIME":     "<DATE>",
    "CARDINAL": "<NUM>",
    "QUANTITY": "<NUM>",
    "ORG":      "<ORG>",
    "PERSON":   "<PERSON>",
    "GPE":      "<LOC>",
}
 
 
@Language.factory("hedge_detector")
def _make_hedge_detector(nlp: Language, name: str):
    return HedgeDetector(nlp, name)
 
 
class HedgeDetector:
    """Custom spaCy component that flags sentences containing hedge lemmas."""
 
    def __init__(self, nlp: Language, name: str):
        self.matcher = Matcher(nlp.vocab)
        # Match on LEMMA when a lemmatizer is present, else LOWER.
        # We register both so blank pipelines still flag hedges.
        self.matcher.add("HEDGE_LEMMA", [[{"LEMMA": {"IN": list(HEDGE_LEMMAS)}}]])
        self.matcher.add("HEDGE_LOWER", [[{"LOWER": {"IN": list(HEDGE_LEMMAS)}}]])
        if not Span.has_extension("has_hedge"):
            Span.set_extension("has_hedge", default=False)
 
    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        hedged_token_ids = {
            i for _, start, end in matches for i in range(start, end)
        }
        for sent in doc.sents:
            for tok in sent:
                if tok.i in hedged_token_ids:
                    sent._.has_hedge = True
                    break
        return doc
 
 
class FinancialNLPProcessor:
    """
    Wraps a spaCy pipeline with financial-domain extensions:
      - custom stopword list (keeps negations)
      - negation detection (negspaCy if available, lemma-level fallback)
      - hedge detection (custom component)
      - entity masking for normalized text
      - noun-phrase chunking
    """
 
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            logger.warning(
                "spaCy model %s not available; using blank('en'). "
                "Install with: python -m spacy download %s",
                model, model,
            )
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
 
        if "hedge_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("hedge_detector", last=True)
 
        self._use_negspacy = False
        if _HAS_NEGSPACY and self.nlp.has_pipe("ner"):
            try:
                self.nlp.add_pipe(
                    "negex",
                    config={"ent_types": list(ENTITIES_TO_MASK.keys())},
                )
                self._use_negspacy = True
            except Exception as e:
                logger.debug("Could not add negex: %s", e)
 
        # Remove negation particles from the stopword list
        for w in NEGATION_WORDS:
            self.nlp.vocab[w].is_stop = False
 
    def process_section(
        self, text: str, section_name: str, sent_id_start: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """Run NLP on a section of text. Returns (sentence_records, next_id)."""
        if not text or not text.strip():
            return [], sent_id_start
 
        doc = self.nlp(text)
        records: list[dict[str, Any]] = []
        sid = sent_id_start
 
        for sent in doc.sents:
            stext = sent.text.strip()
            if len(stext) < 3:
                continue
            tokens = [
                {
                    "text":    tok.text,
                    "lemma":   tok.lemma_.lower() if tok.lemma_ else tok.text.lower(),
                    "pos":     tok.pos_,
                    "is_stop": bool(tok.is_stop),
                    "is_neg":  tok.lower_ in NEGATION_WORDS,
                }
                for tok in sent if not tok.is_space
            ]
            records.append({
                "sent_id":      sid,
                "section":      section_name,
                "text":         stext,
                "text_masked":  self._mask_entities(sent),
                "tokens":       tokens,
                "noun_phrases": self._extract_noun_phrases(sent),
                "has_negation": self._detect_negation(sent, tokens),
                "has_hedge":    bool(sent._.has_hedge) if Span.has_extension("has_hedge") else False,
            })
            sid += 1
        return records, sid
 
    def _extract_noun_phrases(self, sent: Span) -> list[str]:
        try:
            return [
                nc.text.lower().strip()
                for nc in sent.noun_chunks
                if 1 < len(nc.text.strip()) < 50
            ]
        except (ValueError, NotImplementedError):
            # noun_chunks requires a parser; blank pipelines won't have one.
            return []
 
    def _mask_entities(self, sent: Span) -> str:
        """Replace named entities with placeholder tokens; lowercase rest."""
        if not sent.ents:
            return sent.text.lower().strip()
        out: list[str] = []
        for tok in sent:
            if tok.ent_type_ in ENTITIES_TO_MASK and tok.ent_iob_ == "B":
                out.append(ENTITIES_TO_MASK[tok.ent_type_])
            elif tok.ent_type_ in ENTITIES_TO_MASK:
                continue  # inside an entity already emitted
            elif tok.is_punct and out:
                # attach punctuation to previous token (no leading space)
                out[-1] = out[-1] + tok.text
            else:
                out.append(tok.text.lower())
        return " ".join(out).strip()
 
    def _detect_negation(self, sent: Span, tokens: list[dict[str, Any]]) -> bool:
        """Prefer negspaCy's ent._.negex if available; else lemma-level check."""
        if self._use_negspacy:
            for ent in sent.ents:
                if getattr(ent._, "negex", False):
                    return True
        return any(t["is_neg"] for t in tokens)
 
 
# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------
 
 
class PreprocessingPipeline:
    """Reads collector JSON records and writes enriched records."""
 
    SEC_FORMS = {"10-K", "10-Q", "8-K"}
 
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        self.cleaner = FinancialTextCleaner()
        self.sec_seg = SECSectionSegmenter()
        self.trn_seg = TranscriptSegmenter()
        self.nlp = FinancialNLPProcessor(model=nlp_model)
 
    def process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        raw = record.get("raw_text", "") or ""
        form = (record.get("form_type") or "").upper()
        cleaned = self.cleaner.clean(raw)
 
        sections_out: dict[str, Any] = {}
        sentences: list[dict[str, Any]] = []
        sid = 0
 
        if form in self.SEC_FORMS:
            secs = self.sec_seg.segment(cleaned, form)
            sections_out = secs.as_dict()
            for name in ("mdna", "risk_factors", "forward_guidance"):
                recs, sid = self.nlp.process_section(
                    getattr(secs, name), section_name=name, sent_id_start=sid,
                )
                sentences.extend(recs)
        else:
            # Treat as transcript
            tsecs = self.trn_seg.segment(cleaned)
            sections_out = tsecs.as_dict()
            recs, sid = self.nlp.process_section(
                tsecs.prepared_remarks,
                section_name="prepared_remarks",
                sent_id_start=sid,
            )
            sentences.extend(recs)
            for i, turn in enumerate(tsecs.qa_section):
                section_name = f"qa_{turn['role']}_{i}"
                recs, sid = self.nlp.process_section(
                    turn["text"],
                    section_name=section_name,
                    sent_id_start=sid,
                )
                sentences.extend(recs)
 
        record = dict(record)  # copy, don't mutate input
        record["processed"] = {
            "sections":  sections_out,
            "sentences": sentences,
            "stats":     self._compute_stats(sentences),
        }
        return record
 
    def process_file(self, in_path: Path, out_path: Path) -> dict[str, Any]:
        record = json.loads(in_path.read_text(encoding="utf-8"))
        processed = self.process_record(record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(processed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return processed["processed"]["stats"]
 
    @staticmethod
    def _compute_stats(sentences: list[dict[str, Any]]) -> dict[str, Any]:
        by_section: dict[str, int] = {}
        for s in sentences:
            by_section[s["section"]] = by_section.get(s["section"], 0) + 1
        return {
            "n_sentences":           len(sentences),
            "n_tokens":              sum(len(s["tokens"]) for s in sentences),
            "n_negated_sentences":   sum(1 for s in sentences if s["has_negation"]),
            "n_hedged_sentences":    sum(1 for s in sentences if s["has_hedge"]),
            "sentences_by_section":  by_section,
            "preprocessing_version": PREPROCESSING_VERSION,
        }