from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from financial_signal_engine import (
    LMDictionary,
    analyze_records,
    build_comparison_rows,
    infer_data_source,
    load_records,
)
from financial_signal_engine_v2 import SignalEngineV2
from run_pipeline import run_full_pipeline


BASE_DIR = Path(__file__).resolve().parent


def _get_default_engine() -> SignalEngineV2:
    """Build the V2 engine once per Streamlit session (cached in session state)."""
    if "signal_engine" not in st.session_state:
        lm_csv_candidates = [
            BASE_DIR / "data" / "lexicons" / "loughran_mcdonald.csv",
            BASE_DIR / "data" / "loughran_mcdonald.csv",
            BASE_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv",
        ]
        lm_csv = next((p for p in lm_csv_candidates if p.exists()), lm_csv_candidates[0])
        st.session_state.signal_engine = SignalEngineV2(LMDictionary.from_csv(lm_csv))
    return st.session_state.signal_engine

COLOR_MAP = {
    "growth": "#1f8f5f",
    "risk": "#b54708",
    "cost_pressure": "#9f1239",
    "net_operating_signal": "#155eef",
}


def _format_document_label(analysis: dict) -> str:
    return f"{analysis['ticker']} | {analysis['form_type']} | {analysis['filing_date']}"


def _filter_analyses(
    analyses: list[dict],
    tickers: list[str],
    report_types: list[str],
    selected_year: int | None,
) -> list[dict]:
    filtered = []
    ticker_set = {ticker.upper() for ticker in tickers if ticker}
    report_type_set = {report_type.upper() for report_type in report_types if report_type}

    for analysis in analyses:
        if ticker_set and analysis["ticker"].upper() not in ticker_set:
            continue
        if report_type_set and analysis["form_type"].upper() not in report_type_set:
            continue
        try:
            parsed = date.fromisoformat(analysis["filing_date"])
        except Exception:
            parsed = None
        if selected_year and parsed and parsed.year != selected_year:
            continue
        filtered.append(analysis)

    return filtered


def _sentence_histogram_frame(sentence_signals: list[dict]) -> pd.DataFrame:
    if not sentence_signals:
        return pd.DataFrame({"range": [], "count": []})

    values = [sentence["net_score"] for sentence in sentence_signals]
    bins = pd.cut(
        values,
        bins=[-10, -4, -2, 0, 2, 4, 10],
        labels=["<= -4", "-4 to -2", "-2 to 0", "0 to 2", "2 to 4", ">= 4"],
        include_lowest=True,
    )
    counts = pd.Series(bins).value_counts(sort=False).reset_index()
    counts.columns = ["range", "count"]
    return counts


def _phrase_table(analysis: dict) -> pd.DataFrame:
    rows = []
    for dim, key in [
        ("Growth", "top_growth_phrases"),
        ("Risk", "top_risk_phrases"),
        ("Cost Pressure", "top_cost_phrases"),
    ]:
        for item in analysis.get(key, [])[:4]:
            rows.append(
                {
                    "Dimension": dim,
                    "Term": item["term"],
                    "Source": item["source"],
                    "Matches": item["count"],
                }
            )
    return pd.DataFrame(rows)


def _sentence_table(analysis: dict) -> pd.DataFrame:
    rows = []
    for sentence in analysis["top_sentences"][:20]:
        rows.append(
            {
                "Section": sentence["section"],
                "Growth": round(sentence["growth"], 2),
                "Risk": round(sentence["risk"], 2),
                "Cost": round(sentence["cost_pressure"], 2),
                "Net": round(sentence["net_score"], 2),
                "Negated": sentence["has_negation"],
                "Hedged": sentence["has_hedge"],
                "Sentence": sentence["text"],
            }
        )
    return pd.DataFrame(rows)


def _score_tone(value: float, positive: bool = True) -> tuple[str, str]:
    if positive:
        if value >= 6:
            return "High", "#1f8f5f"
        if value >= 3:
            return "Moderate", "#155eef"
        return "Low", "#667085"
    if value >= 6:
        return "Elevated", "#b42318"
    if value >= 3:
        return "Moderate", "#b54708"
    return "Contained", "#667085"


def _metric_card(title: str, value: float, subtitle: str, accent: str) -> str:
    return f"""
    <div style="
        background: linear-gradient(160deg, rgba(255,255,255,0.96), rgba(240,244,248,0.92));
        border: 1px solid rgba(16,24,40,0.08);
        border-left: 8px solid {accent};
        border-radius: 18px;
        padding: 18px 18px 16px 18px;
        box-shadow: 0 10px 30px rgba(16,24,40,0.08);
        min-height: 144px;
    ">
        <div style="font-size: 0.84rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #475467;">
            {title}
        </div>
        <div style="font-size: 2.35rem; font-weight: 800; color: #101828; line-height: 1.1; margin-top: 8px;">
            {value:.2f}
        </div>
        <div style="
            display: inline-block;
            margin-top: 10px;
            padding: 6px 10px;
            border-radius: 999px;
            background: {accent}15;
            color: {accent};
            font-size: 0.84rem;
            font-weight: 700;
        ">
            {subtitle}
        </div>
    </div>
    """


def _render_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
            max-width: 1340px;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(21,94,239,0.10), transparent 24%),
                radial-gradient(circle at top left, rgba(31,143,95,0.10), transparent 20%),
                linear-gradient(180deg, #f8fafc 0%, #eef2f6 100%);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(16,24,40,0.08);
            box-shadow: 0 8px 24px rgba(16,24,40,0.06);
        }
        div[data-testid="stExpander"] details {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(16,24,40,0.08);
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(primary: dict) -> None:
    net = primary["scores"]["net_operating_signal"]
    net_label = "Growth exceeds risk in this sample" if net >= 0 else "Risk outweighs growth in this sample"
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #155eef 55%, #1f8f5f 100%);
            border-radius: 24px;
            padding: 28px 30px 24px 30px;
            color: white;
            box-shadow: 0 18px 45px rgba(15,23,42,0.22);
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.86rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.88;">
                Financial Report Analyzer
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; line-height: 1.08; margin-top: 10px;">
                {primary['company_name']} ({primary['ticker']})
            </div>
            <div style="font-size: 1rem; opacity: 0.92; margin-top: 10px;">
                {primary['form_type']} • {primary['filing_date']} • {primary['source']}
            </div>
            <div style="
                display: inline-block;
                margin-top: 18px;
                padding: 9px 14px;
                border-radius: 999px;
                background: rgba(255,255,255,0.16);
                border: 1px solid rgba(255,255,255,0.18);
                font-weight: 700;
            ">
                {net_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_row(primary: dict) -> None:
    growth_label, growth_color = _score_tone(primary["scores"]["growth"], positive=True)
    risk_label, risk_color = _score_tone(primary["scores"]["risk"], positive=False)
    cost_label, cost_color = _score_tone(primary["scores"]["cost_pressure"], positive=False)
    net_positive = primary["scores"]["net_operating_signal"] >= 0
    net_label, net_color = _score_tone(
        abs(primary["scores"]["net_operating_signal"]), positive=net_positive
    )
    if not net_positive:
        net_label = "Negative balance"
        net_color = "#b42318"

    z = primary.get("zscores") or {}
    z_label = z.get("reference_label") if z else None

    cards = [
        (_metric_card("Growth", primary["scores"]["growth"], growth_label, growth_color), z.get("growth")),
        (_metric_card("Risk", primary["scores"]["risk"], risk_label, risk_color), z.get("risk")),
        (_metric_card("Cost Pressure", primary["scores"]["cost_pressure"], cost_label, cost_color), None),
        (_metric_card("Net Operating Signal", primary["scores"]["net_operating_signal"], net_label, net_color), z.get("net_operating_signal")),
    ]
    for column, (card, z_val) in zip(st.columns(4), cards):
        column.markdown(card, unsafe_allow_html=True)
        if z_val is not None:
            z_color = "#1f8f5f" if z_val >= 0 else "#b42318"
            ref = f" vs {z_label}" if z_label else ""
            column.markdown(
                f'<div style="font-size:0.8rem;color:#475467;margin-top:6px;padding-left:2px;">'
                f'z&#8209;score: <b style="color:{z_color};">{z_val:+.2f}</b>'
                f'<span style="opacity:0.7;">{ref}</span></div>',
                unsafe_allow_html=True,
            )


def _render_key_takeaways(primary: dict) -> None:
    strongest_dimension = max(
        ["growth", "risk", "cost_pressure"],
        key=lambda key: abs(primary["scores"][key]),
    )
    all_phrases = (
        primary.get("top_growth_phrases", [])
        + primary.get("top_risk_phrases", [])
        + primary.get("top_cost_phrases", [])
    )
    dominant_phrase = all_phrases[0]["term"] if all_phrases else "no strong phrase match"
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(16,24,40,0.08);
            border-radius: 18px;
            padding: 18px 20px;
            margin: 1rem 0 1.25rem 0;
            box-shadow: 0 8px 24px rgba(16,24,40,0.06);
        ">
            <div style="font-size: 0.84rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: #475467;">
                Quick Read
            </div>
            <div style="font-size: 1.06rem; color: #101828; margin-top: 10px; line-height: 1.5;">
                The strongest output dimension for this report is
                <span style="font-weight: 800; color: {COLOR_MAP[strongest_dimension]};">{strongest_dimension.replace('_', ' ')}</span>.
                The current net operating signal is
                <span style="font-weight: 800; color: {COLOR_MAP['net_operating_signal']};">{primary['scores']['net_operating_signal']:.2f}</span>,
                and one of the strongest phrase-level drivers in the current document is
                <span style="font-weight: 800;">{dominant_phrase}</span>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Financial Report Analyzer",
        page_icon="FRA",
        layout="wide",
    )
    _render_dashboard_css()

    st.markdown(
        """
        <div style="margin-bottom: 0.35rem;">
            <div style="font-size: 2.5rem; font-weight: 900; color: #101828; line-height: 1;">
                Financial Report Analyzer
            </div>
            <div style="font-size: 1.02rem; color: #475467; margin-top: 8px;">
                Explainable earnings-language dashboard for growth, risk, cost pressure, and net operating signal.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "loaded_records" not in st.session_state:
        st.session_state.loaded_records = load_records(BASE_DIR)

    records = st.session_state.loaded_records
    analyses = analyze_records(records, engine=_get_default_engine())

    if not analyses:
        st.error(
            "No processed data or demo data was found. Add files under data/processed/ "
            "or keep demo_data/sample_documents.json in the repo."
        )
        return

    st.sidebar.header("Controls")
    st.sidebar.subheader("Fetch Reports")
    ticker_text = st.sidebar.text_input("Ticker", value="AAPL")
    current_year = date.today().year
    year_options = list(range(current_year, current_year - 5, -1))
    selected_year = st.sidebar.selectbox("Year", options=year_options, index=0)
    report_type_options = ["10-K", "10-Q", "EARNINGS_CALL"]
    selected_report_types = st.sidebar.multiselect(
        "Report Type",
        options=report_type_options,
        default=["10-K", "10-Q"],
    )
    max_per_type = st.sidebar.selectbox("Documents Per Type", options=[1, 2, 3, 4], index=1)
    fetch_clicked = st.sidebar.button("Fetch Data", use_container_width=True, type="primary")

    start_date = date(selected_year, 1, 1)
    end_date = date(selected_year, 12, 31)

    if fetch_clicked:
        tickers = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]
        sec_forms = [report for report in selected_report_types if report != "EARNINGS_CALL"]
        include_transcripts = "EARNINGS_CALL" in selected_report_types

        with st.spinner("Running data pipeline and loading selected reports..."):
            records = run_full_pipeline(
                tickers=tickers or ["AAPL"],
                form_types=sec_forms or ["10-K", "10-Q"],
                max_per_type=max_per_type,
                skip_sec=not bool(sec_forms),
                skip_transcripts=not include_transcripts,
                start_date=start_date,
                end_date=end_date,
            )
            st.session_state.loaded_records = records
            analyses = analyze_records(records, engine=_get_default_engine())

    st.sidebar.subheader("View Filters")
    selected_tickers = [
        ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()
    ]
    if not selected_tickers:
        selected_tickers = sorted({analysis["ticker"] for analysis in analyses})

    analyses = _filter_analyses(
        analyses,
        tickers=selected_tickers,
        report_types=selected_report_types,
        selected_year=selected_year,
    )

    if not analyses:
        st.warning("No reports matched the selected ticker, year, and report type filters.")
        return

    data_source = infer_data_source(BASE_DIR)
    if data_source == "pipeline_output":
        st.sidebar.write("Data source: `data/filings + data/transcripts`")
    else:
        st.sidebar.write(f"Data source: `{data_source}`")

    labels = [_format_document_label(analysis) for analysis in analyses]
    label_to_analysis = dict(zip(labels, analyses))

    selected_labels = st.sidebar.multiselect(
        "Select reports to compare",
        options=labels,
        default=labels[: min(2, len(labels))],
    )

    if not selected_labels:
        st.info("Select at least one report from the sidebar.")
        return

    selected_analyses = [label_to_analysis[label] for label in selected_labels]
    primary = selected_analyses[0]

    _render_hero(primary)
    if "method" in primary:
        m = primary["method"]
        st.caption(
            f"Engine: `{m['type']}` v{m['engine_version']} — "
            f"{m['lm_growth_words']} growth / {m['lm_risk_words']} risk words loaded"
        )

    _render_metric_row(primary)
    _render_key_takeaways(primary)

    summary_col, histogram_col = st.columns([1, 1])

    with summary_col:
        st.markdown("### Signal Profile")
        st.caption("High-level output balance across the four report-level metrics.")
        profile_frame = pd.DataFrame(
            {
                "dimension": ["growth", "risk", "cost_pressure", "net_operating_signal"],
                "score": [
                    primary["scores"]["growth"],
                    primary["scores"]["risk"],
                    primary["scores"]["cost_pressure"],
                    primary["scores"]["net_operating_signal"],
                ],
            }
        ).set_index("dimension")
        st.bar_chart(profile_frame)

    with histogram_col:
        st.markdown("### Sentence-Level Distribution")
        st.caption("How net signal is distributed across matched sentences in the selected report.")
        histogram_frame = _sentence_histogram_frame(primary["top_sentences"]).set_index("range")
        st.bar_chart(histogram_frame)

    st.markdown("### Top Contributing Phrases")
    st.caption("Phrase-level evidence that drives the report score profile.")
    st.dataframe(_phrase_table(primary), use_container_width=True, hide_index=True)

    st.markdown("### Explainability Trace")
    st.caption("Sentence-by-sentence breakdown of where the major output signals came from.")
    st.dataframe(_sentence_table(primary), use_container_width=True, hide_index=True)

    with st.expander("Lexicon evidence for the selected report"):
        st.write(
            "This engine uses Loughran-McDonald dictionary category matches and curated "
            "multi-word phrases to produce sentence-level growth, risk, and cost-pressure scores. "
            "The table below shows the top 10 sentences by signal strength with their lexicon evidence."
        )
        if primary["top_sentences"]:
            internal_rows = []
            for sentence in primary["top_sentences"][:10]:
                text = sentence["text"]
                internal_rows.append(
                    {
                        "Sentence": text[:100] + "…" if len(text) > 100 else text,
                        "LM Growth": ", ".join(sentence["lm_growth_hits"]),
                        "LM Risk": ", ".join(sentence["lm_risk_hits"]),
                        "Phrases": ", ".join(
                            sentence["phrase_growth_hits"]
                            + sentence["phrase_risk_hits"]
                            + sentence["phrase_cost_hits"]
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(internal_rows), use_container_width=True, hide_index=True)

    st.markdown("### Cross-Report Comparison")
    st.caption("Direct comparison of the most important output metrics across selected reports.")
    comparison_frame = pd.DataFrame(build_comparison_rows(selected_analyses)).set_index("label")
    st.bar_chart(
        comparison_frame[["growth", "risk", "cost_pressure", "net_operating_signal"]]
    )

    if len(selected_analyses) > 1:
        st.markdown("### Net Operating Signal Trend")
        st.caption("Trendline view of overall signal balance across your selected comparison set.")
        trend_frame = comparison_frame.reset_index()[["label", "net_operating_signal"]]
        st.line_chart(trend_frame.set_index("label"))

if __name__ == "__main__":
    main()
