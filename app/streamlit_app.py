"""MedVerify web app: check whether an answer to a medical question is factual.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Runs the same verify_item as the experiments and `python -m medverify verify` (through
medverify.service), so caching, re-asks and logging are the same. Research prototype, not medical advice.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, so `app` imports when run by streamlit

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app import helpers as H  # noqa: E402
from medverify import service  # noqa: E402
from medverify.pipeline import VerifierCallError  # noqa: E402

load_dotenv()
st.set_page_config(page_title="MedVerify", layout="centered")


@st.cache_resource
def verifiers() -> dict:
    return service.make_verifiers()


def show_state(state) -> None:
    """The result of one check: decision, agreement, each verifier, the evidence, the cost."""
    st.markdown(H.badge_html(state.final), unsafe_allow_html=True)
    st.markdown(f"**Agreement:** {H.agreement(state)}")
    reason = H.abstain_reason(state)
    if reason:
        st.info(reason)
    st.subheader("Verifiers")
    for row in H.verifier_rows(state):
        with st.expander(f"{row['verifier']}: {row['vote']} (round {row['round']})"):
            st.write(row["reason"])
            if row["reasked"]:
                st.caption("The first reply had no clear verdict, so this verifier was asked once more.")
    st.subheader("Evidence")
    st.markdown("\n".join(H.evidence_markdown(state)) or "No abstracts were used.")
    blocked = H.blocked_markdown(state)
    if blocked:
        st.markdown("**Blocked to stop source leakage**\n" + "\n".join(blocked))
    st.markdown(f"**Second round:** {H.second_round_text(state)}")
    st.caption(H.cost_line(state))
    with st.expander("Full trajectory (JSON)"):
        st.json(state.model_dump())


def check_one(rule: str, status: dict) -> None:
    with st.form("one"):
        question = st.text_area("Question", placeholder="Does ... in ...?")
        answer = st.text_area("Answer to check")
        pmid = st.text_input("Source PubMed ID (optional)",
                             help="If you know the abstract the answer came from, it is kept out of the evidence.")
        evidence = st.selectbox("Evidence", list(H.EVIDENCE_LABELS), format_func=H.EVIDENCE_LABELS.get)
        submitted = st.form_submit_button("Check")
    if not submitted:
        return
    if not question.strip() or not answer.strip():
        st.warning("Please enter both a question and an answer.")
        return
    if evidence == "source" and not pmid.strip():
        st.warning("The source-abstract mode needs the PubMed ID of that abstract.")
        return
    if not H.ready(status):
        st.error("The verifier models are not available yet: see Setup check in the sidebar.")
        return
    with st.spinner("Searching PubMed and asking the three verifiers..."):
        try:
            state = service.verify_one(question, answer, pmid.strip() or None, verifiers(), evidence, rule)
        except VerifierCallError as err:
            st.error(H.model_error_text(err))
            return
        except Exception as err:  # a bad PMID, no network, ...: a message, not a crash
            st.error(f"Could not check this answer: {err}")
            return
    show_state(state)


def check_csv(rule: str, status: dict) -> None:
    upload = st.file_uploader("CSV with the columns question, answer and optionally pmid", type="csv")
    evidence = st.selectbox("Evidence", list(H.EVIDENCE_LABELS), format_func=H.EVIDENCE_LABELS.get, key="csv_evidence")
    if upload is None or not st.button("Run"):
        return
    try:
        rows = service.parse_rows(upload.getvalue().decode("utf-8-sig"), upload.name)
    except (UnicodeDecodeError, ValueError) as err:
        st.error(str(err))
        return
    if not H.ready(status):
        st.error("The verifier models are not available yet: see Setup check in the sidebar.")
        return
    bar = st.progress(0.0, text=f"0 of {len(rows)} checked")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = service.run_batch(rows, verifiers(), evidence, rule, states_path=Path(tmp) / "states.jsonl",
                                    progress=lambda n, total: bar.progress(n / total, text=f"{n} of {total} checked"))
    except VerifierCallError as err:
        st.error(H.model_error_text(err))
        return
    except Exception as err:
        st.error(f"Could not check this file: {err}")
        return
    st.dataframe([{c: r.get(c, "") for c in H.RESULT_COLUMNS} for r in out])
    st.download_button("Download the results (CSV)", H.results_csv(out), file_name="medverify_results.csv", mime="text/csv")


st.title("MedVerify")
st.write("Checks whether an answer to a medical question is factual, using PubMed evidence and three independent "
         "AI verifiers (Llama 3.1 8B, Mistral 7B, Phi-4-mini). **ACCEPT** = factual, **REJECT** = hallucinated, "
         "**NOT SURE** = sent for human review.")
st.caption(H.DISCLAIMER)

with st.sidebar:
    st.header("Setup check")
    status = H.setup_status()
    for level, text in H.setup_messages(status):
        getattr(st, level)(text)
    choice = st.selectbox("Second evidence round", ["auto", "off", "split", "no-majority"],
                          help="auto = the trigger chosen on the validation split, if experiments/final_config.json exists")
    rule, why = service.resolve_second_round(choice)
    st.caption(f"Using {rule}: {why}.")
    st.caption(H.DISCLAIMER)

one, many = st.tabs(["Check one answer", "Check a CSV file"])
with one:
    check_one(rule, status)
with many:
    check_csv(rule, status)
