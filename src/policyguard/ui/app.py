"""Minimal Streamlit UI for PolicyGuard, talking to the FastAPI service (policyguard.api.app).

Run the API first (see README), then:
    streamlit run src/policyguard/ui/app.py
"""

from __future__ import annotations

import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(page_title="PolicyGuard", page_icon="📄")
st.title("PolicyGuard")
st.caption("Ask a question about company HR/IT policy. Answers are grounded in retrieved policy excerpts with citations.")

api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": str, "response": dict}
if "pending" not in st.session_state:
    st.session_state.pending = None  # {"thread_id": str, "question": str, "response": dict} awaiting review


def render_response(response: dict) -> None:
    st.write(response.get("answer") or "*(no answer)*")

    citations = response.get("citations") or []
    if citations:
        st.caption("Sources: " + ", ".join(f"{c['doc_id']} · {c['section']}" for c in citations))

    invalid = response.get("invalid_citations") or []
    if invalid:
        st.warning(
            "Model cited sources not present in the retrieved context: "
            + ", ".join(f"{c['doc_id']} · {c['section']}" for c in invalid)
        )

    if response.get("status") == "cannot_answer":
        st.info("No relevant policy documents were found for this question.")
    if response.get("human_reviewed"):
        st.caption("Reviewed by a human before being returned.")


def ask(question: str) -> None:
    try:
        resp = requests.post(f"{api_url}/ask", json={"question": question}, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Request to PolicyGuard API failed: {exc}")
        return

    data = resp.json()
    if data["status"] == "needs_review":
        st.session_state.pending = {"thread_id": data["thread_id"], "question": question, "response": data}
    else:
        st.session_state.history.append({"question": question, "response": data})


def resolve(action: str, answer: str | None = None) -> None:
    pending = st.session_state.pending
    payload = {"thread_id": pending["thread_id"], "action": action}
    if answer is not None:
        payload["answer"] = answer

    try:
        resp = requests.post(f"{api_url}/resolve", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Request to PolicyGuard API failed: {exc}")
        return

    st.session_state.history.append({"question": pending["question"], "response": resp.json()})
    st.session_state.pending = None


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_response(turn["response"])

if st.session_state.pending:
    pending = st.session_state.pending
    with st.chat_message("user"):
        st.write(pending["question"])
    with st.chat_message("assistant"):
        st.write("Draft answer (unverified, pending human review):")
        st.write(pending["response"]["answer"])
        if pending["response"].get("invalid_citations"):
            st.warning(
                "Cited sources not present in the retrieved context: "
                + ", ".join(f"{c['doc_id']} · {c['section']}" for c in pending["response"]["invalid_citations"])
            )

        col1, col2, col3 = st.columns(3)
        if col1.button("Approve"):
            resolve("approve")
            st.rerun()
        if col3.button("Reject"):
            resolve("reject")
            st.rerun()

        edited = st.text_area("Edit answer instead", key="edit_answer")
        if col2.button("Submit edit"):
            if edited.strip():
                resolve("edit", answer=edited.strip())
                st.rerun()
            else:
                st.error("Enter a replacement answer before submitting.")
else:
    question = st.chat_input("Ask a policy question...")
    if question:
        ask(question)
        st.rerun()
