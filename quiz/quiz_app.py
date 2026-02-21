
import json
import random
from typing import List, Dict, Any, Optional
import streamlit as st

JSON_FILE = "quiz_data.json"


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize
    for q in data:
        q["type"] = str(q.get("type", "tf")).lower()
        if q["type"] == "tf":
            q.setdefault("options", ["True", "False"])
        q.setdefault("answer", "")
    return data


def sample_questions(data: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    shuffled = data.copy()
    rnd.shuffle(shuffled)
    return shuffled[: max(0, min(n, len(shuffled)))]


def check_tf(selected: Optional[str], correct: str) -> bool:
    return (selected is not None) and (str(selected).strip() == str(correct).strip())


def check_multi(selected: List[int], correct: List[int]) -> bool:
    return set(selected) == set(correct)


def highlight_option(text: str, is_correct: bool, is_selected: bool) -> None:
    """Render an option line, highlighting correct options, while preserving LaTeX rendering.

    IMPORTANT: Streamlit does not render MathJax inside raw HTML blocks, so we use st.markdown only.
    """
    prefix = "✅" if is_correct else ("❌" if is_selected else "⬜")
    suffix = " *(selected)*" if is_selected else ""
    # Use markdown so inline $...$ formulas are rendered by MathJax.
    st.markdown(f"{prefix} {text}{suffix}")


def reset_quiz():
    """Reset quiz state and clear widget values so the UI truly starts fresh."""
    # Clear widget keys (checkbox/radio) from previous runs
    prefixes = ("q_", "tf_", "multi_")
    for k in list(st.session_state.keys()):
        if k.startswith(prefixes):
            del st.session_state[k]

    st.session_state.page = "quiz"
    st.session_state.questions = []
    st.session_state.user_answers = {}
    st.session_state.results = {}
    # Bump a nonce used to namespace widget keys if needed
    st.session_state.quiz_nonce = int(st.session_state.get("quiz_nonce", 0)) + 1



def main():
    st.set_page_config(page_title="ML Quiz 654AA", layout="centered")
    st.title("Machine Learning Quiz 654AA")

    # Session state init
    if "page" not in st.session_state:
        st.session_state.page = "quiz"  # quiz | results
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "quiz_nonce" not in st.session_state:
        st.session_state.quiz_nonce = 0

    with st.sidebar:
        st.header("Settings")

        if st.button("New quiz", use_container_width=True):
            reset_quiz()
            st.rerun()

        n_questions = st.number_input("Questions", min_value=1, max_value=200, value=10, step=1, key="n_questions")
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="seed")
        show_explain = st.checkbox("Show explanation", value=True)
        if st.session_state.page == "results":
            if st.button("New quiz"):
                reset_quiz()
                st.rerun()

    # Load & sample whenever settings change (n_questions / seed) or when starting a new quiz
    if st.session_state.page == "quiz":
        current_sig = (int(n_questions), int(seed), int(st.session_state.quiz_nonce))
        if ("sample_sig" not in st.session_state) or (st.session_state.sample_sig != current_sig) or (not st.session_state.questions):
            try:
                data = load_questions(JSON_FILE)
            except Exception as e:
                st.error(f"Errore loading JSON: {e}")
                st.stop()
            st.session_state.questions = sample_questions(data, int(n_questions), int(seed) + int(st.session_state.quiz_nonce))
            st.session_state.sample_sig = current_sig

    if st.session_state.page == "quiz":
        st.subheader("Questions")

        user_answers: Dict[int, Any] = {}

        for idx, q in enumerate(st.session_state.questions, start=1):
            qkey = f"q_{st.session_state.quiz_nonce}_{q['id']}"
            st.markdown("---")
            # No Q1 / IDs shown: just show the question text
            st.markdown(f"**{q['question']}**")

            if q["type"] == "tf":
                # No default selection: index=None
                sel = st.radio(
                    "Select:",
                    q["options"],
                    index=None,
                    key=f"{qkey}_tf",
                    horizontal=True,
                )
                user_answers[q["id"]] = sel
            elif q["type"] == "multi":
                selected: List[int] = []
                for i, opt in enumerate(q.get("options", []), start=1):
                    if st.checkbox(opt, key=f"{qkey}_m_{i}"):
                        selected.append(i)
                user_answers[q["id"]] = selected
            else:
                st.warning(f"Unknonw question: {q.get('type')}")

        st.markdown("---")
        if st.button("Submit", type="primary"):
            # Validate TF answered
            missing = []
            for q in st.session_state.questions:
                if q["type"] == "tf" and user_answers.get(q["id"], None) is None:
                    missing.append(q["question"])
            if missing:
                st.error("Answer all missing questions before submitting.")
                st.stop()

            # Compute results
            score = 0
            per_q = []
            for q in st.session_state.questions:
                qid = q["id"]
                if q["type"] == "tf":
                    ok = check_tf(user_answers[qid], q.get("correct", ""))
                else:
                    ok = check_multi(user_answers[qid], q.get("correct", []))
                per_q.append({"id": qid, "ok": ok})
                if ok:
                    score += 1

            st.session_state.user_answers = user_answers
            st.session_state.results = {"score": score, "total": len(st.session_state.questions), "per_q": per_q}
            st.session_state.page = "results"
            st.rerun()

    else:
        # RESULTS PAGE
        res = st.session_state.results
        st.subheader("Results")
        st.success(f"Final score: **{res.get('score', 0)} / {res.get('total', 0)}**")
        st.caption("Note: MULTI option answer requires all correct answers to be selected.")

        st.markdown("---")
        # Map for quick ok lookup
        ok_map = {x["id"]: x["ok"] for x in res.get("per_q", [])}

        for q in st.session_state.questions:
            qid = q["id"]
            ok = ok_map.get(qid, False)

            st.markdown(f"### {'✅' if ok else '❌'} {q['question']}")
            ua = st.session_state.user_answers.get(qid)

            if q["type"] == "tf":
                correct_tf = str(q.get("correct", "")).strip()
                ua_tf = None if ua is None else str(ua).strip()
                st.write("**Options (correct answer highlighted):**")
                for opt in q.get("options", ["True", "False"]):
                    highlight_option(
                        opt,
                        is_correct=(str(opt).strip() == correct_tf),
                        is_selected=(ua_tf == str(opt).strip()),
                    )
            else:
                correct_idx = q.get("correct", [])
                opts = q.get("options", [])
                selected_idx = ua or []
                st.write("**Options (correct answer highlighted):**")
                for i, opt in enumerate(opts, start=1):
                    highlight_option(opt, is_correct=(i in correct_idx), is_selected=(i in selected_idx))

            if show_explain and str(q.get("answer", "")).strip():
                st.markdown("**Explanation:**")
                st.markdown(q["answer"])
            st.markdown("---")


if __name__ == "__main__":
    main()
