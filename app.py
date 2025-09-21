# app.py
import os
import streamlit as st
import pandas as pd
from collections import Counter

from utils.extract_text import extract_text
from utils.scorer import evaluate_resume
from utils.preprocess import extract_sections
from utils.llm_utils import generate_feedback

st.set_page_config(page_title="Resume Relevance — Final MVP", layout="wide")

# ensure temp dir exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# session state
if "results" not in st.session_state:
    st.session_state.results = []
if "jd_texts" not in st.session_state:
    st.session_state.jd_texts = []
if "jd_names" not in st.session_state:
    st.session_state.jd_names = []

st.title("🚀 Automated Resume Relevance Check — Final MVP")

# Sidebar
st.sidebar.header("Settings")
use_llm = st.sidebar.checkbox("Enable LLM suggestions (OpenAI key required)", value=False)
hard_weight = st.sidebar.slider("Hard score weight", 0.0, 1.0, 0.5)
semantic_weight = round(1.0 - hard_weight, 2)
st.sidebar.markdown("---")
st.sidebar.write("Flow: Upload JD(s) → Upload Resumes → Run Evaluation → Inspect & Download")

# Section: Job Descriptions
st.header("1) Job Description(s)")
jd_text_input = st.text_area("Paste primary JD here (optional)", height=140)
uploaded_jds = st.file_uploader("Or upload JD files (txt/pdf/docx)", type=["txt","pdf","docx"], accept_multiple_files=True)

if st.button("Add pasted JD") and jd_text_input.strip():
    st.session_state.jd_texts.append(jd_text_input.strip())
    st.session_state.jd_names.append("pasted_JD")
    st.success("Primary JD added.")

if uploaded_jds:
    for f in uploaded_jds:
        path = os.path.join(TEMP_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        txt = extract_text(path)
        st.session_state.jd_texts.append(txt)
        st.session_state.jd_names.append(f.name)
    st.success(f"Added {len(uploaded_jds)} JD(s).")

if st.session_state.jd_texts:
    st.markdown("**Current JDs:**")
    for i, name in enumerate(st.session_state.jd_names):
        st.write(f"{i}: {name}")

# Section: Resumes Upload
st.header("2) Upload resumes (multiple)")
uploaded_resumes = st.file_uploader("Upload resume files (pdf/docx/txt)", type=["pdf","docx","txt"], accept_multiple_files=True)

# Run Evaluation
if st.button("Run Evaluation"):
    if not st.session_state.jd_texts:
        st.error("Please add at least one JD first.")
    elif not uploaded_resumes:
        st.error("Please upload at least one resume.")
    else:
        st.session_state.results = []  # reset previous results
        with st.spinner("Evaluating resumes..."):
            for f in uploaded_resumes:
                # save to temp and extract
                path = os.path.join(TEMP_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                resume_txt = extract_text(path)
                # evaluate against each JD and pick best
                per_jd = []
                for j_idx, jd in enumerate(st.session_state.jd_texts):
                    r = evaluate_resume(f.name, resume_txt, jd, hard_weight=hard_weight, semantic_weight=semantic_weight)
                    r["jd_index"] = j_idx
                    r["jd_name"] = st.session_state.jd_names[j_idx] if j_idx < len(st.session_state.jd_names) else f"JD_{j_idx}"
                    per_jd.append(r)
                # pick best final_score (tie-breaker semantic)
                best = sorted(per_jd, key=lambda x: (x["final_score"], x["semantic_score"]), reverse=True)[0]
                st.session_state.results.append(best)
        st.success("Evaluation completed!")

# Show results table
if st.session_state.results:
    df = pd.DataFrame(st.session_state.results).sort_values("final_score", ascending=False).reset_index(drop=True)
    st.subheader("Results")
    st.dataframe(df[["filename","jd_name","final_score","verdict","matched_skills","missing_skills"]])

    # analytics - skill gap
    st.subheader("Skill-gap analytics")
    cnt = Counter()
    for m in df["missing_skills"].fillna(""):
        for s in [x.strip().lower() for x in str(m).split(",") if x.strip()]:
            cnt[s] += 1
    if cnt:
        top = cnt.most_common(15)
        skills = [k for k,v in top]
        counts = [v for k,v in top]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, max(3, len(skills)*0.25)))
        ax.barh(range(len(skills))[::-1], counts[::-1])
        ax.set_yticks(range(len(skills)))
        ax.set_yticklabels(skills[::-1])
        ax.set_xlabel("Missing count (how many resumes lack this skill)")
        ax.set_title("Top missing skills across uploaded resumes")
        st.pyplot(fig)
    else:
        st.info("No missing-skills data to show yet.")

    # Inspect candidate
    st.subheader("Inspect candidate")
    idx = st.number_input("Choose candidate index", min_value=0, max_value=len(df)-1, value=0, step=1)
    cand = st.session_state.results[int(idx)]
    st.markdown(f"### {cand['filename']}  — JD: {cand['jd_name']}")
    st.write("Final score:", cand["final_score"])
    st.write("Verdict:", cand["verdict"])
    st.write("Matched skills:", cand["matched_skills"] or "—")
    st.write("Missing skills:", cand["missing_skills"] or "—")

    # Show sections extracted (basic)
    if st.checkbox("Show parsed resume sections"):
        sections = extract_sections(cand["resume_text"])
        for k,v in sections.items():
            st.markdown(f"**{k.title()}**")
            st.write(v[:1000])  # show first 1000 chars

    # Score simulator: tick missing skills
    st.markdown("**Score Simulator** — tick missing skills to simulate improvement")
    missing_list = [s.strip() for s in cand["missing_skills"].split(",")] if cand["missing_skills"] else []
    if missing_list:
        cols = st.columns(3)
        selected = []
        for i, m in enumerate(missing_list):
            c = cols[i % 3].checkbox(m, key=f"sim_{idx}_{i}")
            if c:
                selected.append(m)
        if st.button("Simulate selected improvements"):
            num = len(selected)
            jd_skill_count = max(1, len([x for x in cand["matched_skills"].split(",") if x.strip()]) + len(missing_list))
            per_skill_hard = 100.0 / jd_skill_count
            per_skill_sem = 3.0
            new_hard = min(100.0, cand["hard_score"] + per_skill_hard * num)
            new_sem = min(100.0, cand["semantic_score"] + per_skill_sem * num)
            new_final = round(new_hard * hard_weight + new_sem * semantic_weight, 2)
            st.write("New projected final score:", new_final)

    # LLM suggestions
    if use_llm:
        st.subheader("LLM Suggestions (actionable)")
        try:
            fb = generate_feedback(st.session_state.jd_texts[cand["jd_index"]], cand["resume_text"],
                                   missing_skills=[s for s in cand["missing_skills"].split(",") if s.strip()])
            if isinstance(fb, dict) and "suggestions" in fb:
                st.write("Suggestions:")
                for s in fb["suggestions"]:
                    st.write("- ", s)
                if fb.get("verdict"):
                    st.write("LLM verdict:", fb["verdict"])
            else:
                # textual fallback
                st.write(fb)
        except Exception as e:
            st.error("LLM error: " + str(e))

    # download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv, file_name="results.csv", mime="text/csv")
