# utils/scorer.py
from .preprocess import extract_skills_from_text, clean_text
from .embeddings import embed_text, cosine_sim
from rapidfuzz import fuzz

def hard_match_score(jd_text, resume_text, skill_vocab=None):
    jd_skills = extract_skills_from_text(jd_text, skill_vocab)
    resume_skills = extract_skills_from_text(resume_text, skill_vocab)
    matched = []
    missing = []
    for s in jd_skills:
        if s in resume_skills:
            matched.append(s)
        else:
            # fuzzy fallback using resume text tokens
            found = False
            for r in resume_skills:
                if fuzz.token_sort_ratio(s, r) >= 80:
                    found = True
                    break
            if found:
                matched.append(s)
            else:
                missing.append(s)
    if len(jd_skills) == 0:
        return 0.0, matched, missing
    score = (len(matched) / len(jd_skills)) * 100.0
    return round(score,2), matched, missing

def semantic_score(jd_text, resume_text):
    try:
        jd_emb = embed_text(jd_text)
        res_emb = embed_text(resume_text)
        sim = cosine_sim(jd_emb, res_emb)
        return round(sim * 100.0, 2)
    except Exception:
        return 0.0

def final_score(hard_score, semantic_score, hard_weight=0.5, semantic_weight=0.5):
    fs = hard_score * hard_weight + semantic_score * semantic_weight
    return round(fs, 2)

def verdict_from_score(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

def evaluate_resume(filename, resume_text, jd_text, hard_weight=0.5, semantic_weight=0.5, skill_vocab=None):
    """
    Returns a dict with all fields used by the app.
    """
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    hard, matched, missing = hard_match_score(jd_text, resume_text, skill_vocab)
    sem = semantic_score(jd_text, resume_text)
    final = final_score(hard, sem, hard_weight, semantic_weight)
    verdict = verdict_from_score(final)

    return {
        "filename": filename,
        "hard_score": hard,
        "semantic_score": sem,
        "final_score": final,
        "verdict": verdict,
        "matched_skills": ", ".join(matched),
        "missing_skills": ", ".join(missing),
        "resume_text": resume_text
    }
