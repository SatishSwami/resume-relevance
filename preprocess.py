# utils/preprocess.py
import re

# Base skill vocabulary (extend as needed)
DEFAULT_SKILLS = [
    "python","java","c++","sql","machine learning","deep learning","tensorflow",
    "pytorch","nlp","computer vision","aws","docker","kubernetes","git","linux",
    "matlab","opencv","react","node.js","html","css","javascript","excel","sql",
    "pandas","numpy","scikit-learn","tableau","power bi","spark"
]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_skills_from_text(text: str, skill_vocab=None):
    """
    Very simple skill detector: checks if any term from vocab appears in text.
    """
    text_l = clean_text(text).lower()
    skills = set()
    skill_vocab = skill_vocab or DEFAULT_SKILLS
    for s in skill_vocab:
        if s.lower() in text_l:
            skills.add(s.lower())
    return sorted(list(skills))

# basic sections extractor (heuristic)
def extract_sections(text: str):
    text = clean_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sections = {}
    current = "summary"
    sections[current] = []
    headers = ["skills","experience","education","projects","certifications","achievements","internship"]
    for line in lines:
        low = line.lower()
        if any(h in low for h in headers) and len(line.split()) < 6:
            # start new section
            # use the header word as key
            for h in headers:
                if h in low:
                    current = h
                    break
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k,v in sections.items()}
