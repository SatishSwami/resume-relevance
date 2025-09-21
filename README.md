# Automated Resume Relevance Check — Final MVP

## Run locally
1. Create and activate virtualenv (Windows PowerShell):

python -m venv venv
.\venv\Scripts\activate.bat
pip install -r requirements.txt

2. (Optional) Set OpenAI key:

$env:OPENAI_API_KEY="sk-XXXX"

3. Run:
streamlit run app.py


## Features
- Multi-JD support (evaluate resumes against multiple JDs)
- Hard-match + Semantic embedding score (final score)
- LLM-powered suggestions (optional, requires OpenAI key)
- Skill-gap analytics and CSV export
- Score simulator (simulate adding missing skills)

## Notes
- Do not commit your OpenAI key.
- If `sentence-transformers` installation is heavy, allow it to finish (it may download a model).
