# utils/llm_utils.py
import os
import json
import re

# Try to support both new (openai>=1.0) and old openai (<1.0) APIs.
try:
    # new-style client (openai>=1.0)
    from openai import OpenAI as _OpenAIClient
    _HAS_NEW_OPENAI = True
except Exception:
    _HAS_NEW_OPENAI = False

try:
    import openai as _old_openai  # for old versions (0.x)
    _HAS_OLD_OPENAI = True
except Exception:
    _HAS_OLD_OPENAI = False

def _ensure_api_key(key):
    """
    Ensure API key is available to whichever client we use.
    Returns the key (or None).
    """
    k = key or os.getenv("OPENAI_API_KEY")
    if k:
        # new client reads from env var by default, set it to be safe
        os.environ["OPENAI_API_KEY"] = k
        if _HAS_OLD_OPENAI:
            try:
                _old_openai.api_key = k
            except Exception:
                pass
    return k

def _chat_completion(messages, model="gpt-3.5-turbo", max_tokens=250, temperature=0.2):
    """
    Unified wrapper that calls ChatCompletion using the installed openai version.
    Returns the assistant text (string) or raises an Exception.
    """
    if _HAS_NEW_OPENAI:
        # New-style usage: client.chat.completions.create(...)
        client = _OpenAIClient()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # The new client returns objects where content is at resp.choices[0].message.content
        try:
            return resp.choices[0].message.content
        except Exception:
            # fallback to raw dict-like access
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"Unexpected response structure from new OpenAI client: {e}")
    elif _HAS_OLD_OPENAI:
        # Old-style usage: openai.ChatCompletion.create(...)
        resp = _old_openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp["choices"][0]["message"]["content"]
    else:
        raise ImportError("OpenAI client not installed. Install with `pip install openai`")
def generate_feedback(jd_text, resume_text, missing_skills=None, openai_api_key=None, model="gpt-3.5-turbo"):
    """
    Returns dict with keys: 'suggestions' (list) and 'verdict' (string).
    If OpenAI is unavailable or returns a quota/auth error, this returns safe fallback suggestions.
    Relies on helper functions _ensure_api_key(...) and _chat_completion(...) defined earlier in the file.
    """
    missing = missing_skills or []
    key = _ensure_api_key(openai_api_key)

    # If no API key or no client available, return fallback immediately
    if not key or (not _HAS_NEW_OPENAI and not _HAS_OLD_OPENAI):
        suggestions = []
        for s in (missing[:3] if missing else []):
            suggestions.append(f"Add a short project bullet demonstrating {s}.")
        if not suggestions:
            suggestions = [
                "Make the skills section concise and relevant to the JD.",
                "Add 1-2 short projects showing tools from the JD.",
                "Mention quantifiable impact (e.g., improved X by Y%)."
            ]
        return {"suggestions": suggestions, "verdict": "LLM unavailable (fallback)"}

    prompt = f"""
You are a concise resume coach. Given the Job Description and Candidate Resume, provide:
1) Up to 3 short, actionable suggestions (each <= 20 words) to make the resume more relevant.
2) A one-line verdict: "Strong fit", "Partial fit", or "Weak fit".

Return JSON like:
{{"suggestions":["...","..."], "verdict":"..."}}

Job Description:
{jd_text}

Resume:
{resume_text}

Missing skills: {', '.join(missing) if missing else 'None'}
"""

    messages = [
        {"role": "system", "content": "You are a concise resume coach."},
        {"role": "user", "content": prompt}
    ]

    try:
        text = _chat_completion(messages=messages, model=model, max_tokens=250, temperature=0.2)
        # Attempt to parse JSON from model output
        try:
            import json
            return json.loads(text)
        except Exception:
            # Try to extract a JSON substring in case model included explanation text
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            # Fallback: split lines into suggestions and use last line as verdict
            lines = [ln.strip("-• ") for ln in text.splitlines() if ln.strip()]
            return {"suggestions": lines[:3], "verdict": lines[-1] if lines else "No verdict"}
    except Exception as exc:
        # Normalize error message for pattern matching
        err_str = str(exc).lower()

        # Quota / rate limit -> provide helpful fallback suggestions
        if "quota" in err_str or "insufficient" in err_str or "rate limit" in err_str or "429" in err_str:
            suggestions = []
            for s in (missing[:3] if missing else []):
                suggestions.append(f"Add a short project bullet demonstrating {s}.")
            if not suggestions:
                suggestions = [
                    "Make the skills section concise and relevant to the JD.",
                    "Add 1-2 short projects showing tools from the JD.",
                    "Mention quantifiable impact (e.g., improved X by Y%)."
                ]
            return {"suggestions": suggestions, "verdict": "LLM unavailable (quota/rate limit) - fallback suggestions"}

        # Authentication / invalid key -> explain to user
        if "invalid" in err_str or "incorrect api key" in err_str or "401" in err_str:
            return {"suggestions": ["OpenAI API key invalid or revoked. Disable LLM or set a valid key."],
                    "verdict": "LLM auth error (invalid key)"}

        # Other OpenAI-related errors -> return generic fallback and include small error note
        return {"suggestions": [f"Could not fetch LLM suggestions: {str(exc)}",
                                "Add a small project line showing relevant skill(s)."],
                "verdict": "LLM error (fallback suggestions)"}


    prompt = f"""
You are a concise resume coach. Given the Job Description and a Candidate Resume, provide:
1) Up to 3 short, actionable suggestions (each <= 20 words) to make the resume more relevant.
2) A one-line verdict ("Strong fit", "Partial fit", or "Weak fit").

Return JSON like:
{{"suggestions":["...","..."], "verdict":"..."}}

Job Description:
{jd_text}

Resume:
{resume_text}

Missing skills: {', '.join(missing) if missing else 'None'}
"""

    messages = [
        {"role": "system", "content": "You are a concise resume coach."},
        {"role": "user", "content": prompt}
    ]

    try:
        text = _chat_completion(messages=messages, model=model, max_tokens=250, temperature=0.2)
        # Try to parse JSON from model output
        try:
            return json.loads(text)
        except Exception:
            # Try to extract JSON-like substring
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            # fallback: split lines into suggestions
            lines = [ln.strip("-• ") for ln in text.splitlines() if ln.strip()]
            return {"suggestions": lines[:3], "verdict": lines[-1] if lines else ""}
    except Exception as e:
        # Return a helpful error in the response object so UI can show it
        return {"suggestions": [f"LLM error: {e}"], "verdict": ""}
