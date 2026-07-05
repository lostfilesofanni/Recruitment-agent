import re
from typing import Optional, Tuple

def route_query(query: str) -> Tuple[str, Optional[str]]:
    """
    Pure Python router using keyword matching and regex.
    No LLM calls are made for routing.
    Returns (intent, candidate_name).
    """
    q = query.lower().strip()
    
    # 1. Confirmation
    if any(k == q for k in ["yes", "y", "confirm", "approve", "finalize", "finalize shortlist", "finalize the shortlist"]):
        return "confirmation", None
        
    # 2. Count Applicants
    if any(k in q for k in ["how many", "count", "number of", "total"]):
        return "count_applicants", None
        
    # 3. Rewrite JD
    if any(k in q for k in ["rewrite", "revise", "improve", "update jd", "rewrite jd"]):
        return "rewrite_jd", None
        
    # 4. Interview Questions
    if any(k in q for k in ["interview", "question"]):
        # Extract candidate name if query is like "Interview questions for Candidate A"
        candidate = None
        match = re.search(r"(?:interview\s+questions|questions)\s+(?:for|of)\s+([a-zA-Z0-9\s\.\-_]+)", q)
        if match:
            candidate = match.group(1).strip().title()
        else:
            # If query is "Interview questions Candidate A"
            match = re.search(r"questions\s+([a-zA-Z0-9\s\.\-_]+)", q)
            if match:
                candidate = match.group(1).strip().title()
        return "interview_questions", candidate

    # 5. Salary Lookup
    if any(k in q for k in ["salary", "pay", "compensation", "wages", "expectations", "earn"]):
        return "salary_lookup", None
        
    # 6. Screen Candidates (Get top candidates)
    # Check if a specific candidate is named for screening, e.g. "screen John Doe"
    candidate = None
    match = re.search(r"(?:screen|evaluate|match)\s+([a-zA-Z0-9\s\.\-_]+)", q)
    if match:
        candidate = match.group(1).strip().title()
        
    return "screen_candidates", candidate
