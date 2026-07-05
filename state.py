from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    State definition for the LangGraph recruitment chatbot.
    Contains exactly the 8 required fields:
    - query: User query or search input
    - intent: Classified intent of the query
    - jd: Raw Job Description text
    - parsed_jd: Structured Job Description fields (e.g. required skills, qualifications)
    - candidate: Current candidate name or details being processed
    - retrieved_docs: Relevant candidate resumes/documents retrieved from DB
    - response: System generated response
    - confirmed: Boolean indicating user confirmation/approval status
    """
    query: str
    intent: str
    jd: str
    parsed_jd: Optional[Dict[str, Any]]
    candidate: Optional[str]
    retrieved_docs: Optional[List[Dict[str, Any]]]
    response: str
    confirmed: bool
