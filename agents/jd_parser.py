import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class ParsedJobDescription(BaseModel):
    role_title: str = Field(description="The formal title of the job role.")
    required_skills: List[str] = Field(description="A list of required technical or soft skills.")
    preferred_skills: List[str] = Field(description="A list of preferred or optional skills.")
    min_experience_years: int = Field(description="Minimum years of experience required. If not specified, default to 0.")
    qualifications: List[str] = Field(description="Required educational qualifications or certifications.")
    summary: str = Field(description="A brief 2-3 sentence summary of the job role.")

def parse_job_description(jd_text: str) -> Dict[str, Any]:
    """
    Parses raw Job Description text into a structured Pydantic object.
    Falls back to a basic heuristic dict if API key is not configured or in case of errors.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "your_openai" in api_key.lower() or api_key == "":
        # Simple heuristic fallback if API key is not present
        return parse_jd_fallback(jd_text)
    
    try:
        # Initialize LangChain OpenAI chat client
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ParsedJobDescription)
        
        prompt = (
            "Analyze the following Job Description text and extract structured information:\n\n"
            f"{jd_text}"
        )
        
        parsed = structured_llm.invoke(prompt)
        # Convert pydantic model to dict
        return parsed.model_dump()
    except Exception as e:
        print(f"[Warning] OpenAI parsing failed: {e}. Using fallback parser.")
        return parse_jd_fallback(jd_text)

def parse_jd_fallback(jd_text: str) -> Dict[str, Any]:
    """
    A lightweight rule-based fallback parser if OpenAI key is not valid/provided.
    """
    lines = [line.strip() for line in jd_text.split("\n") if line.strip()]
    role_title = "Software Engineer"
    if lines:
        role_title = lines[0][:50]  # Take first line as title limit to 50 chars
        
    # Heuristics
    skills = []
    for word in ["python", "java", "react", "javascript", "sql", "c++", "aws", "docker", "kubernetes", "langchain", "langgraph"]:
        if word in jd_text.lower():
            skills.append(word.capitalize())
            
    if not skills:
        skills = ["Software Development"]
        
    return {
        "role_title": role_title,
        "required_skills": skills,
        "preferred_skills": [],
        "min_experience_years": 2,
        "qualifications": ["Bachelor's degree in Computer Science or equivalent"],
        "summary": jd_text[:150] + "..." if len(jd_text) > 150 else jd_text
    }
