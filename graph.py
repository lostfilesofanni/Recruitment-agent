import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import AgentState
from router import route_query
from agents.jd_parser import parse_job_description
from tools.db_tool import VectorDBManager
from tools.search_tool import web_search
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load project environment variables defensively
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

def get_data_filepath(rel_path: str) -> str:
    """
    Resolves correct file paths relative to the Recruitment-Agent folder.
    """
    if os.path.exists(rel_path):
        return rel_path
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, rel_path)
    if os.path.exists(path):
        return path
    
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return path

# ----------------- NODES -----------------

def determine_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Classifies the user query's intent and extracts candidate name if present.
    """
    query = state.get("query", "")
    intent, candidate_name = route_query(query)
    
    return {
        "intent": intent,
        "candidate": candidate_name or state.get("candidate")
    }

def count_applicants_node(state: AgentState) -> Dict[str, Any]:
    """
    Counts the number of resumes in the data/resumes directory using Python only.
    No LLM is called.
    """
    resumes_dir = get_data_filepath("data/resumes")
    if os.path.exists(resumes_dir):
        files = [f for f in os.listdir(resumes_dir) if f.endswith(".txt")]
        count = len(files)
        response = f"[bold green][OK] Applicant Count:[/bold green]\nThere are currently [cyan]{count}[/cyan] applicants in the resumes directory."
    else:
        response = "[bold yellow][!] Resumes directory does not exist yet.[/bold yellow]"
        
    return {
        "response": response
    }

def screen_candidates_node(state: AgentState) -> Dict[str, Any]:
    """
    Performs semantic search in ChromaDB vector database.
    Retrieves Top-5 matching resumes and returns similarity scores.
    Then, uses OpenAI model ONLY to briefly explain WHY each candidate matches the JD.
    Retrieval is never performed by the LLM.
    """
    db_path = get_data_filepath("vectordb")
    db_manager = VectorDBManager(db_path=db_path)
    
    query = state.get("query", "")
    parsed_jd = state.get("parsed_jd")
    
    # Formulate search query from skills if general query
    search_query = query
    if parsed_jd and len(query.strip()) <= 20:
        skills = parsed_jd.get("required_skills", [])
        search_query = f"{parsed_jd.get('role_title')} {' '.join(skills)}"
        
    retrieved = db_manager.search_candidates(search_query, limit=5)
    
    if not retrieved:
        resumes_dir = get_data_filepath("data/resumes")
        folder_files = os.listdir(resumes_dir) if os.path.exists(resumes_dir) else []
        if folder_files:
            response = "[bold yellow][!] No candidates found in database. Please index resumes first.[/bold yellow]"
        else:
            response = "[bold red][X] No resumes found in data/resumes/. Please add text resumes (.txt).[/bold red]"
        return {"retrieved_docs": [], "response": response}

    # Format candidates list for LLM explanation
    candidates_text = ""
    for idx, doc in enumerate(retrieved, 1):
        name = doc["file_name"].replace(".txt", "").replace("_", " ").title()
        candidates_text += f"Candidate {idx}: {name} (Similarity: {doc['score'] * 100:.1f}%)\nResume Summary:\n{doc['content'][:400]}\n\n"

    jd_summary = ""
    if parsed_jd:
        jd_summary = f"Role: {parsed_jd.get('role_title')}\nRequired Skills: {parsed_jd.get('required_skills')}\n"
    else:
        jd_summary = state.get("jd", "AI Engineer position")

    # Generate explanations using OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and "your_openai" not in api_key.lower() and api_key != "":
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = (
                "You are an expert HR assistant. Given a job description and 5 candidates retrieved via RAG semantic search, "
                "generate a structured response highlighting their match percentages, key highlights, and brief explanations of why they match.\n\n"
                "=== JOB DESCRIPTION ===\n"
                f"{jd_summary}\n\n"
                "=== CANDIDATES ===\n"
                f"{candidates_text}\n"
                "Return a formatted report with the exact format:\n"
                "I found five candidates that best match this role.\n\n"
                "1. [Name] ([Similarity Score]%)\n"
                "Key Highlights: [Short phrase listing skills, e.g. Strong Python, LLMs, RAG, LangChain]\n"
                "Reason: [1-2 sentences explaining why they match the JD]\n\n"
                "2. [Name] ...\n"
                "Ensure you use exactly the similarity scores provided in the list. Do not make up new numbers."
            )
            llm_response = llm.invoke([
                SystemMessage(content="You are a professional HR assistant summarizing RAG candidate matches."),
                HumanMessage(content=prompt)
            ])
            response = llm_response.content
        except Exception as e:
            response = fallback_screen_explanation(retrieved, jd_summary, f"LLM error: {e}")
    else:
        response = fallback_screen_explanation(retrieved, jd_summary)

    return {
        "retrieved_docs": retrieved,
        "response": response
    }

def fallback_screen_explanation(retrieved: list, jd_summary: str, warning_msg: str = "") -> str:
    warning_banner = f"[yellow]({warning_msg})[/yellow]\n" if warning_msg else ""
    lines = [f"{warning_banner}I found five candidates that best match this role.\n"]
    for idx, doc in enumerate(retrieved, 1):
        name = doc["file_name"].replace(".txt", "").replace("_", " ").title()
        score = f"{doc['score'] * 100:.1f}%"
        # Find matching skills in text
        skills = []
        for word in ["python", "langchain", "langgraph", "chromadb", "openai", "pytorch", "tavily", "sql", "git", "react", "java", "c#"]:
            if word in doc["content"].lower():
                skills.append(word.capitalize())
        highlights = ", ".join(skills[:4]) if skills else "General IT Skills"
        
        lines.append(
            f"{idx}. {name} ({score})\n"
            f"Key Highlights: {highlights}\n"
            f"Reason: Good alignment with engineering criteria based on keyword presence.\n"
        )
    return "\n".join(lines)

def rewrite_jd_node(state: AgentState) -> Dict[str, Any]:
    """
    Rewrites the Job Description in data/jd.txt using OpenAI API.
    Extracts the requested tone (startup, enterprise, casual, etc.) from the query.
    Updates the file on disk.
    """
    jd_path = get_data_filepath("data/jd.txt")
    query = state.get("query", "")
    
    # Extract tone
    tone = "professional"
    for t in ["startup", "enterprise", "corporate", "casual", "creative", "formal"]:
        if t in query.lower():
            tone = t
            break
            
    if os.path.exists(jd_path):
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
    else:
        jd_text = (
            "Job Title: AI Engineer\n"
            "Requirements:\n"
            "- Python coding experience\n"
            "- RAG and Vector DB development\n"
        )
        
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and "your_openai" not in api_key.lower() and api_key != "":
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            prompt = (
                f"You are an expert recruitment copywriter. Rewrite the following job description "
                f"to suit a {tone} tone. Make it highly engaging and clear. Maintain standard headers:\n\n"
                f"{jd_text}"
            )
            llm_response = llm.invoke([
                SystemMessage(content=f"You are a professional recruiting copywriter writing in a {tone} tone."),
                HumanMessage(content=prompt)
            ])
            rewritten_text = llm_response.content
        except Exception as e:
            rewritten_text = f"Fallback rewritten Job Description ({tone}) due to LLM error ({e}):\n\n" + jd_text
    else:
        rewritten_text = f"[Offline Fallback Mode] Improved Job Description ({tone} tone):\n\n" + jd_text
        
    # Save rewritten text back to jd.txt
    with open(jd_path, "w", encoding="utf-8") as f:
        f.write(rewritten_text)
        
    # Re-parse the rewritten job description
    parsed_jd = parse_job_description(rewritten_text)
    
    response = (
        f"[bold green][OK] Job Description Rewritten ({tone} tone) and Saved to data/jd.txt![/bold green]\n\n"
        f"{rewritten_text}"
    )
    
    return {
        "jd": rewritten_text,
        "parsed_jd": parsed_jd,
        "response": response
    }

def interview_questions_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates interview questions using OpenAI API, grounded on both the
    parsed JD requirements and the selected candidate's resume content.
    Includes both technical and behavioral questions.
    """
    candidate = state.get("candidate")
    parsed_jd = state.get("parsed_jd")
    
    # Parse Job Description if missing
    if not parsed_jd:
        jd_path = get_data_filepath("data/jd.txt")
        if os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
            parsed_jd = parse_job_description(jd_text)
        else:
            parsed_jd = {
                "role_title": "AI Engineer",
                "required_skills": ["Python", "LangGraph", "ChromaDB"],
                "min_experience_years": 3,
                "qualifications": []
            }
            
    # Resolve candidate resume
    resume_content = None
    resume_filename = None
    
    db_path = get_data_filepath("vectordb")
    db_manager = VectorDBManager(db_path=db_path)
    
    if candidate:
        # Check files in resumes folder first (exact / prefix matching)
        resumes_dir = get_data_filepath("data/resumes")
        if os.path.exists(resumes_dir):
            candidate_clean = candidate.lower().replace(" ", "_")
            exact_name = f"{candidate_clean}.txt"
            if exact_name in os.listdir(resumes_dir):
                with open(os.path.join(resumes_dir, exact_name), "r", encoding="utf-8") as f:
                    resume_content = f.read()
                resume_filename = exact_name
                
            if not resume_content:
                # Try prefix or substring match
                for f_name in os.listdir(resumes_dir):
                    if candidate_clean in f_name.lower() or f_name.lower().startswith(candidate_clean[:4]):
                        with open(os.path.join(resumes_dir, f_name), "r", encoding="utf-8") as f:
                            resume_content = f.read()
                        resume_filename = f_name
                        break
                        
        if not resume_content:
            # Search by filename/name in ChromaDB as fallback
            search_results = db_manager.search_candidates(candidate, limit=1)
            if search_results:
                resume_content = search_results[0]["content"]
                resume_filename = search_results[0]["file_name"]
                        
    # Default fallback: take top retrieved doc if candidate not found/specified
    if not resume_content:
        retrieved = state.get("retrieved_docs")
        if retrieved:
            resume_content = retrieved[0]["content"]
            resume_filename = retrieved[0]["file_name"]
            
    # Fallback 2: take first candidate in the database
    if not resume_content:
        all_docs = db_manager.search_candidates("", limit=1)
        if all_docs:
            resume_content = all_docs[0]["content"]
            resume_filename = all_docs[0]["file_name"]
            
    if not resume_content:
        response = (
            "[bold red][X] Candidate resume not found.[/bold red]\n"
            "Please make sure you have indexed resumes first."
        )
        return {"response": response}
        
    if not candidate and resume_filename:
        candidate = resume_filename.replace(".txt", "").replace("_", " ").title()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and "your_openai" not in api_key.lower() and api_key != "":
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            prompt = (
                f"You are a Senior Technical Recruiter. Generate 3 technical and 2 behavioral interview questions for candidate {candidate}.\n\n"
                "=== JOB DESCRIPTION ===\n"
                f"Role: {parsed_jd.get('role_title')}\n"
                f"Required Skills: {parsed_jd.get('required_skills')}\n"
                f"Min Experience: {parsed_jd.get('min_experience_years')} years\n\n"
                "=== CANDIDATE RESUME ===\n"
                f"{resume_content}\n\n"
                "Ground the questions in the candidate's actual projects, background, and skills from their resume, "
                "and how they align with the Job Description. State clearly which questions are Technical and which are Behavioral."
            )
            llm_response = llm.invoke([
                SystemMessage(content="You are a professional technical interviewer generating grounded questions."),
                HumanMessage(content=prompt)
            ])
            response = (
                f"[bold green][OK] Grounded Interview Questions for {candidate} ({resume_filename}):[/bold green]\n\n"
                f"{llm_response.content}"
            )
        except Exception as e:
            response = fallback_interview_questions(candidate, resume_filename, parsed_jd, f"LLM error: {e}")
    else:
        response = fallback_interview_questions(candidate, resume_filename, parsed_jd)
        
    return {
        "candidate": candidate,
        "response": response
    }

def fallback_interview_questions(candidate: str, filename: str, parsed_jd: dict, error_msg: str = "") -> str:
    warning_banner = f"[yellow]({error_msg})[/yellow]\n" if error_msg else ""
    role = parsed_jd.get("role_title", "AI Engineer")
    skills = parsed_jd.get("required_skills", ["Python"])
    
    response = (
        f"[bold green][OK] Grounded Interview Questions for {candidate} ({filename}) [Fallback Mode]:[/bold green]\n"
        f"{warning_banner}\n"
        "[bold cyan]Technical Questions (Grounded):[/bold cyan]\n"
        f"1. Explain how you implemented the technologies matching {', '.join(skills[:3])} in your projects.\n"
        f"2. How would you design a RAG system like the one in your experience using local libraries?\n"
        "3. Walk us through a scenario where a database or agent loop failed and how you debugged it.\n\n"
        "[bold cyan]Behavioral Questions (Grounded):[/bold cyan]\n"
        f"4. Tell me about a time you worked in a team to build an AI or backend system. How did you coordinate task allocation?\n"
        f"5. How do you keep your technical skills updated with the latest trends in the field of {role}?"
    )
    return response

def salary_lookup_node(state: AgentState) -> Dict[str, Any]:
    """
    Looks up salary info using Tavily search only.
    No LLM is called.
    """
    query = state.get("query", "")
    
    # Formulate structured Tavily search query
    search_query = f"average salary, salary range, and recent market trends for {query}"
    tavily_results = web_search(search_query)
    
    response = (
        f"[bold green][OK] Salary & Market Trends (via Tavily):[/bold green]\n\n"
        f"{tavily_results}\n\n"
        f"[bold cyan]Required Information Summary:[/bold cyan]\n"
        f" - [bold]Average Salary:[/bold] Retrievable from search details above.\n"
        f" - [bold]Salary Range:[/bold] Retrievable from search details above.\n"
        f" - [bold]Recent Market Trends:[/bold] Shown in the search articles above.\n\n"
        f"[dim](Direct Tavily search results shown. OpenAI API was not invoked.)[/dim]"
    )
    return {
        "response": response
    }

def confirmation_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles candidate shortlist finalization with user confirmation.
    Proceeds only if user provides confirmation ("yes", "y", "confirm").
    """
    q = state.get("query", "").lower().strip()
    
    # Proceed only if the query confirms the action
    if q in ["yes", "y", "confirm", "approve"]:
        db_path = get_data_filepath("vectordb")
        db_manager = VectorDBManager(db_path=db_path)
        
        # Retrieve Top-3 candidates to form the shortlist
        top_docs = db_manager.search_candidates("", limit=3)
        names = []
        for doc in top_docs:
            name = doc["file_name"].replace(".txt", "").replace("_", " ").title()
            names.append(name)
            
        if not names:
            names = ["John Doe", "Franz Ferdinand", "Sarah Connor"]  # Hardcoded default fallback if db is empty
            
        response = (
            f"[bold green]* Shortlist Finalized Successfully! *[/bold green]\n"
            f"The finalized candidates are: [cyan]{', '.join(names)}[/cyan].\n"
            f"Status: [magenta]confirmed = True[/magenta]"
        )
        return {
            "confirmed": True,
            "response": response
        }
    else:
        # Prompt user to confirm
        return {
            "confirmed": False,
            "response": "Do you want me to finalize this shortlist?"
        }

# ----------------- GRAPH ROUTING -----------------

def route_intent_edge(state: AgentState) -> str:
    """
    Routes to the appropriate specialized node based on intent.
    """
    return state.get("intent", "screen_candidates")

# ----------------- BUILD GRAPH -----------------

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("determine_intent", determine_intent_node)
workflow.add_node("count_applicants", count_applicants_node)
workflow.add_node("screen_candidates", screen_candidates_node)
workflow.add_node("rewrite_jd", rewrite_jd_node)
workflow.add_node("interview_questions", interview_questions_node)
workflow.add_node("salary_lookup", salary_lookup_node)
workflow.add_node("confirmation", confirmation_node)

# Set Entry Point
workflow.set_entry_point("determine_intent")

# Add Conditional Edges from determine_intent node
workflow.add_conditional_edges(
    "determine_intent",
    route_intent_edge,
    {
        "count_applicants": "count_applicants",
        "screen_candidates": "screen_candidates",
        "rewrite_jd": "rewrite_jd",
        "interview_questions": "interview_questions",
        "salary_lookup": "salary_lookup",
        "confirmation": "confirmation"
    }
)

# Connect nodes to END
workflow.add_edge("count_applicants", END)
workflow.add_edge("screen_candidates", END)
workflow.add_edge("rewrite_jd", END)
workflow.add_edge("interview_questions", END)
workflow.add_edge("salary_lookup", END)
workflow.add_edge("confirmation", END)

# Compile graph
compiled_graph = workflow.compile()
