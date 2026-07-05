import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables
dotenv_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

from graph import compiled_graph, get_data_filepath
from tools.db_tool import VectorDBManager
from agents.jd_parser import parse_job_description

console = Console()

def display_welcome_banner():
    banner_text = Text()
    banner_text.append("[AI] Recruitment System Chatbot v1.2\n", style="bold cyan")
    banner_text.append("Agentic AI Bootcamp Hackathon (Project 2 Backend)\n", style="italic green")
    banner_text.append("-" * 55 + "\n", style="dim")
    banner_text.append("Test Commands:\n", style="bold yellow")
    banner_text.append(" • 'How many applicants?'\n")
    banner_text.append(" • 'Get top candidates for AI Engineer'\n")
    banner_text.append(" • 'Rewrite this JD for a startup'\n")
    banner_text.append(" • 'Generate interview questions for Sarah Connor'\n")
    banner_text.append(" • 'Salary expectations for AI Engineer in Hyderabad'\n")
    banner_text.append(" • 'Finalize shortlist'\n")
    banner_text.append("-" * 55 + "\n", style="dim")
    banner_text.append("System Commands:\n", style="bold yellow")
    banner_text.append(" • 'state'             - Dump current graph state fields\n")
    banner_text.append(" • 'help'              - Show this menu\n")
    banner_text.append(" • 'exit' or 'quit'    - Exit the application\n")
    
    console.print(Panel(banner_text, border_style="cyan"))

def display_state(session_state: dict):
    table = Table(title="Current LangGraph State Fields", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    for key, value in session_state.items():
        val_str = str(value)
        if len(val_str) > 100:
            val_str = val_str[:97] + "..."
        table.add_row(key, val_str)
        
    console.print(table)

def main():
    display_welcome_banner()
    
    # Check API Keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    warnings = []
    if not openai_key or "your_openai" in openai_key.lower() or openai_key == "":
        warnings.append("OPENAI_API_KEY is not set. Running in fallback offline mode.")
    if not tavily_key or "your_tavily" in tavily_key.lower() or tavily_key == "":
        warnings.append("TAVILY_API_KEY is not set. Web search features will be disabled.")
        
    if warnings:
        for w in warnings:
            console.print(f"[bold yellow][!] {w}[/bold yellow]")
        console.print("-" * 55, style="dim")
        
    # Initialize the session state
    session_state = {
        "query": "",
        "intent": "",
        "jd": "",
        "parsed_jd": None,
        "candidate": None,
        "retrieved_docs": None,
        "response": "",
        "confirmed": False
    }
    
    # 1. Startup auto-indexing of resumes
    console.print("[cyan]Auto-indexing resumes on startup...[/cyan]")
    try:
        resumes_dir = get_data_filepath("data/resumes")
        db_path = get_data_filepath("vectordb")
        db_manager = VectorDBManager(db_path=db_path)
        if os.path.exists(resumes_dir):
            files = [f for f in os.listdir(resumes_dir) if f.endswith(".txt")]
            for file in files:
                file_path = os.path.join(resumes_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                db_manager.add_resume(file_name=file, content=content)
            console.print(f"[bold green][OK] Auto-indexed {len(files)} resumes successfully![/bold green]")
        else:
            console.print("[bold yellow][!] Resumes directory not found. Directory will be created at runtime.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red][X] Auto-indexing failed: {e}[/bold red]")
        
    # 2. Startup parsing of job description
    console.print("[cyan]Loading and parsing Job Description on startup...[/cyan]")
    try:
        jd_path = get_data_filepath("data/jd.txt")
        if os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
            session_state["jd"] = jd_text
            session_state["parsed_jd"] = parse_job_description(jd_text)
            console.print("[bold green][OK] Loaded and parsed job description successfully![/bold green]")
        else:
            console.print("[bold yellow][!] data/jd.txt not found. It will be initialized on first run.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red][X] Loading Job Description failed: {e}[/bold red]")
        
    console.print("-" * 55, style="dim")
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]Recruiter[/bold green]")
            user_input_clean = user_input.strip()
            
            if not user_input_clean:
                continue
                
            if user_input_clean.lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting Recruitment Chatbot. Goodbye![/bold red]")
                break
                
            if user_input_clean.lower() == "help":
                display_welcome_banner()
                continue
                
            if user_input_clean.lower() == "state":
                display_state(session_state)
                continue
            
            # Prepare state payload containing ONLY the 8 allowed fields
            payload = {
                "query": user_input_clean,
                "intent": session_state["intent"],
                "jd": session_state["jd"],
                "parsed_jd": session_state["parsed_jd"],
                "candidate": session_state["candidate"],
                "retrieved_docs": session_state["retrieved_docs"],
                "response": session_state["response"],
                "confirmed": session_state["confirmed"]
            }
            
            # Run graph
            with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                output_state = compiled_graph.invoke(payload)
                
            # Update session state with the result of graph execution
            for key in session_state.keys():
                if key in output_state and output_state[key] is not None:
                    session_state[key] = output_state[key]
                    
            # Determine Node and Tool based on intent
            intent_to_info = {
                "count_applicants": ("count_applicants_node", "Python OS File System Counter"),
                "screen_candidates": ("screen_candidates_node", "ChromaDB Retriever + OpenAI Reasoner"),
                "rewrite_jd": ("rewrite_jd_node", "OpenAI GPT-4o-mini API"),
                "interview_questions": ("interview_questions_node", "OpenAI GPT-4o-mini API"),
                "salary_lookup": ("salary_lookup_node", "Tavily Search API"),
                "confirmation": ("confirmation_node", "Human Input Evaluator")
            }
            
            node_name, tool_name = intent_to_info.get(session_state["intent"], ("unknown_node", "None"))
            
            # Print Graph execution metadata panel
            meta_text = Text()
            meta_text.append("Intent:       ", style="bold yellow")
            meta_text.append(f"{session_state['intent']}\n", style="cyan")
            meta_text.append("Node:         ", style="bold yellow")
            meta_text.append(f"{node_name}\n", style="magenta")
            meta_text.append("Tool:         ", style="bold yellow")
            meta_text.append(f"{tool_name}", style="green")
            
            console.print(Panel(meta_text, title="[bold blue]Graph Execution Context[/bold blue]", border_style="blue"))
            
            # Print response
            console.print(Panel(session_state["response"], title="[bold green]Assistant Response[/bold green]", border_style="green"))
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted. Exiting chatbot...[/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()
