import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load project env configuration defensively
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

def clean_non_ascii(text: str) -> str:
    """
    Cleans non-ASCII characters to prevent console encoding crashes on Windows.
    """
    # Replace Rupee symbol with Rs.
    text = text.replace("\u20b9", "Rs. ")
    # Replace curly quotes and long dashes
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Strip any other remaining non-ASCII characters
    return text.encode("ascii", "ignore").decode("ascii")

def web_search(query: str) -> str:
    """
    Performs a web search using Tavily search API.
    Returns formatted results.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or "your_tavily" in api_key.lower() or api_key == "":
        return "Tavily API key is not configured in .env. General search is disabled."
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No web search results found."
        
        formatted_results = []
        for r in results:
            title = r.get("title", "No Title")
            url = r.get("url", "No URL")
            content = r.get("content", "")
            raw_text = f"Title: {title}\nURL: {url}\nContent: {content}\n"
            formatted_results.append(clean_non_ascii(raw_text))
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"
