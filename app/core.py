"""
Core module for Smart GTM Agent
Handles all data collection and analysis operations
"""
import os
import json
import time
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from scrapegraph_py import Client
from langchain_nebius import ChatNebius
from http.client import RemoteDisconnected
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import SecretStr

load_dotenv("api.env")

SMARTCRAWLER_KEY = os.getenv("SMARTCRAWLER_API_KEY")
NEBIUS_KEY = os.getenv("NEBIUS_API_KEY")

if not SMARTCRAWLER_KEY or not NEBIUS_KEY:
    raise EnvironmentError(
        "SMARTCRAWLER_API_KEY and NEBIUS_API_KEY must be set in api.env"
    )

# Initialize clients
sg_client = Client(api_key=SMARTCRAWLER_KEY)
llm = ChatNebius(model="NousResearch/Hermes-4-70B", api_key=SecretStr(NEBIUS_KEY))

# =====================================================
# Utility Functions
# =====================================================

def normalize_url(url: str) -> str:
    """
    Ensure URL has proper protocol.
    
    Args:
        url: URL string
        
    Returns:
        Normalized URL with protocol
    """
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def extract_company_name(url: str) -> str:
    """
    Extract company name from URL.
    
    Args:
        url: Company website URL
        
    Returns:
        Cleaned company name
    """
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    if match:
        name = match.group(1)
        # Remove common TLDs and format
        name = re.sub(r'\.(com|io|net|org|ai|co|dev)$', '', name)
        name = name.replace("-", " ").replace(".", " ")
        return name.title()
    return url


# =====================================================
# Data Collection Functions
# =====================================================

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_smartcrawler(url: str) -> str:
    """
    Extract structured company data using SmartCrawler.
    
    Args:
        url: Company website URL (must start with http:// or https://)
        
    Returns:
        Structured company information as markdown
    """
    # Normalize URL
    url = normalize_url(url)
    
    try:
        schema = {
            "type": "object",
            "properties": {
                "Overview": {"type": "string"},
                "Founders": {"type": "array"},
                "Funding": {"type": "array"},
                "Industry": {"type": "string"},
                "Market Size": {"type": "string"},
                "Competitors": {"type": "array"},
            },
        }
        
        print(f"\n[Crawl] Starting for: {url}")
        crawl_response = sg_client.crawl(
            url=url,
            prompt="Extract detailed company information",
            data_schema=schema,
            cache_website=True,
            depth=2,
            max_pages=5,
            same_domain_only=True,
        )
        
        crawl_id = crawl_response.get("id") or crawl_response.get("task_id")
        
        if not crawl_id:
            return f"❌ No crawl ID found. Please check URL ({url}) or API key."
        
        print(f"[Crawl] Crawl started with ID: {crawl_id}, polling for result...")
        
        # Poll for results with timeout
        max_attempts = 60  # 5 minutes total
        for attempt in range(max_attempts):
            time.sleep(5)
            
            try:
                result = sg_client.get_crawl(crawl_id)
                status = result.get("status")
                
                print(f"[Crawl] Attempt {attempt + 1}/{max_attempts} - Status: {status}")
                
                if status == "success" and result.get("result"):
                    print("[Crawl] ✅ Completed successfully.")
                    
                    # Extract content from pages
                    pages = result["result"].get("pages", [])
                    if pages:
                        markdown_content = "\n\n".join(
                            p.get(
                                "markdown",
                                json.dumps(
                                    p.get("content", {}), indent=2, ensure_ascii=False
                                ),
                            )
                            for p in pages
                        )
                    else:
                        markdown_content = json.dumps(
                            result["result"], indent=2, ensure_ascii=False
                        )
                    
                    # Limit content size to avoid token issues
                    if len(markdown_content) > 10000:
                        markdown_content = markdown_content[:10000] + "\n\n[Content truncated due to length]"
                    
                    # LLM summarize
                    prompt = (
                        "You are a precise company research assistant.\n"
                        "Summarize the following data into structured company insights.\n"
                        "Required sections: Overview, Founders, Funding, Industry, Market Size, Competitors.\n"
                        "Keep your response concise and well-structured.\n\n"
                        f"DATA:\n{markdown_content}"
                    )
                    
                    summary = llm.invoke(prompt).content
                    return summary
                    
                elif status == "failed":
                    error_msg = result.get("error", "Unknown error")
                    return f"❌ Crawl failed: {error_msg}"
                    
                # Still processing
                elapsed = 5 * (attempt + 1)
                if elapsed % 30 == 0:  # Log every 30 seconds
                    print(f"[Crawl] Still processing... ({elapsed}s elapsed)")
                
            except Exception as poll_error:
                print(f"[Crawl] Polling error on attempt {attempt + 1}: {poll_error}")
                if attempt == max_attempts - 1:
                    raise
                continue
        
        return "⏱️ Crawl timeout after 5 minutes. The website may be too large or slow to respond."
        
    except Exception as e:
        print(f"[Crawl] Exception: {e}")
        return f"❌ Exception during crawling: {str(e)}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def searchscraper_request(query: str, num_results: int = 5) -> dict:
    """
    Execute SearchScraper query with retry logic.
    
    Args:
        query: Search query
        num_results: Number of results to retrieve
        
    Returns:
        Search results dictionary
    """
    try:
        print(f"\n[SearchScraper] Searching for: {query}")
        resp = sg_client.searchscraper(user_prompt=query, num_results=num_results)
        print(f"[SearchScraper] ✅ Search completed")
        return resp
    except RemoteDisconnected:
        print("[SearchScraper] RemoteDisconnected, retrying...")
        raise
    except Exception as e:
        print(f"[SearchScraper] Exception: {e}")
        raise


def run_searchscraper(company_overview: str) -> str:
    """
    Fetch and analyze competitor data.
    
    Args:
        company_overview: Company overview text (from SmartCrawler)
        
    Returns:
        Structured competitor analysis
    """
    # Extract company name from overview (first 3 words typically contain the name)
    overview_lines = company_overview.split('\n')
    first_line = overview_lines[0] if overview_lines else company_overview
    
    # Try to extract company name more intelligently
    company_name_match = first_line.split()[:5]  # Take first 5 words
    company_name = " ".join(company_name_match)
    
    # Remove common words that shouldn't be in search
    company_name = re.sub(r'\b(the|a|an|is|are|was|were)\b', '', company_name, flags=re.IGNORECASE)
    company_name = company_name.strip()
    
    print(f"[SearchScraper] Extracted company name for search: '{company_name}'")
    
    # Create targeted queries
    queries = [
        f"{company_name} competitors direct rivals",
        f"{company_name} alternative similar companies",
    ]
    
    all_results = []
    
    try:
        # Run multiple queries for comprehensive data
        for query in queries:
            try:
                raw_resp = searchscraper_request(query, num_results=5)
                if raw_resp and raw_resp.get("result"):
                    all_results.append(raw_resp["result"])
            except Exception as query_error:
                print(f"[SearchScraper] Query '{query}' failed: {query_error}")
                continue
        
        if not all_results:
            return "❌ No competitor data found. Try providing more context about the company."
        
        # Combine search results into structured format
        combined_results = []
        for result in all_results:
            if isinstance(result, dict):
                # Extract companies information if available
                if "companies" in result and result["companies"]:
                    companies_text = []
                    for company in result["companies"]:
                        if isinstance(company, dict):
                            name = company.get("name", "Unknown")
                            desc = company.get("description", "No description")
                            companies_text.append(f"**{name}**: {desc}")
                        else:
                            companies_text.append(str(company))
                    combined_results.append("\n".join(companies_text))
                else:
                    # Fallback: convert entire dict to string
                    combined_results.append(str(result))
            else:
                combined_results.append(str(result))
        
        combined_data = "\n\n---\n\n".join(combined_results)
        
        # Limit combined data size
        if len(combined_data) > 8000:
            combined_data = combined_data[:8000] + "\n\n[Data truncated due to length]"
        
        # Generate structured competitor analysis
        prompt = (
            "You are an expert competitive intelligence analyst.\n\n"
            "MISSION: Analyze the search data and identify the most direct, relevant competitors.\n\n"
            "OUTPUT FORMAT:\n\n"
            "# 🏢 Direct Competitors Analysis\n\n"
            "## [Company Name 1]\n"
            "**🎯 Core Focus:** [What they do - concise]\n"
            "**💰 Business Model:** [Revenue model]\n"
            "**📊 Market Position:** [Leader/Challenger/Niche]\n"
            "**🏢 Company Size:** [Stage + rough employee count]\n"
            "**💵 Funding/Revenue:** [Latest financial data if available]\n"
            "**🎯 Target Market:** [Primary customer segments]\n"
            "**⚡ Key Differentiator:** [Main competitive advantage]\n"
            "**🌍 Geographic Reach:** [Primary markets]\n\n"
            "REQUIREMENTS:\n"
            "- Include 4-6 most relevant competitors\n"
            "- Keep descriptions concise (max 15 words per bullet)\n"
            "- Skip indirect/tangential competitors\n"
            "- Use 'Not available' only if truly no data\n"
            "- Focus on direct competitors in the same space\n\n"
            f"COMPANY ANALYZED:\n{company_overview[:500]}\n\n"
            f"SEARCH DATA:\n{combined_data}"
        )
        
        result = llm.invoke(prompt).content
        
        # Ensure result is string
        if isinstance(result, list):
            result = "\n".join(str(item) for item in result)
        else:
            result = str(result) if result else ""
        
        # Validation
        if not result or len(result) < 100:
            return "❌ Unable to extract meaningful competitor data.\n\n**Suggestion:** The search results may not contain competitor information. Try a more specific company name."
        
        # Clean up formatting
        if "# 🏢 Direct Competitors Analysis" not in result:
            result = "# 🏢 Direct Competitors Analysis\n\n" + result
        
        return result
        
    except Exception as e:
        print(f"[SearchScraper] Failed: {e}")
        return f"❌ Error fetching competitor data: {str(e)}\n\nPlease try again or provide more specific company information."


# Export main components
__all__ = [
    'normalize_url',
    'extract_company_name',
    'run_smartcrawler',
    'run_searchscraper',
    'llm',
    'sg_client',
]