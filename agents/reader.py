import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.pdf_reader import get_full_paper_text

load_dotenv()

# Using the Qwen model we selected earlier (or switch back to llama-3.1-8b-instant if preferred)
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2)

def read_paper(paper: dict) -> dict:
    """
    Fetches the FULL TEXT of the paper and extracts a highly structured academic summary.
    """
    print(f"    -> Downloading and parsing full PDF for: '{paper['title'][:50]}...'")
    full_text = get_full_paper_text(paper['link'])
    
    prompt = f"""
You are an expert scientific reviewer. Analyze the following research paper and extract key information into a highly structured academic summary.

Return ONLY a valid JSON object with the following exact keys. If a specific detail is genuinely not in the text, write "Not specified in text".

- "objective_and_scope": The main goal and scope of the study (1-2 sentences).
- "domain_and_motivation": The background, context, or research gap addressed.
- "main_tasks_or_contributions": The specific tasks, objectives, or novel contributions.
- "dataset_or_methodology": Details about the data used, experimental setup, or general methodology.
- "proposed_framework": The specific models, algorithms, or approaches proposed.
- "evaluation_strategy_and_metrics": How the method was evaluated and which metrics were used.
- "key_results": The main quantitative or qualitative findings.
- "limitations": Any mentioned constraints or limitations of the study.
- "conclusion": The final takeaway or broader impact.
- "keywords": A list of 3 to 5 relevant keywords.

Paper Title: {paper['title']}
Paper Full Text (excerpt): 
{full_text}

Output strictly in JSON format, no markdown, no extra text.
"""
    
    try:
        response = llm.invoke(prompt)
        clean_response = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_summary = json.loads(clean_response)
        
        return {**paper, "structured_summary": structured_summary}
    
    except json.JSONDecodeError:
        return {**paper, "structured_summary": {"error": "Failed to parse LLM response as JSON", "raw": response.content}}