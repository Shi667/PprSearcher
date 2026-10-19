import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.pdf_reader import get_full_paper_text

load_dotenv()

# Utilisation du modèle Qwen avec une limite stricte pour éviter les erreurs OTPM
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=600)

def read_paper(paper: dict) -> dict:
    """
    Fetches the FULL TEXT of the paper. If it fails, gracefully falls back to the abstract.
    """
    print(f"    -> Downloading and parsing full PDF for: '{paper['title'][:50]}...'")
    full_text = get_full_paper_text(paper['link'])
    
    # SMART FALLBACK: If PDF extraction failed or returned very little text, use the abstract!
    if full_text.startswith("Error") or len(full_text) < 300:
        print(f"    -> ⚠️ PDF extraction failed or too short. Falling back to abstract.")
        text_to_analyze = f"ABSTRACT:\n{paper['abstract']}\n\n(Note: Full PDF extraction failed or was insufficient. Base your analysis primarily on this abstract.)"
    else:
        text_to_analyze = full_text

    prompt = f"""
You are an expert scientific reviewer. Analyze the following research paper text and extract key information into a highly structured academic summary.

Return ONLY a valid JSON object with the following exact keys. If a specific detail is genuinely not in the provided text, write "Not specified in text".

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
Paper Text to Analyze: 
{text_to_analyze}

Output strictly in JSON format, no markdown, no extra text. Be extremely concise.
"""
    
    try:
        response = llm.invoke(prompt)
        clean_response = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_summary = json.loads(clean_response)
        
        return {**paper, "structured_summary": structured_summary}
    
    except json.JSONDecodeError:
        return {**paper, "structured_summary": {"error": "Failed to parse LLM response as JSON", "raw": response.content}}