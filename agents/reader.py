import sys
import os
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Initialiser le LLM (utilise le même modèle qui a fonctionné pour toi)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

def read_paper(paper: dict) -> dict:
    """
    Prend un dictionnaire de papier et renvoie un résumé structuré selon un format académique rigoureux.
    """
    prompt = f"""
You are an expert scientific reviewer. Analyze the following research paper abstract and extract the key information into a highly structured academic summary.

Return ONLY a valid JSON object with the following exact keys. If a specific detail is not mentioned in the abstract, write "Not specified in abstract".

- "objective_and_scope": The main goal and scope of the study (1-2 sentences).
- "domain_and_motivation": The background, context, or research gap addressed.
- "main_tasks_or_contributions": The specific tasks, objectives, or novel contributions.
- "dataset_or_methodology": Details about the data used or the general methodology.
- "proposed_framework": The specific models, algorithms, or approaches proposed.
- "evaluation_strategy_and_metrics": How the method was evaluated and which metrics were used.
- "key_results": The main quantitative or qualitative findings.
- "limitations": Any mentioned constraints or limitations of the study.
- "conclusion": The final takeaway or broader impact.
- "keywords": A list of 3 to 5 relevant keywords.

Paper Title: {paper['title']}
Paper Abstract: {paper['abstract']}

Output strictly in JSON format, no markdown, no extra text.
"""
    
    try:
        response = llm.invoke(prompt)
        # Nettoyer la réponse au cas où le LLM ajouterait des backticks ```json ... ```
        clean_response = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        structured_summary = json.loads(clean_response)
        
        # On retourne le papier original avec le résumé structuré ajouté
        return {**paper, "structured_summary": structured_summary}
    
    except json.JSONDecodeError:
        # Fallback si le LLM ne renvoie pas un JSON valide
        return {**paper, "structured_summary": {"error": "Failed to parse LLM response as JSON", "raw": response.content}}


if __name__ == "__main__":
    test_paper = {
        "title": "MerLin: A Discovery Engine for Photonic and Hybrid Quantum Machine Learning",
        "authors": ["Cassandre Notton", "Benjamin Stott", "Philippe Schoeb", "et al."],
        "abstract": "Identifying where quantum models may offer practical benefits in near term quantum machine learning (QML) requires moving beyond isolated algorithmic proposals toward systematic and empirical exploration across models, datasets, and hardware constraints. We introduce MerLin, an open-source framework designed as a discovery engine for photonic and hybrid quantum machine learning. MerLin integrates optimized strong simulation of linear optical circuits into standard PyTorch and scikit-learn workflows, enabling systematic benchmarking. We evaluate MerLin on multiple datasets and hardware constraints, demonstrating its ability to identify practical advantages of quantum models over classical baselines in specific regimes.",
        "link": "http://arxiv.org/abs/2602.11092v2"
    }

    print("Analyzing paper with academic structure...\n")
    result = read_paper(test_paper)
    
    print(f"TITLE: {result['title']}\n")
    print("STRUCTURED SUMMARY:")
    print(json.dumps(result["structured_summary"], indent=2))