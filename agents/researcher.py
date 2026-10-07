import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pyexpat.errors import messages 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from tools.arxiv_tool import search_arxiv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


load_dotenv()  #

@tool


def arxiv_search_tool(query: str, max_results: int = 5) -> str:
    """Recherche des articles scientifiques sur arXiv à partir d'une requête texte."""
    papers = search_arxiv(query, max_results)

    if not papers:
        return "Aucun papier trouvé pour la requête donnée."
    
    formatted_results = []
    for i, p in enumerate(papers, 1):
        formatted_results.append(
            f"[{i}] Titre: {p['title']}\n"
            f"Auteurs: {', '.join(p['authors'])}\n"
            f"Résumé: {p['abstract'][:500]}...\n" 
            f"Lien: {p['link']}"
        )
    return "\n\n---\n\n".join(formatted_results)



llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
agent_llm = llm.bind_tools([arxiv_search_tool])


def run_researcher(topic: str, max_iters: int = 5) -> str:

    messages = [
        SystemMessage(content=(
        "Tu es un assistant de recherche scientifique expert. "
        "Ton objectif est de trouver des articles pertinents sur arXiv concernant le sujet de l'utilisateur. "
        "1. Utilise l'outil de recherche arXiv. "
        "2. Évalue la pertinence des résultats. "
        "3. Si les résultats sont mauvais ou vides, reformule ta requête (essaie en anglais, ou avec des mots-clés plus précis) et recherche à nouveau. "
        "4. Arrête-toi quand tu as trouvé au moins 3 bons articles OU après 3 tentatives maximum. "
        "5. À la fin, résume brièvement les meilleurs articles trouvés avec leurs liens."
        )),
        HumanMessage(content=f"Trouve des articles de recherche récents et pertinents sur le sujet suivant : {topic}")
    ]

    for iteration in range(max_iters):
        print(f"\n[Agent] Itération {iteration + 1}/{max_iters}...")
        
        
        response = agent_llm.invoke(messages)
        messages.append(response) 
        
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"  -> Appel de l'outil avec la requête : '{tool_call['args'].get('query')}'")
                
                
                tool_result = arxiv_search_tool.invoke(tool_call["args"])
                
                
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))
        else:
            
            print("  -> L'agent a terminé sa recherche.")
            break

  
    return messages[-1].content


if __name__ == "__main__":
    print("Lancement de l'agent de recherche...")
    resultat_final = run_researcher("quantum machine learning")
    print("\n" + "="*50)
    print("RÉSULTAT FINAL DE L'AGENT :")
    print("="*50)
    print(resultat_final)


