import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

ARXIV_API = "http://export.arxiv.org/api/query"



def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
   
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    response = requests.get(ARXIV_API, params=params, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data from arXiv API: {response.status_code}")
    
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link = entry.find("atom:id", ns).text
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        
        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "link": link,
        })
    
    return papers


if __name__ == "__main__":
    print("Recherche en cours sur 'quantum machine learning'...\n")
    resultats = search_arxiv("quantum machine learning")
    
    for i, papier in enumerate(resultats, 1):
        print(f"--- Papier {i} ---")
        print(f"Titre  : {papier['title']}")
        print(f"Auteurs: {', '.join(papier['authors'])}")
        print(f"Lien   : {papier['link']}")
        print(f"Résumé : {papier['abstract'][:150]}...\n") # On affiche juste le début du résumé