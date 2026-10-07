import requests
import xml.etree.ElementTree as ET
from typing import List, Dict

ARXIV_API = "http://export.arxiv.org/api/query"

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Query the arXiv API and return a list of dictionaries containing paper details.
    """
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
    print("Searching for 'quantum machine learning'...\n")
    results = search_arxiv("quantum machine learning", max_results=2)
    
    for i, paper in enumerate(results, 1):
        print(f"--- Paper {i} ---")
        print(f"Title  : {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Link   : {paper['link']}")
        print(f"Abstract: {paper['abstract'][:150]}...\n") # We only display the beginning of the abstract