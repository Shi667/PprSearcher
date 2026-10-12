import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Set

ARXIV_API = "http://export.arxiv.org/api/query"

def search_arxiv(query: str, max_results: int = 10, exclude_ids: Set[str] = None) -> List[Dict]:
 
    if exclude_ids is None:
        exclude_ids = set()

    # Request more than needed so we can filter out seen ones and still have enough
    fetch_count = max_results + len(exclude_ids) + 5

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": fetch_count,
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

        # Extract the arXiv ID from the link (e.g. "2011.00633v1")
        arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else link

        # Skip if we've already seen this paper
        if arxiv_id in exclude_ids:
            continue

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "link": link,
            "arxiv_id": arxiv_id,  # NEW: store the ID for memory tracking
        })

        if len(papers) >= max_results:
            break

    return papers


if __name__ == "__main__":
    print("Searching for 'quantum machine learning'...\n")
    results = search_arxiv("quantum machine learning", max_results=2)

    for i, paper in enumerate(results, 1):
        print(f"--- Paper {i} ---")
        print(f"Title  : {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Link   : {paper['link']}")
        print(f"ID     : {paper['arxiv_id']}")
        print(f"Abstract: {paper['abstract'][:150]}...\n")