import json
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime

MEMORY_FILE = Path(__file__).parent / "seen_papers.json"

def load_memory() -> List[Dict]:
    """Load the list of seen papers (each with id, title, date_added)."""
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Backward compatibility: handle old format (just a set of IDs)
            if isinstance(data, dict) and "papers" in data:
                return data["papers"]
            elif isinstance(data, dict) and "seen_ids" in data:
                # Migrate old format to new format
                return [{"id": pid, "title": "Unknown title", "date_added": None} 
                        for pid in data["seen_ids"]]
            return []
    except Exception:
        return []

def save_memory(papers: List[Dict]) -> None:
   
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"papers": papers}, f, indent=2, ensure_ascii=False)

def get_seen_ids() -> Set[str]:
    
    return {p["id"] for p in load_memory() if "id" in p}

def add_papers(new_papers: List[Dict]) -> List[Dict]:

    current = load_memory()
    current_ids = {p["id"] for p in current}
    
    for paper in new_papers:
        if paper["id"] not in current_ids:
            current.append({
                "id": paper["id"],
                "title": paper.get("title", "Unknown title"),
                "date_added": datetime.now().isoformat()
            })
            current_ids.add(paper["id"])
    
    save_memory(current)
    return current

def clear_memory() -> None:
   
    save_memory([])

def count_seen() -> int:
  
    return len(load_memory())

def get_all_papers() -> List[Dict]:
  
    return load_memory()