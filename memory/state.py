from typing import List, Dict, TypedDict

class ResearchState(TypedDict):
    topic: str
    num_papers: int             
    raw_papers: List[Dict]
    analyzed_papers: List[Dict]
    synthesis: str             
    critique: str
    gaps: List[str]
    iteration: int