from typing import List, Dict, TypedDict

class ResearchState(TypedDict):
    topic: str                  
    raw_papers: List[Dict]      
    analyzed_papers: List[Dict] 
    synthesis: str              
    critique: str               
    gaps: List[str]             
    iteration: int            