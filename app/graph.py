import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from memory.state import ResearchState

from agents.researcher import run_researcher
from agents.reader import read_paper
from langchain_groq import ChatGroq


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)



def node_research(state: ResearchState) -> dict:

    print(f"\n🔍 [Graph] Searching for: '{state['topic']}'...")

    from tools.arxiv_tool import search_arxiv
    papers = search_arxiv(state['topic'], max_results=3) 
    print(f"  -> {len(papers)} papers found.")
    return {"raw_papers": papers}

def node_read(state: ResearchState) -> dict:

    print("\n📖 [Graph] Reading and structuring papers...")
    analyzed = []
    for i, paper in enumerate(state["raw_papers"]):
        print(f"  -> Analyzing paper {i+1}/{len(state['raw_papers'])}...")
        analyzed_paper = read_paper(paper)
        analyzed.append(analyzed_paper)
    print("  -> Analysis complete.")
    return {"analyzed_papers": analyzed}

def node_synthesize(state: ResearchState) -> dict:

    print("\n📝 [Graph] Generating synthesis...")
    
 
    context = ""
    for i, p in enumerate(state["analyzed_papers"], 1):
        summary = p.get("structured_summary", {})
        context += f"--- PAPER {i} ---\n"
        context += f"Title: {p['title']}\n"
        context += f"Authors: {', '.join(p['authors'])}\n"
        context += f"Link: {p['link']}\n\n"
        context += f"Objective and Scope: {summary.get('objective_and_scope', 'N/A')}\n"
        context += f"Domain and Motivation: {summary.get('domain_and_motivation', 'N/A')}\n"
        context += f"Main Tasks/Contributions: {summary.get('main_tasks_or_contributions', 'N/A')}\n"
        context += f"Dataset/Methodology: {summary.get('dataset_or_methodology', 'N/A')}\n"
        context += f"Proposed Framework: {summary.get('proposed_framework', 'N/A')}\n"
        context += f"Evaluation Strategy and Metrics: {summary.get('evaluation_strategy_and_metrics', 'N/A')}\n"
        context += f"Key Results: {summary.get('key_results', 'N/A')}\n"
        context += f"Limitations: {summary.get('limitations', 'N/A')}\n"
        context += f"Conclusion: {summary.get('conclusion', 'N/A')}\n"
        context += f"Keywords: {', '.join(summary.get('keywords', []))}\n\n"

    prompt = f"""
You are an expert academic researcher. Write a structured State-of-the-Art review on the topic: "{state['topic']}".

Use the following analyzed papers as your sole source of information. 
Organize the review by key themes or methodologies. Cite the papers by their title and include their links.

Papers data:
{context}

Output format: Markdown. Be concise, academic, and well-structured. Include a references section at the end with all paper links.
"""
    response = llm.invoke(prompt)
    print("  -> Synthesis generated successfully.")
    return {"synthesis": response.content}



def build_research_graph():

    graph = StateGraph(ResearchState)


    graph.add_node("research", node_research)
    graph.add_node("read", node_read)
    graph.add_node("synthesize", node_synthesize)


    graph.add_edge(START, "research")
    graph.add_edge("research", "read")
    graph.add_edge("read", "synthesize")
    graph.add_edge("synthesize", END)

    # Compile the graph
    return graph.compile()



if __name__ == "__main__":
    print(" Launching the LangGraph orchestrator...")
    

    app = build_research_graph()
    

    initial_state = {
        "topic": "quantum machine learning benchmarking",
        "raw_papers": [],
        "analyzed_papers": [],
        "synthesis": ""
    }
    

    for event in app.stream(initial_state):
        
        node_name = list(event.keys())[0]
        print(f"\n✅ Node '{node_name}' completed.")
        
    print("\n" + "="*60)
    print(" FINAL SYNTHESIS (STATE OF THE ART)")
    print("="*60)

    final_state = app.invoke(initial_state)
    print(final_state["synthesis"])