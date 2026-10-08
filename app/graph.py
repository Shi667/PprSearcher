import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from memory.state import ResearchState
from tools.arxiv_tool import search_arxiv
from agents.reader import read_paper
from agents.critic import critique_synthesis
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

def node_research(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0) + 1
    print(f"\n [Graph] Research iteration {iteration}...")
    
    if state.get("gaps") and len(state["gaps"]) > 0:
        query = state["gaps"][0]
        print(f"  -> Targeted search for gap: '{query}'")
    else:
        query = state["topic"]
        print(f"  -> Initial search for: '{query}'")
        
    papers = search_arxiv(query, max_results=2)
    print(f"  -> {len(papers)} new papers found.")
    
    current_papers = state.get("raw_papers", [])
    return {
        "raw_papers": current_papers + papers,
        "iteration": iteration
    }

def node_read(state: ResearchState) -> dict:
    print("\n [Graph] Reading and structuring new papers...")
    new_papers = state["raw_papers"][-2:] if len(state["raw_papers"]) >= 2 else state["raw_papers"]
    
    analyzed = []
    for i, paper in enumerate(new_papers):
        print(f"  -> Analyzing new paper {i+1}/{len(new_papers)}...")
        analyzed.append(read_paper(paper))
        
    current_analyzed = state.get("analyzed_papers", [])
    print("  -> Analysis complete.")
    return {"analyzed_papers": current_analyzed + analyzed}

def node_synthesize(state: ResearchState) -> dict:
    print("\n [Graph] Generating/Updating synthesis...")
    
    context = ""
    for i, p in enumerate(state["analyzed_papers"], 1):
        summary = p.get("structured_summary", {})
        context += f"--- PAPER {i} ---\n"
        context += f"Title: {p['title']}\n"
        context += f"Objective: {summary.get('objective_and_scope', 'N/A')}\n"
        context += f"Method: {summary.get('proposed_framework', 'N/A')}\n"
        context += f"Results: {summary.get('key_results', 'N/A')}\n\n"

    prompt = f"""
You are an expert academic researcher. Write a comprehensive, structured State-of-the-Art review on: "{state['topic']}".

Use ALL the following analyzed papers as your source. Integrate the new findings seamlessly with any previous context.
Organize by key themes. Cite papers by title.

Papers data:
{context}

Output format: Markdown. Be concise, academic, and well-structured.
"""
    response = llm.invoke(prompt)
    print("  -> Synthesis updated successfully.")
    return {"synthesis": response.content}

def node_critique(state: ResearchState) -> dict:
    print("\n [Graph] Critic evaluating the synthesis...")
    result = critique_synthesis(state["synthesis"], state["topic"])
    
    verdict = result.get("verdict", "No verdict.")
    gaps = result.get("gaps", [])
    
    print(f"  -> Verdict: {verdict}")
    print(f"  -> Gaps identified: {gaps if gaps else 'None (Synthesis is solid!)'}")
    
    return {
        "critique": verdict,
        "gaps": gaps
    }

def should_continue(state: ResearchState) -> str:
    max_iterations = 2
    gaps = state.get("gaps", [])
    current_iteration = state.get("iteration", 0)
    
    if current_iteration >= max_iterations:
        print("\n [Graph] Max iterations reached. Ending.")
        return "end"
    elif len(gaps) > 0:
        print(f"\n🔄 [Graph] Gaps found. Looping back to research (Iteration {current_iteration + 1})...")
        return "research"
    else:
        print("\n✅ [Graph] Synthesis approved by critic. Ending.")
        return "end"

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("research", node_research)
    graph.add_node("read", node_read)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("critique", node_critique)

    graph.add_edge(START, "research")
    graph.add_edge("research", "read")
    graph.add_edge("read", "synthesize")
    graph.add_edge("synthesize", "critique")

    graph.add_conditional_edges(
        "critique",
        should_continue,
        {
            "research": "research",
            "end": END
        }
    )

    return graph.compile()

graph = build_research_graph()

if __name__ == "__main__":
    print(" Launching the LangGraph orchestrator with Critic Loop...")
    
    app = build_research_graph()
    
    initial_state = {
        "topic": "quantum machine learning benchmarking",
        "raw_papers": [],
        "analyzed_papers": [],
        "synthesis": "",
        "critique": "",
        "gaps": [],
        "iteration": 0
    }
    
    for event in app.stream(initial_state):
        node_name = list(event.keys())[0]
        print(f"\n✅ Node '{node_name}' completed.")
        
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*70)
    print("📄 FINAL APPROVED SYNTHESIS (STATE OF THE ART)")
    print("="*70)
    print(final_state["synthesis"])