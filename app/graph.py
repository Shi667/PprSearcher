import sys
import os
import json

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
    num_papers = state.get("num_papers", 2)
    print(f"\n🔍 [Graph] Research iteration {iteration}...")
    
    if state.get("gaps") and len(state["gaps"]) > 0:
        query = state["gaps"][0]
        print(f"  -> Targeted search for gap: '{query}'")
    else:
        query = state["topic"]
        print(f"  -> Initial search for: '{query}'")
        
    
    papers = search_arxiv(query, max_results=num_papers)
    print(f"  -> {len(papers)} new papers found.")
    
    current_papers = state.get("raw_papers", [])
    return {
        "raw_papers": current_papers + papers,
        "iteration": iteration
    }

def node_read(state: ResearchState) -> dict:
    print("\n📖 [Graph] Reading and structuring new papers...")
  
    new_papers = state["raw_papers"][-state.get("num_papers", 2):] if len(state["raw_papers"]) >= state.get("num_papers", 2) else state["raw_papers"]
    
    analyzed = []
    for i, paper in enumerate(new_papers):
        print(f"  -> Analyzing new paper {i+1}/{len(new_papers)}...")
        analyzed.append(read_paper(paper))
        
    current_analyzed = state.get("analyzed_papers", [])
    print("  -> Analysis complete.")
    return {"analyzed_papers": current_analyzed + analyzed}

def node_synthesize(state: ResearchState) -> dict:
  
    print("\n📝 [Graph] Generating intermediate synthesis for critique...")
    
    context = ""
    for i, p in enumerate(state["analyzed_papers"], 1):
        summary = p.get("structured_summary", {})
        context += f"--- PAPER {i} ---\n"
        context += f"Title: {p['title']}\n"
        context += f"Objective: {summary.get('objective_and_scope', 'N/A')}\n"
        context += f"Method: {summary.get('proposed_framework', 'N/A')}\n"
        context += f"Results: {summary.get('key_results', 'N/A')}\n\n"

    prompt = f"""
You are an expert academic researcher. Write a structured State-of-the-Art review on: "{state['topic']}".
Use the following analyzed papers as your source.

Papers data:
{context}

Output format: Markdown. Be concise and academic.
"""
    response = llm.invoke(prompt)
    print("  -> Intermediate synthesis generated.")
    return {"synthesis": response.content}

def node_critique(state: ResearchState) -> dict:
    print("\n⚖️ [Graph] Critic evaluating the synthesis...")
    result = critique_synthesis(state["synthesis"], state["topic"])
    
    verdict = result.get("verdict", "No verdict.")
    gaps = result.get("gaps", [])
    
    print(f"  -> Verdict: {verdict}")
    print(f"  -> Gaps identified: {gaps if gaps else 'None (Synthesis is solid!)'}")
    
    return {
        "critique": verdict,
        "gaps": gaps
    }

def node_curate(state: ResearchState) -> dict:
    """Node: Selects the BEST `num_papers` from all fetched papers and generates the FINAL synthesis."""
    print("\n🏆 [Graph] Curating and selecting the best papers...")
    num_papers = state.get("num_papers", 2)
    all_papers = state.get("analyzed_papers", [])
    
    if len(all_papers) <= num_papers:
        print(f"  -> Already have {len(all_papers)} papers (<= {num_papers}). Keeping all.")
        final_papers = all_papers
    else:
        print(f"  -> Evaluating {len(all_papers)} papers to select the top {num_papers}...")
        
        papers_text = "\n\n".join([
            f"ID: {i}\nTitle: {p['title']}\nObjective: {p.get('structured_summary', {}).get('objective_and_scope', '')}"
            for i, p in enumerate(all_papers)
        ])
        
        prompt = f"""
You are an expert academic curator. You have a list of {len(all_papers)} papers related to the topic: "{state['topic']}".
Your task is to select the EXACT top {num_papers} most relevant, high-quality papers for a state-of-the-art review.

Papers:
{papers_text}

Return ONLY a valid JSON array of the IDs (integers) of the top {num_papers} papers, ordered by relevance (most relevant first).
Example: [2, 0]
"""
        try:
            response = llm.invoke(prompt)
            clean_response = response.content.strip().removeprefix("```json").removesuffix("```").strip()
            top_ids = json.loads(clean_response)
            
            # Filter and sort based on LLM's ranking
            final_papers = [all_papers[i] for i in top_ids if i < len(all_papers)][:num_papers]
            titles = [f"'{p['title'][:50]}...'" for p in final_papers]
            print(f"  -> Selected top {num_papers} papers: {', '.join(titles)}")
        except Exception as e:
            print(f"  -> Curation parsing failed, falling back to first {num_papers} papers. Error: {e}")
            final_papers = all_papers[:num_papers]
    
    # Generate the FINAL synthesis based ONLY on the curated papers
    print("  -> Generating final synthesis from curated papers...")
    context = ""
    for i, p in enumerate(final_papers, 1):
        summary = p.get("structured_summary", {})
        context += f"# {p['title']}\n\n"
        context += f"**Authors:** {', '.join(p['authors'])}  \n"
        context += f"**Link:** {p['link']}\n\n"
        context += f"## Objective and Scope\n{summary.get('objective_and_scope', 'N/A')}\n\n"
        context += f"## Domain and Motivation\n{summary.get('domain_and_motivation', 'N/A')}\n\n"
        context += f"## Main Tasks and Contributions\n{summary.get('main_tasks_or_contributions', 'N/A')}\n\n"
        context += f"## Dataset and Methodology\n{summary.get('dataset_or_methodology', 'N/A')}\n\n"
        context += f"## Proposed Framework\n{summary.get('proposed_framework', 'N/A')}\n\n"
        context += f"## Evaluation Strategy and Metrics\n{summary.get('evaluation_strategy_and_metrics', 'N/A')}\n\n"
        context += f"## Key Results\n{summary.get('key_results', 'N/A')}\n\n"
        context += f"## Limitations\n{summary.get('limitations', 'N/A')}\n\n"
        context += f"## Conclusion\n{summary.get('conclusion', 'N/A')}\n\n"
        context += f"**Keywords:** {', '.join(summary.get('keywords', []))}\n\n---\n\n"

    final_prompt = f"""
You are an expert academic researcher. Format the following curated papers into a clean, professional State-of-the-Art review on: "{state['topic']}".

Keep the exact structure provided (Title as H1, sections as H2). Do not add extra commentary.

Papers data:
{context}
"""
    response = llm.invoke(final_prompt)
    print("  -> Final curation and synthesis complete.")
    
    return {
        "analyzed_papers": final_papers,
        "synthesis": response.content
    }

def should_continue(state: ResearchState) -> str:
    max_iterations = 2
    gaps = state.get("gaps", [])
    current_iteration = state.get("iteration", 0)
    
    if current_iteration >= max_iterations:
        print("\n🛑 [Graph] Max iterations reached. Moving to final curation.")
        return "curate"
    elif len(gaps) > 0:
        print(f"\n🔄 [Graph] Gaps found. Looping back to research (Iteration {current_iteration + 1})...")
        return "research"
    else:
        print("\n✅ [Graph] Synthesis approved by critic. Moving to final curation.")
        return "curate"

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("research", node_research)
    graph.add_node("read", node_read)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("critique", node_critique)
    graph.add_node("curate", node_curate) 

    graph.add_edge(START, "research")
    graph.add_edge("research", "read")
    graph.add_edge("read", "synthesize")
    graph.add_edge("synthesize", "critique")

    graph.add_conditional_edges(
        "critique",
        should_continue,
        {
            "research": "research",
            "curate": "curate" 
        }
    )
    
    graph.add_edge("curate", END) 

    return graph.compile()

\
graph = build_research_graph()

if __name__ == "__main__":
    print("🚀 Launching the LangGraph orchestrator with Critic Loop and Curator...")
    
    app = build_research_graph()
    
    initial_state = {
        "topic": "argument mining",
        "num_papers": 2,
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
    print(f"📄 FINAL CURATED SYNTHESIS (Exactly {initial_state['num_papers']} papers)")
    print("="*70)
    print(final_state["synthesis"])