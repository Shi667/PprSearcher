import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from memory.state import ResearchState
from tools.arxiv_tool import search_arxiv
from agents.reader import read_paper
from agents.critic import critique_synthesis
from memory.paper_memory import get_seen_ids, add_papers
from langchain_groq import ChatGroq

# PERFECT LIMITATION 1: Hard cap output tokens to 800 (safely under the 1000 limit)
llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2, max_tokens=800)


def node_research(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0) + 1
    num_papers = state.get("num_papers", 2)
    seen_ids = get_seen_ids()

    print(f"\n🔍 [Graph] Research iteration {iteration}...")
    print(f"  -> Memory: {len(seen_ids)} papers already seen, filtering them out.")

    if state.get("gaps") and len(state["gaps"]) > 0:
        query = state["gaps"][0]
        print(f"  -> Targeted search for gap: '{query}'")
    else:
        query = state["topic"]
        print(f"  -> Initial search for: '{query}'")

    papers = search_arxiv(query, max_results=num_papers, exclude_ids=seen_ids)
    print(f"  -> {len(papers)} NEW papers found (after filtering seen ones).")

    current_papers = state.get("raw_papers", [])
    return {
        "raw_papers": current_papers + papers,
        "iteration": iteration
    }


def node_read(state: ResearchState) -> dict:
    print("\n📖 [Graph] Reading and structuring new papers...")
    num_papers = state.get("num_papers", 2)
    new_papers = state["raw_papers"][-num_papers:] if len(state["raw_papers"]) >= num_papers else state["raw_papers"]

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
        context += f"--- PAPER {i} ---\nTitle: {p['title']}\nObjective: {summary.get('objective_and_scope', 'N/A')[:100]}...\n\n"

    # PERFECT LIMITATION 2: Force extreme conciseness in the intermediate step
    prompt = f"""
You are an expert academic researcher. Write a VERY CONCISE State-of-the-Art review on: "{state['topic']}".
Use the following analyzed papers as your source.

Rules:
1. Be extremely brief. Maximum 300 words total.
2. Use short bullet points.
3. Output strictly in Markdown.

Papers data:
{context}
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
    print("\n🏆 [Graph] Curating and selecting the best papers...")
    num_papers = state.get("num_papers", 2)
    all_papers = state.get("analyzed_papers", [])

    if len(all_papers) <= num_papers:
        print(f"  -> Already have {len(all_papers)} papers (<= {num_papers}). Keeping all.")
        final_papers = all_papers
    else:
        print(f"  -> Evaluating {len(all_papers)} papers to select the top {num_papers}...")

        papers_text = "\n\n".join([
            f"ID: {i}\nTitle: {p['title']}\nObjective: {p.get('structured_summary', {}).get('objective_and_scope', '')[:100]}..."
            for i, p in enumerate(all_papers)
        ])

        # PERFECT LIMITATION 3: LLM ONLY outputs a JSON array of IDs. No long text generation.
        prompt = f"""
You are an expert academic curator. Select the EXACT top {num_papers} most relevant papers from this list.

Papers:
{papers_text}

Return ONLY a valid JSON array of the IDs (integers), ordered by relevance.
Example: [2, 0]
"""
        try:
            response = llm.invoke(prompt)
            clean_response = response.content.strip().removeprefix("```json").removesuffix("```").strip()
            top_ids = json.loads(clean_response)
            final_papers = [all_papers[i] for i in top_ids if i < len(all_papers)][:num_papers]
            titles = [f"'{p['title'][:40]}...'" for p in final_papers]
            print(f"  -> Selected top {num_papers} papers: {', '.join(titles)}")
        except Exception as e:
            print(f"  -> Curation parsing failed, falling back to first {num_papers} papers. Error: {e}")
            final_papers = all_papers[:num_papers]

    # ── Save to memory ──
    new_papers = [
        {"id": p.get("arxiv_id", ""), "title": p.get("title", "Unknown")}
        for p in final_papers if p.get("arxiv_id")
    ]
    if new_papers:
        add_papers(new_papers)
        print(f"  -> 💾 Saved {len(new_papers)} papers to memory.")

    # PERFECT LIMITATION 4: Use PYTHON to build the final detailed report, NOT the LLM.
    # This costs 0 LLM tokens and guarantees no OTPM errors, while keeping all 10 fields!
    print("  -> Building final synthesis using Python formatting (0 LLM tokens)...")
    final_synthesis = f"# State-of-the-Art Review: {state['topic']}\n\n"
    final_synthesis += f"*Generated by Deep Research Agent. Selected {len(final_papers)} highly relevant papers.*\n\n---\n\n"
    
    for i, p in enumerate(final_papers, 1):
        summary = p.get("structured_summary", {})
        final_synthesis += f"## {i}. {p['title']}\n\n"
        final_synthesis += f"**Authors:** {', '.join(p['authors'])}  \n"
        final_synthesis += f"**Link:** {p['link']}\n\n"
        final_synthesis += f"### Objective and Scope\n{summary.get('objective_and_scope', 'N/A')}\n\n"
        final_synthesis += f"### Domain and Motivation\n{summary.get('domain_and_motivation', 'N/A')}\n\n"
        final_synthesis += f"### Main Tasks and Contributions\n{summary.get('main_tasks_or_contributions', 'N/A')}\n\n"
        final_synthesis += f"### Dataset and Methodology\n{summary.get('dataset_or_methodology', 'N/A')}\n\n"
        final_synthesis += f"### Proposed Framework\n{summary.get('proposed_framework', 'N/A')}\n\n"
        final_synthesis += f"### Evaluation Strategy and Metrics\n{summary.get('evaluation_strategy_and_metrics', 'N/A')}\n\n"
        final_synthesis += f"### Key Results\n{summary.get('key_results', 'N/A')}\n\n"
        final_synthesis += f"### Limitations\n{summary.get('limitations', 'N/A')}\n\n"
        final_synthesis += f"### Conclusion\n{summary.get('conclusion', 'N/A')}\n\n"
        final_synthesis += f"**Keywords:** {', '.join(summary.get('keywords', []))}\n\n---\n\n"

    return {
        "analyzed_papers": final_papers,
        "synthesis": final_synthesis
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


graph = build_research_graph()


if __name__ == "__main__":
    print("🚀 Launching the LangGraph orchestrator with Perfect Limitation...")
    app = build_research_graph()

    initial_state = {
        "topic": "argument mining",
        "num_papers": 2,
        "raw_papers": [],
        "analyzed_papers": [],
        "synthesis": "",
        "critique": "",
        "gaps": [],
        "iteration": 0,
    }

    for event in app.stream(initial_state):
        node_name = list(event.keys())[0]
        print(f"\n✅ Node '{node_name}' completed.")

    final_state = app.invoke(initial_state)
    print("\n" + "="*70)
    print("📄 FINAL CURATED SYNTHESIS")
    print("="*70)
    print(final_state["synthesis"])