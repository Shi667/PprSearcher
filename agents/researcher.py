import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from tools.arxiv_tool import search_arxiv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


load_dotenv()

@tool
def arxiv_search_tool(query: str, max_results: int = 5) -> str:
    """Search for scientific papers on arXiv based on a text query."""
    papers = search_arxiv(query, max_results)

    if not papers:
        return "No papers found for the given query."
    
    formatted_results = []
    for i, p in enumerate(papers, 1):
        formatted_results.append(
            f"[{i}] Title: {p['title']}\n"
            f"Authors: {', '.join(p['authors'])}\n"
            f"Abstract: {p['abstract'][:500]}...\n" 
            f"Link: {p['link']}"
        )
    return "\n\n---\n\n".join(formatted_results)


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
agent_llm = llm.bind_tools([arxiv_search_tool])


def run_researcher(topic: str, max_iters: int = 5) -> str:

    messages = [
        SystemMessage(content=(
            "You are an expert scientific research assistant. "
            "Your goal is to find relevant papers on arXiv about the user's topic. "
            "1. Use the arXiv search tool. "
            "2. Evaluate the relevance of the results. "
            "3. If the results are poor or empty, reformulate your query (try in English, or with more precise keywords) and search again. "
            "4. Stop when you have found at least 3 good papers OR after 3 attempts maximum. "
            "5. At the end, briefly summarize the best papers found with their links."
        )),
        HumanMessage(content=f"Find recent and relevant research papers on the following topic: {topic}")
    ]

    for iteration in range(max_iters):
        print(f"\n[Agent] Iteration {iteration + 1}/{max_iters}...")
        
        response = agent_llm.invoke(messages)
        messages.append(response) 
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"  -> Calling tool with query: '{tool_call['args'].get('query')}'")
                
                tool_result = arxiv_search_tool.invoke(tool_call["args"])
                
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))
        else:
            print("  -> Agent finished its research.")
            break

    return messages[-1].content


if __name__ == "__main__":
    print("Launching the research agent...")
    final_result = run_researcher("quantum machine learning")
    print("\n" + "="*50)
    print("AGENT FINAL RESULT:")
    print("="*50)
    print(final_result)