import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)

def critique_synthesis(synthesis: str, topic: str) -> dict:
    """
    Evaluates the synthesis and identifies gaps that require further research.
    Returns a dictionary with 'verdict' and 'gaps' (list of specific search queries).
    """
    prompt = f"""
        You are a strict, expert academic reviewer. Your task is to evaluate the following State-of-the-Art review on the topic: "{topic}".

        Review the synthesis critically. Ask yourself:
        1. Are there obvious missing themes or sub-topics that are crucial for this subject?
        2. Are the sources diverse enough, or do they all focus on the same narrow aspect?
        3. Is the methodology or evaluation clearly explained?

        Synthesis to review:
        {synthesis}

        Return ONLY a valid JSON object with exactly these two keys:
        - "verdict": A short (2-3 sentences) professional assessment of the synthesis quality.
        - "gaps": A list of 1 to 3 specific, concise search queries (in English) that should be executed on arXiv to fill the identified gaps. If the synthesis is excellent and comprehensive, return an EMPTY list [].

        Output strictly in JSON format, no markdown, no extra text.
        """
    try:
        response = llm.invoke(prompt)
        clean_response = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean_response)
    except json.JSONDecodeError:
    
        return {"verdict": "Critic failed to parse.", "gaps": []}