import requests
import fitz  # PyMuPDF
import tempfile
import os

def get_full_paper_text(arxiv_link: str) -> str:
    """
    Downloads the arXiv PDF and extracts the text.
    Limits extraction to the first 15 pages and 12,000 characters 
    to stay well within LLM context windows and keep it fast.
    """
    try:
        # Convert abstract link to PDF link 
        # e.g., "http://arxiv.org/abs/2011.00633v1" -> "https://arxiv.org/pdf/2011.00633v1.pdf"
        pdf_url = arxiv_link.replace("abs", "pdf").rstrip("/") + ".pdf"
        
        # Download the PDF
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Extract text using PyMuPDF
        doc = fitz.open(tmp_path)
        text = ""
        
        # Read up to the first 15 pages (usually contains Intro, Method, Results, Conclusion)
        max_pages = min(15, len(doc))
        for i in range(max_pages):
            text += doc[i].get_text()
        
        
        
        doc.close()
        os.unlink(tmp_path) # Clean up temp file
        
        # Cap the text length to avoid overwhelming the LLM context window
        return text[:12000] 
        
    except Exception as e:
        # Print the exact error to the console so we can debug it
        print(f"    -> ❌ PDF Fetch Error for {arxiv_link}: {str(e)}")
        return f"Error fetching or parsing PDF: {str(e)}"
        
    except Exception as e:
        return f"Error fetching or parsing PDF: {str(e)}"