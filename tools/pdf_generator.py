import markdown
import re
from xhtml2pdf import pisa
from io import BytesIO

def sanitize_text(text: str) -> str:
 
    # 1. Direct replacement of the exact black square character
    text = text.replace('■', '-')
    
    # 2. Replacement of known Unicode lookalikes and weird dashes
    text = text.replace('\u25a0', '-')  # Black square
    text = text.replace('\u25aa', '-')  # Black small square
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\uf02d', '-')  # Private use area character
    text = text.replace('\u00a0', ' ')  # Non-breaking space to normal space
    
    # 3. Fallback: Replace any other weird, non-standard printable characters
    text = re.sub(r'[^\x00-\x7F\u00C0-\u017F]+', '-', text)
    
    # 4. Clean up multiple spaces or hyphens
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'-+', '-', text)
    
    return text

def generate_pdf_from_markdown(markdown_content: str, output_path: str) -> bool:
    """
    Converts Markdown to a beautifully formatted academic PDF using xhtml2pdf.
    Pure Python, no external system libraries required!
    """
    try:
        # 1. Sanitize the text FIRST to remove ALL weird Unicode artifacts
        clean_content = sanitize_text(markdown_content)
        
        # 2. Convert Markdown to HTML
        html_content = markdown.markdown(
            clean_content,
            extensions=['tables', 'fenced_code']
        )
        
        # 3. MAGIC TRICK: Force a page break before every <h1> EXCEPT the first one.
        # This ensures the main title stays on page 1, but every subsequent paper 
        # title starts on a brand new page.
        html_content = html_content.replace('<h1>', '<h1 class="first-title">', 1)
        html_content = html_content.replace('<h1>', '<h1 style="page-break-before: always; margin-top: 40px;">')
        html_content = html_content.replace('<h1 class="first-title">', '<h1>')
        
        # 4. Wrap in a clean, academic HTML template with embedded CSS
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{ size: A4; margin: 2.5cm 2cm; }}
                body {{ 
                    font-family: 'Georgia', 'Times New Roman', serif; 
                    font-size: 11pt; 
                    line-height: 1.6; 
                    color: #1a1a1a; 
                }}
                h1 {{ 
                    font-size: 18pt; 
                    text-align: center; 
                    border-bottom: 2px solid #0F766E; 
                    padding-bottom: 10px; 
                    margin-bottom: 30px; 
                }}
                h2 {{ 
                    font-size: 14pt; 
                    color: #0F766E; 
                    margin-top: 25px; 
                    border-bottom: 1px solid #e5e5e5; 
                    padding-bottom: 5px; 
                }}
                h3 {{ 
                    font-size: 12pt; 
                    color: #333; 
                    margin-top: 20px; 
                }}
                p {{ margin-bottom: 12px; text-align: justify; }}
                strong {{ font-weight: bold; }}
                a {{ color: #0F766E; text-decoration: none; }}
                ul, ol {{ margin-bottom: 12px; padding-left: 20px; }}
                li {{ margin-bottom: 6px; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                    font-family: 'Arial', sans-serif; 
                    font-size: 10pt; 
                }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f9f9f9; font-weight: bold; }}
                hr {{ border: 0; border-top: 1px solid #ccc; margin: 30px 0; }}
                code {{ 
                    font-family: 'Courier New', monospace; 
                    background: #f4f4f4; 
                    padding: 2px 4px; 
                    border-radius: 3px; 
                    font-size: 9pt; 
                }}
                pre {{ 
                    background: #f4f4f4; 
                    padding: 10px; 
                    border-radius: 5px; 
                    overflow-x: auto; 
                    font-family: 'Courier New', monospace; 
                    font-size: 9pt; 
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # 5. Generate PDF using xhtml2pdf
        with open(output_path, "w+b") as result_file:
            pisa_status = pisa.CreatePDF(
                BytesIO(full_html.encode("utf-8")),
                dest=result_file,
                encoding="utf-8"
            )
            
        return not pisa_status.err
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return False