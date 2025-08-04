import fitz  # PyMuPDF
import re
import os
import json
from typing import List, Dict, Any, Optional

class SimpleLegalParser:
    def __init__(self, output_dir: str = "./data/clean"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Patterns to identify and skip unwanted content
        self.skip_patterns = [
            r'^\s*table\s+of\s+contents\s*$',
            r'^\s*index\s+to\s+act\s*$',
            r'^\s*arrangement\s+of\s+sections\s*$',
            r'^\s*page\s+\d+\s*$',
            r'^\s*\d+\s*$',  # Standalone numbers (often page numbers)
            r'Government\s+Gazette',
            r'^\s*_+\s*$',   # Lines of underscores
            r'^\s*={3,}\s*$' # Lines of equal signs
        ]
        
        # Start markers for actual content
        self.content_markers = [
            r'chapter\s+1',
            r'part\s+1',
            r'interpretation',
            r'preliminary',
            r'short\s+title'
        ]

    def is_skip_line(self, line: str) -> bool:
        """Check if line should be skipped"""
        if not line.strip():
            return True
        return any(re.search(pattern, line.lower()) for pattern in self.skip_patterns)

    def is_content_start(self, line: str) -> bool:
        """Check if line indicates start of main content"""
        return any(re.search(pattern, line.lower()) for pattern in self.content_markers)

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF and extract content"""
        print(f"📄 Processing: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        content = []
        in_content = False
        current_text = []
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Skip empty or unwanted lines
                if self.is_skip_line(line):
                    continue
                
                # Check for content start if we haven't found it yet
                if not in_content and self.is_content_start(line):
                    in_content = True
                
                # Once we're in content, collect text
                if in_content:
                    # Remove any repeated whitespace
                    clean_line = re.sub(r'\s+', ' ', line).strip()
                    if clean_line:
                        current_text.append(clean_line)
        
        # Join all text with proper spacing
        full_text = '\n'.join(current_text)
        
        # Clean up extra whitespace and formatting
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        
        return {
            "title": os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title(),
            "source_file": pdf_path,
            "content": full_text
        }

    def save_output(self, parsed_content: Dict[str, Any]):
        """Save parsed content to both text and JSON formats"""
        base_name = re.sub(r'[^a-z0-9]+', '_', parsed_content["title"].lower())
        
        # Save as text
        txt_path = os.path.join(self.output_dir, f"{base_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"DOCUMENT: {parsed_content['title']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(parsed_content["content"])
        print(f"📄 Saved text to: {txt_path}")
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_content, f, indent=2, ensure_ascii=False)
        print(f"📄 Saved JSON to: {json_path}")

def main():
    parser = SimpleLegalParser()
    
    # Process all PDFs in the raw directory
    raw_dir = "./data/raw"
    for filename in os.listdir(raw_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(raw_dir, filename)
            try:
                content = parser.parse_pdf(pdf_path)
                parser.save_output(content)
                print(f"✅ Successfully processed: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()